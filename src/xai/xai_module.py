"""
Модуль объяснимого ИИ (Explainable AI) на базе SHAP
===================================================
Ответственность модуля: математическое объяснение прогнозов двух моделей.
 
Иерархия объяснений:
  1. Модель ТЯЖЕСТИ (severity, 0/1/2) — SHAP объясняет «почему авария»:
     какие признаки толкают агрегат к классу «Авария».
  2. Модель ТИПА отказа (overheat/cavitation/electrical) — SHAP объясняет
     «почему именно этот тип»: какие признаки определили выбор типа.
 
Модель ничего не знает об агентах, промптах и RAG — это намеренно.
На выходе — строго типизированный SymptomVector, который агентный модуль
использует для построения промпта.
"""

import pandas as pd
import numpy as np
import shap
import joblib
import os
import sys

from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config.settings import THRESHOLDS, FAULT_TYPES
from src.data.data_preprocessor import DataPreprocessor
from src.visualisation.xai_visualisation import (plot_severity_waterfall, 
                                                 plot_severity_summary_by_fault_type,
                                                 plot_fault_waterfall, 
                                                 plot_fault_summary_by_type)


# Типы данных

@dataclass
class SymptomContribution:
    """Вклад одного признака в прогноз для целевого класса."""
    
    feature: str        # техническое имя признака (например, vibration_mean_30)
    sensor: str         # физический датчик (vibration / temperature / current / pressure)
    window: str         # временное окно или тип признака (mean_15, diff_30, ...)
    value: float        # текущее значение признака
    shap_weight: float  # значение Шэпли: > 0 → толкает к аварии, < 0 → от аварии
    direction: str      # приближение к ошибке или удаление от неё

    # Порог по ГОСТ для физического датчика (None, если не применимо)
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None


@dataclass
class SymptomVector:
    """
    Структурированный вектор симптомов — выход XAIExplainer.
    Передаётся в агентный модуль для построения промпта и RAG-запроса.
    """

    pump_id: str
    timestamp: str                  # время инцидента, без значения по умолчанию
    predicted_class: int            # 0=Норма, 1=Предупреждение, 2=Авария
    probabilities: List[float]      # [P(Норма), P(Предупреждение), P(Авария)]
    critical_probability: float     # P(Авария) × 100, %
    top_symptoms: List[SymptomContribution]
    shap_base_value: float          # Базовое значение SHAP (до добавления признаков)
    inferred_fault: str = "unknown" # Диагноз, поставленный XAI
    true_fault: str = "unknown"     # Фактический диагноз из логов (для проверки)

    # Объяснение модели ТИПА отказа («почему именно этот тип»)
    fault_probabilities: List[float] = field(default_factory=list) # [P(overheat),P(cav),P(elec)]
    fault_confidence: float = 0.0 # max(fault_probabilities) × 100, %
    fault_top_symptoms: List[SymptomContribution] = field(default_factory=list)


# Основной класс

class XAIExplainer:
    """
    Математическое ядро объяснимости.

    Принимает строку признаков и возвращает SymptomVector с SHAP-значениями
    модели тяжести и (если состояние нештатное) модели типа отказа.

    Промпты, RAG, форматирование для UI — в агентном модуле.
    """

    def __init__(self, model_path: str, fault_model_path: str):
        """
        Args:
            model_path:       путь к модели ТЯЖЕСТИ (xgboost_pump_model.joblib).
            fault_model_path: путь к модели ТИПА отказа (fault_xgboost_model.joblib).
                              Если None/не найден — объяснение типа отключается
                              (inferred_fault='unknown'), модуль продолжает работать.
        """

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}. Сначала запустите ml_pipeline.py"
            )
        self.model = joblib.load(model_path)
        # TreeExplainer: оптимизирован для деревьев, не требует background data
        self.explainer = shap.TreeExplainer(self.model)
        self.target_class_idx = 2  # класс "Авария"

        # Вторая модель: тип отказа (overheat / cavitation / electrical)
        self.fault_model = None
        self.fault_explainer = None
        if fault_model_path and os.path.exists(fault_model_path):
            self.fault_model = joblib.load(fault_model_path)
            self.fault_explainer = shap.TreeExplainer(self.fault_model)
        else:
            print("[WARN] Модель типа отказа не загружена — тип будет 'unknown'. "
                  "Запустите fault_classifier_pipeline.py и передайте fault_model_path.")

    # Вспомогательное: сборка вкладов признаков из вектора SHAP

    def _build_contributions(self, feature_row: pd.DataFrame,
                             shap_vals: np.ndarray) -> List[SymptomContribution]:
        """
        Превращает вектор SHAP в отсортированный по |вкладу| список SymptomContribution.
        """

        names = feature_row.columns.tolist()
        vals = feature_row.values[0]
        out = []
        for i, fname in enumerate(names):
            sensor = fname.split('_')[0]
            w = round(float(shap_vals[i]), 4)
            out.append(SymptomContribution(
                feature=fname,
                sensor=sensor,
                window='_'.join(fname.split('_')[1:]),
                value=round(float(vals[i]), 3),
                shap_weight=w,
                direction='towards_fault' if w > 0 else 'against_fault',
                warning_threshold=THRESHOLDS.get(sensor, {}).get('warning'),
                critical_threshold=THRESHOLDS.get(sensor, {}).get('critical'),
            ))
        out.sort(key=lambda c: abs(c.shap_weight), reverse=True)
        return out
    
    def _shap_for_class(self, explainer, feature_row: pd.DataFrame, 
                        class_idx: int) -> np.ndarray:
        """
        Единое извлечение вектора SHAP для заданного класса (мультикласс XGBoost).
        """
        
        obj = explainer(feature_row)
        return obj.values[0, :, class_idx]  # type: ignore

    # Объяснение модели типа отказа

    def _explain_fault_type(self, feature_row: pd.DataFrame, top_k: int = 5) -> tuple:
        """
        Вторая модель: предсказывает тип отказа и объясняет ВЫБОР ТИПА через SHAP
        (по предсказанному классу типа).

        Returns:
            (inferred_fault, fault_probabilities, fault_confidence, fault_top_symptoms)
        """

        if self.fault_model is None:
            return "unknown", [], 0.0, []

        fault_proba = self.fault_model.predict_proba(feature_row)[0]
        fault_idx = int(np.argmax(fault_proba))
        inferred_fault = FAULT_TYPES[fault_idx]
        fault_confidence = round(float(fault_proba[fault_idx]) * 100, 1)

        shap_vals = self._shap_for_class(self.fault_explainer, feature_row, fault_idx)
        fault_top = self._build_contributions(feature_row, shap_vals)[:top_k]

        return inferred_fault, [float(p) for p in fault_proba], fault_confidence, fault_top

    # Главный метод

    def explain_prediction(self, feature_row: pd.DataFrame, pump_id: str = "Unknown",
                           timestamp: str = None, true_fault: str = "unknown",  # type: ignore
                           top_k: int = 5) -> SymptomVector:
        """
        Объясняет прогноз обеих моделей.

        timestamp обязателен: время возникновения инцидента — значимое
        диагностическое данное, заглушки недопустимы.
        """

        if timestamp is None:
            raise ValueError("explain_prediction: не передан timestamp инцидента. ")

        # 1. Модель тяжести: прогноз
        probabilities = self.model.predict_proba(feature_row)[0]
        predicted_class = int(np.argmax(probabilities))
        critical_prob = probabilities[self.target_class_idx]

        # 2. SHAP модели тяжести по классу «Авария» («почему авария»)
        base_val = (self.explainer.expected_value[self.target_class_idx]
                    if isinstance(self.explainer.expected_value, (list, np.ndarray))
                    else self.explainer.expected_value)
        shap_vals_critical = self._shap_for_class(self.explainer, feature_row, self.target_class_idx)
        top_symptoms = self._build_contributions(feature_row, shap_vals_critical)[:top_k]

        # 3. Модель типа: запускаем только на нештатных состояниях (Warning или Авария)
        inferred_fault, fault_probabilities, fault_confidence, fault_top_symptoms = \
            ("unknown", [], 0.0, [])
        if predicted_class != 0:
            (inferred_fault, fault_probabilities,
             fault_confidence, fault_top_symptoms) = self._explain_fault_type(feature_row, top_k)

        # 4. Упаковка в строго типизированный контракт
        return SymptomVector(
            pump_id=pump_id,
            timestamp=str(timestamp),
            predicted_class=predicted_class,
            probabilities=[float(p) for p in probabilities],
            critical_probability=round(critical_prob * 100, 1),
            shap_base_value=round(base_val, 4), # type: ignore
            top_symptoms=top_symptoms,
            inferred_fault=inferred_fault,
            true_fault=true_fault,
            fault_probabilities=fault_probabilities,
            fault_confidence=fault_confidence,
            fault_top_symptoms=fault_top_symptoms,
        )

# Точка входа (тестирование модуля)

if __name__ == "__main__":
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))))
    model_path = os.path.join(project_root, 'models', 'severity', 'severity_xgboost_model.joblib')
    fault_model_path = os.path.join(project_root, 'models', 'fault_type', 'fault_xgboost_model.joblib')
    data_path = os.path.join(project_root, 'data', 'processed', 'preprocessed_pumps_dataset.csv')
    plots_dir = os.path.join(project_root, 'artifacts', 'graphs')

    print("Инициализация XAIExplainer (две модели)...")
    xai = XAIExplainer(model_path, fault_model_path=fault_model_path)

    print("Загрузка тестовых данных...")
    df = pd.read_csv(data_path)

    critical_cases = df[df['target'] == 2]
    if len(critical_cases) == 0:
        print("В данных нет аварийных строк — проверьте датасет.")
        exit(1)
    if 'timestamp' not in critical_cases.columns:
        raise KeyError("В датасете нет колонки 'timestamp' — проверьте препроцессор.")

    preprocessor = DataPreprocessor(window_sizes=[15, 30, 60])
    feature_cols = preprocessor.FEATURE_COLS

    # Тест 1: объяснение одного инцидента ОБЕИМИ моделями
    sample_row = critical_cases.iloc[-1:][feature_cols]
    pump_id_val = str(critical_cases.iloc[-1]['pump_id'])
    ts_val = str(critical_cases.iloc[-1]['timestamp'])
    true_fault_val = str(critical_cases.iloc[-1].get('fault_type', 'unknown'))

    print(f"\n{'─'*55}")
    print(f"Анализ инцидента: {pump_id_val} @ {ts_val}")
    print(f"{'─'*55}")

    sv = xai.explain_prediction(sample_row, pump_id=pump_id_val,
                                timestamp=ts_val, true_fault=true_fault_val, top_k=5)

    print(f"[Модель тяжести] Класс {sv.predicted_class} "
          f"({['Норма','Warning','Авария'][sv.predicted_class]}), "
          f"P(Авария)={sv.critical_probability}%")
    print("  Топ-признаки «почему авария» (|SHAP| тяжести):")
    for i, s in enumerate(sv.top_symptoms, 1):
        print(f"    {i}. {s.feature} = {s.value}  SHAP {s.shap_weight:+.3f}")

    print(f"\n[Модель типа]   Тип: {sv.inferred_fault} (факт: {sv.true_fault}), "
          f"уверенность {sv.fault_confidence}%")
    print(f"  Вероятности типов {FAULT_TYPES}: "
          f"{[round(p, 3) for p in sv.fault_probabilities]}")
    print("  Топ-признаки «почему этот тип» (|SHAP| типа):")
    for i, s in enumerate(sv.fault_top_symptoms, 1):
        print(f"    {i}. {s.feature} = {s.value}  SHAP {s.shap_weight:+.3f}")

    # Тест 2: SHAP Waterfall модели ТЯЖЕСТИ
    print(f"\nПостроение SHAP Waterfall (модель тяжести)...")
    plot_severity_waterfall(xai, sample_row, pump_id=pump_id_val,
                   save_path=os.path.join(plots_dir, f'shap_severity_plot1_waterfall_{pump_id_val}.png'))

    # Тест 3: SHAP Beeswarm модели ТЯЖЕСТИ по типам аварий
    print("Построение SHAP Beeswarm модели тяжести по типам аварий...")
    plot_severity_summary_by_fault_type(xai, df, feature_cols, plots_dir, max_display=15)

    # Тест 4: объяснения модели ТИПА — waterfall и beeswarm по типам
    print("\nПостроение объяснений модели ТИПА отказа...")
    plot_fault_waterfall(xai, sample_row, pump_id=pump_id_val,
                         save_path=os.path.join(plots_dir, 
                                                f'shap_fault_plot1_waterfall_{pump_id_val}.png'))
    plot_fault_summary_by_type(xai, df, feature_cols, plots_dir, max_display=15)