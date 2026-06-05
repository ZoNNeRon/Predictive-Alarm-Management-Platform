"""
Модуль объяснимого ИИ (Explainable AI) на базе SHAP
===================================================
Ответственность модуля: математическое объяснение прогнозов XGBoost.
Модуль ничего не знает об агентах, промптах и RAG — это намеренно.
На выходе — строго типизированный SymptomVector, который агентный
модуль использует для построения промпта.
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
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config.settings import THRESHOLDS
from data_preprocessor import DataPreprocessor
from visualisation_instruments import plot_waterfall, plot_summary_by_fault_type


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
    predicted_class: int           # 0=Норма, 1=Предупреждение, 2=Авария
    probabilities: List[float]     # [P(Норма), P(Предупреждение), P(Авария)]
    critical_probability: float    # P(Авария) × 100, %
    top_symptoms: List[SymptomContribution]
    shap_base_value: float         # Базовое значение SHAP (до добавления признаков)
    inferred_fault: str = "unknown"  # Диагноз, поставленный XAI
    true_fault: str = "unknown"      # Фактический диагноз из логов (для проверки)


# Основной класс 

_GOST_THRESHOLDS = THRESHOLDS  # пороги ГОСТ 32601-2013, централизованы в config/settings


def _parse_feature_name(feature: str):
    """
    Разбирает имя признака на составляющие.
    Например: 'vibration_mean_30' → sensor='vibration', window='mean_30'
              'temperature_diff_30' → sensor='temperature', window='diff_30'
    """
    sensors = ['vibration', 'temperature', 'current', 'pressure']
    for s in sensors:
        if feature.startswith(s):
            window = feature[len(s) + 1:] # убирает '<sensor>_'
            return s, window
    return 'unknown', feature


class XAIExplainer:
    """
    Математическое ядро объяснимости.

    Принимает строку признаков и возвращает SymptomVector с SHAP-значениями.

    Промпты, RAG, форматирование для UI — в агентном модуле.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}. Сначала запустите ml_pipeline.py"
            )
        self.model = joblib.load(model_path)
        # TreeExplainer: оптимизирован для деревьев, не требует background data
        self.explainer = shap.TreeExplainer(self.model)
        self.target_class_idx = 2 # класс "Авария"

    def explain_prediction(self, feature_row: pd.DataFrame, pump_id: str = "Unknown", 
                           true_fault: str = "unknown", top_k: int = 5) -> SymptomVector:
        """
        Анализирует строку признаков, вычисляет SHAP-веса, 
        и автоматически определяет тип физического отказа (SHAP-сигнатура).
        """
        # 1. Базовые предсказания модели
        probabilities = self.model.predict_proba(feature_row)[0]
        predicted_class = int(np.argmax(probabilities)) # 0 (Норма), 1 (Warning), 2 (Авария)
        critical_prob = probabilities[self.target_class_idx]
        
        # 2. Вычисление SHAP
        shap_values_obj = self.explainer(feature_row)
        
        # Безопасное извлечение базового значения (expected_value)
        # У XGBoost для мультикласса это обычно массив
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            base_val = self.explainer.expected_value[self.target_class_idx]
        else:
            base_val = self.explainer.expected_value
            
        shap_vals_critical = shap_values_obj.values[0, :, self.target_class_idx] # type: ignore
        
        feature_names = feature_row.columns.tolist()
        feature_values = feature_row.values[0]
        
        # 3. Сбор всех симптомов (парсинг)
        contributions = []
        for i in range(len(feature_names)):
            feat_name = feature_names[i]
            val = round(feature_values[i], 3)
            weight = round(shap_vals_critical[i], 4)
            
            # Извлекаем тип датчика (первое слово до подчеркивания)
            sensor = feat_name.split('_')[0]
            
            # Определяем направление влияния
            direction = 'towards_fault' if weight > 0 else 'against_fault'
            
            # (Опционально) Извлечение порогов ГОСТ из вашего конфига THRESHOLDS
            warning_thr = THRESHOLDS.get(sensor, {}).get('warning')
            critical_thr = THRESHOLDS.get(sensor, {}).get('critical')
            
            contributions.append(SymptomContribution(
                feature=feat_name,
                sensor=sensor,
                window='_'.join(feat_name.split('_')[1:]),
                value=val,
                shap_weight=weight,
                direction=direction,
                warning_threshold=warning_thr,
                critical_threshold=critical_thr,
            ))
            
        # 4. Сортировка по МОДУЛЮ SHAP (чтобы видеть и сильные плюсы, и сильные минусы)
        contributions.sort(key=lambda x: abs(x.shap_weight), reverse=True)
        top_symptoms = contributions[:top_k]
        
        # 5. ИНТЕЛЛЕКТУАЛЬНЫЙ АНАЛИЗ СИГНАТУРЫ (Определение типа аварии)
        inferred_fault = "unknown"
        if top_symptoms and predicted_class == 2: # Ищем причину только если это авария
            # Агрегируем суммарный положительный вклад по датчикам
            sensor_impact = {'temperature': 0.0, 'vibration': 0.0, 'current': 0.0, 'pressure': 0.0}
            for s in top_symptoms:
                if s.shap_weight > 0:
                    sensor_impact[s.sensor] += s.shap_weight
                    
            # Дерево решений (Физика процессов)
            if sensor_impact['temperature'] > 0.3:
                inferred_fault = 'overheat'
            elif sensor_impact['pressure'] > 0.2:
                inferred_fault = 'cavitation'
            elif sensor_impact['current'] > sensor_impact['vibration']:
                inferred_fault = 'electrical'
            elif sensor_impact['vibration'] > 0:
                inferred_fault = 'cavitation'

        # 6. Упаковка в строго типизированный контракт (dataclass)
        return SymptomVector(
            pump_id=pump_id,
            predicted_class=predicted_class,
            probabilities=[float(p) for p in probabilities],
            critical_probability=round(critical_prob * 100, 1),
            shap_base_value=round(base_val, 4),     # type: ignore
            top_symptoms=top_symptoms,
            inferred_fault=inferred_fault,
            true_fault=true_fault
        )


# Точка входа (тестирование модуля) 

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, 'models', 'xgboost_pump_model.joblib')
    data_path = os.path.join(project_root, 'data', 'processed', 'preprocessed_pumps_dataset.csv')
    plots_dir = os.path.join(project_root, 'data', 'graphs')

    print("Инициализация XAIExplainer...")
    try:
        xai = XAIExplainer(model_path)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    print("Загрузка тестовых данных...")
    df = pd.read_csv(data_path)

    critical_cases = df[df['target'] == 2]
    if len(critical_cases) == 0:
        print("В данных нет аварийных строк — проверьте датасет.")
        exit(1)

    preprocessor = DataPreprocessor(window_sizes=[15, 30, 60])
    feature_cols = preprocessor.FEATURE_COLS

    # Тест 1: объяснение одного инцидента 
    sample_row = critical_cases.iloc[-1:][feature_cols]
    pump_id_val = critical_cases.iloc[-1]['pump_id']
    ts_val = str(critical_cases.iloc[-1].get('timestamp', 'N/A'))

    print(f"\n{'─'*55}")
    print(f"Анализ инцидента: {pump_id_val} @ {ts_val}")
    print(f"{'─'*55}")

    pump_id_val = critical_cases.iloc[-1]['pump_id']
    # Извлекаем истинный тип аварии из датасета
    true_fault_val = critical_cases.iloc[-1].get('fault_type', 'unknown')
    
    # Получаем математическое объяснение
    sv = xai.explain_prediction(sample_row, pump_id=pump_id_val, true_fault=true_fault_val, top_k=5)

    print(f"Прогноз модели:   Класс {sv.predicted_class} "
          f"({['Норма','Warning','Авария'][sv.predicted_class]})")
    print(f"Диагноз XAI:      {sv.inferred_fault} (Фактически: {sv.true_fault})")
    print(f"Вероятности:      Норма={sv.probabilities[0]:.3f}  "
          f"Warning={sv.probabilities[1]:.3f}  Авария={sv.probabilities[2]:.3f}")
    print(f"P(Авария):        {sv.critical_probability}%")
    print(f"Базовое SHAP:     {sv.shap_base_value}")
    print(f"\nТоп-{len(sv.top_symptoms)} симптомов (сортировка по |SHAP|):")
    for i, s in enumerate(sv.top_symptoms, 1):
        sign  = '+' if s.shap_weight > 0 else ''
        arrow = '↑ к аварии' if s.direction == 'towards_fault' else '↓ от аварии'
        thresh_info = ''
        if s.critical_threshold and s.value >= s.critical_threshold:
            thresh_info = f'  [ПРЕВЫШЕН ГОСТ Critical: {s.critical_threshold}]'
        elif s.critical_threshold:
            thresh_info = f'  [порог ГОСТ: {s.critical_threshold}]'
        print(f"  {i}. [{s.sensor:12s}] {s.feature}")
        print(f"     Значение: {s.value}  SHAP: {sign}{s.shap_weight}  {arrow}{thresh_info}")

    # Тест 2: Waterfall plot для одного инцидента 
    print(f"\nПостроение SHAP Waterfall plot...")
    plot_waterfall(xai, 
        sample_row, pump_id=pump_id_val,
        save_path=os.path.join(plots_dir, f'shap_plot1_waterfall_{pump_id_val}.png')
    )

    # Тест 3: Summary/Beeswarm по всей тестовой выборке (насос 5) 
    print("Построение отдельных SHAP Beeswarm для каждого типа аварии...")
    plot_summary_by_fault_type(xai, df, feature_cols, plots_dir, max_display=15)