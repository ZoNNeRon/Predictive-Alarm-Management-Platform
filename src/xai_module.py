"""
Модуль объяснимого ИИ (Explainable AI) на базе SHAP
=====================================================
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # без GUI — для серверного режима и Streamlit

from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessor import DataPreprocessor


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
    timestamp: Optional[str]
    predicted_class: int           # 0=Норма, 1=Предупреждение, 2=Авария
    probabilities: List[float]     # [P(Норма), P(Предупреждение), P(Авария)]
    critical_probability: float    # P(Авария) × 100, %
    top_symptoms: List[SymptomContribution]
    shap_base_value: float         # Базовое значение SHAP (до добавления признаков)
    all_contributions: List[SymptomContribution] = field(default_factory=list)


# Основной класс 

# Пороги ГОСТ 32601-2013 для физических датчиков
_GOST_THRESHOLDS = {
    'vibration':   {'warning': 3.0,  'critical': 8.0},
    'temperature': {'warning': 82.0, 'critical': 93.0},
    'current':     {'warning': None, 'critical': None}, # дополнить при необходимости
    'pressure':    {'warning': None, 'critical': None}, # дополнить при необходимости
}


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

    def explain(self, feature_row: pd.DataFrame,
                pump_id: str = 'MNHV_Unknown',
                timestamp: str = None, # type: ignore
                top_k: int = 5) -> SymptomVector:
        """
        Вычисляет SHAP-объяснение и возвращает SymptomVector.

        Args:
            feature_row: DataFrame из 1 строки (должен содержать FEATURE_COLS).
            pump_id: идентификатор агрегата.
            timestamp: метка времени инцидента (строка ISO).
            top_k: число главных симптомов в SymptomVector.top_symptoms.

        Returns:
            SymptomVector — структурированный результат для агентного модуля.
        """
        if len(feature_row) != 1:
            raise ValueError(
                f"explain() принимает ровно 1 строку, получено: {len(feature_row)}"
            )

        # 1. Прогноз вероятностей
        proba = self.model.predict_proba(feature_row)[0].tolist()
        pred_class = int(np.argmax(proba))
        critical_prob = round(proba[self.target_class_idx] * 100, 1)

        # 2. SHAP values
        # shap_values_obj.values: shape (1, n_features, n_classes) для XGBoost multiclass
        shap_obj = self.explainer(feature_row)
        shap_vals_all = shap_obj.values[0, :, self.target_class_idx] # для класса "Авария" # type: ignore
        base_value = float(shap_obj.base_values[0, self.target_class_idx]) # type: ignore
        feature_names = feature_row.columns.tolist()
        feature_values = feature_row.values[0]

        # 3. Список всех вкладов
        all_contribs: List[SymptomContribution] = []
        for fname, fval, sw in zip(feature_names, feature_values, shap_vals_all):
            sensor, window = _parse_feature_name(fname)
            thresh = _GOST_THRESHOLDS.get(sensor, {})
            all_contribs.append(SymptomContribution(
                feature=fname,
                sensor=sensor,
                window=window,
                value=round(float(fval), 3),
                shap_weight=round(float(sw), 4),
                direction='towards_fault' if sw > 0 else 'against_fault',
                warning_threshold=thresh.get('warning'),
                critical_threshold=thresh.get('critical'),
            ))

        # 4. Сортируем по |shap_weight|: важность, а не знак.
        #    КРИТИЧНО: признак с весом -0.8 информативнее, чем признак с +0.1.
        #    Агент должен знать оба — и "что растёт к аварии", и "что аномально снижается".
        all_contribs.sort(key=lambda c: abs(c.shap_weight), reverse=True)

        return SymptomVector(
            pump_id=pump_id,
            timestamp=timestamp,
            predicted_class=pred_class,
            probabilities=[round(p, 4) for p in proba],
            critical_probability=critical_prob,
            top_symptoms=all_contribs[:top_k],
            shap_base_value=round(base_value, 4),
            all_contributions=all_contribs,
        )

    # Визуализация 

    def plot_waterfall(self, feature_row: pd.DataFrame,
                       pump_id: str = 'MNHV_Unknown',
                       save_path: str = None) -> str: # type: ignore
        """
        Waterfall plot: объяснение одного конкретного предсказания.
        Показывает, как каждый признак смещает прогноз от базового значения.
        """
        shap_obj = self.explainer(feature_row)
        explanation = shap.Explanation(
            values=shap_obj.values[0, :, self.target_class_idx], # type: ignore
            base_values=shap_obj.base_values[0, self.target_class_idx], # type: ignore
            data=feature_row.values[0],
            feature_names=feature_row.columns.tolist()
        )

        fig, ax = plt.subplots(figsize=(12, 7))
        plt.sca(ax)
        shap.plots.waterfall(explanation, max_display=12, show=False)
        ax.set_title(
            f'SHAP Waterfall — Агрегат {pump_id}: объяснение прогноза «Авария»\n'
            f'Красные увеличивают вероятность аварии, синие - уменьшают',
            fontsize=11, pad=12
        )
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'data', 'graphs', f'shap_waterfall_{pump_id}.png'
            )
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Waterfall plot сохранён: {save_path}")
        return save_path

    def plot_summary(self, X_sample: pd.DataFrame,
                     save_path: str = None, # type: ignore
                     max_display: int = 15) -> str:
        """
        Summary (beeswarm) plot: глобальная важность признаков по всей выборке.
        Показывает не только важность (ось X), но и направление влияния
        (высокие значения признака → красный цвет → рост или падение прогноза).

        Для диплома: демонстрирует, что модель опирается на физически осмысленные
        признаки (temperature_mean_60, vibration_diff_30), а не на артефакты данных.
        """
        shap_values = self.explainer.shap_values(X_sample) # список массивов по числу классов для XGBoost
        shap_for_critical = shap_values[self.target_class_idx] \
            if isinstance(shap_values, list) else shap_values[:, :, self.target_class_idx]

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.sca(ax)
        shap.summary_plot(
            shap_for_critical, X_sample,
            max_display=max_display,
            show=False, plot_type='dot'
        )
        plt.title(
            f'SHAP Summary (Beeswarm) — Глобальная важность признаков\n'
            f'Класс «Авария» | N={len(X_sample):,} строк тестовой выборки',
            fontsize=11, pad=12
        )
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'data', 'graphs', 'shap_summary_beeswarm.png'
            )
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Summary plot сохранён: {save_path}")
        return save_path


# Точка входа (тестирование модуля) 

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, 'models', 'xgboost_pump_model.joblib')
    data_path = os.path.join(project_root, 'data', 'processed', 'processed_features.csv')
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

    sv = xai.explain(sample_row, pump_id=pump_id_val, timestamp=ts_val, top_k=5)

    print(f"Прогноз модели:   Класс {sv.predicted_class} "
          f"({['Норма','Warning','Авария'][sv.predicted_class]})")
    print(f"Вероятности:      Норма={sv.probabilities[0]:.3f}  "
          f"Warning={sv.probabilities[1]:.3f}  Авария={sv.probabilities[2]:.3f}")
    print(f"P(Авария):        {sv.critical_probability}%")
    print(f"Базовое SHAP:     {sv.shap_base_value}")
    print(f"\nТоп-{len(sv.top_symptoms)} симптомов (сортировка по |SHAP|):")
    for i, s in enumerate(sv.top_symptoms, 1):
        sign  = '+' if s.shap_weight > 0 else ''
        arrow = '↑ к аварии' if s.direction == 'towards_fault' else '↓ от аварии'
        thresh_info = ''
        if s.critical_threshold:
            thresh_info = f'  [ГОСТ Critical: {s.critical_threshold}]'
        print(f"  {i}. [{s.sensor:12s}] {s.feature}")
        print(f"     Значение: {s.value}  SHAP: {sign}{s.shap_weight}  {arrow}{thresh_info}")

    # Тест 2: Waterfall plot для одного инцидента 
    print(f"\nПостроение SHAP Waterfall plot...")
    xai.plot_waterfall(
        sample_row, pump_id=pump_id_val,
        save_path=os.path.join(plots_dir, f'shap_waterfall_{pump_id_val}.png')
    )

    # Тест 3: Summary/Beeswarm по всей тестовой выборке (насос 5) 
    print("Построение SHAP Summary (Beeswarm) plot...")
    test_pump = df[df['pump_id'] == pump_id_val][feature_cols]
    sample_for_summary = test_pump.sample(n=min(10000, len(test_pump)), random_state=42)
    xai.plot_summary(
        sample_for_summary,
        save_path=os.path.join(plots_dir, 'shap_summary_beeswarm.png')
    )