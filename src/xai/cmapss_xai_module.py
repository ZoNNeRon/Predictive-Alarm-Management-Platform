"""
XAI валидации на NASA C-MAPSS (SHAP поверх модели тяжести)
==========================================================
src/xai/cmapss_xai_module.py

Объяснимость CMAPSS-модели тяжести тем же инструментом, что и в основном
пайплайне: XAIExplainer (src/xai/xai_module.py) переиспользуется БЕЗ
ИЗМЕНЕНИЙ - он штатно работает без модели типа отказа
(fault_model_path=None -> объяснение типа отключено, C-MAPSS не даёт метку
типа по двигателю).

Итоговый вывод - СВОДНЫЙ (философия «обобщённой подтверждённой картины»,
а не россыпи частных графиков):

  1. Консольный SymptomVector по предотказной строке каждого сабсета -
     топ-признаки с физической расшифровкой сенсоров (T50, Ps30, phi...).
  2. cmapss_summary_plot4 - усреднённая по сабсетам важность признаков
     (mean ± std): модель везде опирается на одну и ту же физику.
  3. cmapss_summary_plot5 - НАЛОЖЕННЫЙ beeswarm: SHAP-точки всех сабсетов
     на одном графике (значения признаков переведены в перцентили внутри
     своего сабсета - сырые и z-нормированные шкалы не смешиваются).

Посабсетные графики (waterfall + beeswarm) - только по флагу --detail
либо автоматически при запуске одного сабсета (материал приложения).

Предусловие: обучены и сохранены XGBoost-модели сабсетов
(python -m src.ml.cmapss_ml_pipeline all).

Запуск: python -m src.xai.cmapss_xai_module [FD001|...|all] [--detail]
"""

import argparse
import os
import sys
from typing import Tuple, cast

import pandas as pd
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings.settings_cmapss import (
    CMAPSS_SUBSETS, CMAPSS_MODELS_SUBDIR,
    CMAPSS_SENSOR_DESC, CMAPSS_STAGE_LABELS)
from src.data.cmapss_dataset import prepare_subset
from src.xai.xai_module import XAIExplainer, SymptomVector
from src.visualisation.cmapss_visualisation import (
    plot_shap_beeswarm, plot_shap_waterfall_readable,
    plot_summary_shap_importance, plot_summary_shap_beeswarm)

MODELS_DIR = os.path.join(_PROJECT_ROOT, *CMAPSS_MODELS_SUBDIR)
GRAPHS_DIR = os.path.join(_PROJECT_ROOT, 'artifacts', 'graphs')

# Строк на сабсет в сводных SHAP-фигурах (4 сабсета -> ~4800 точек суммарно)
MAX_SHAP_ROWS = 1200


def build_explainer(subset: str) -> XAIExplainer:
    """
    XAIExplainer над сохранённой XGBoost-моделью сабсета.

    fault_model_path=None: объяснение типа отказа отключено штатным путём
    xai_module (C-MAPSS не размечает тип по двигателю).
    """

    model_path = os.path.join(
        MODELS_DIR, f'cmapss_{subset.lower()}_xgboost_model.joblib')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}. Сначала обучите её: "
            f"python -m src.ml.cmapss_ml_pipeline {subset}")
    return XAIExplainer(model_path, fault_model_path=None)


def pick_pre_failure_row(test_df: pd.DataFrame,
                         fc: list) -> Tuple[pd.DataFrame, str, str, float]:
    """
    Предотказная строка для локального объяснения.

    Берётся двигатель официального теста, обрезанный ближе всех к отказу
    (минимальный финальный RUL), и его последняя строка - реальный кейс
    «за несколько циклов до разрушения».
    Возвращает (строка признаков, pump_id, cycle-как-время, RUL строки).
    """

    final_rul = test_df.groupby('pump_id')['RUL'].min()
    pid = str(final_rul.idxmin())
    # cast: pandas-stubs теряют тип DataFrame на булевом срезе
    traj = cast(pd.DataFrame, test_df[test_df['pump_id'] == pid])
    last = traj.sort_values('timestamp').iloc[-1:]
    row = cast(pd.DataFrame, last[fc])
    return row, pid, str(int(last['cycle'].iloc[0])), float(last['RUL'].iloc[0])


def print_symptom_vector(sv: SymptomVector, rul: float) -> None:
    """Консольный отчёт: класс, вероятность, топ-признаки с физикой сенсоров."""

    print(f"\n[XAI] Агрегат {sv.pump_id}, цикл {sv.timestamp} "
          f"(истинный RUL = {rul:.0f} циклов)")
    print(f"  Класс модели: {sv.predicted_class} "
          f"({CMAPSS_STAGE_LABELS[sv.predicted_class]}), "
          f"P(Авария) = {float(sv.critical_probability):.1f}%")
    print("  Топ-признаки «почему авария» (|SHAP| модели тяжести):")
    for i, s in enumerate(sv.top_symptoms, 1):
        desc = CMAPSS_SENSOR_DESC.get(s.sensor, s.sensor)
        print(f"    {i}. {s.feature} = {s.value}  SHAP {s.shap_weight:+.3f}  "
              f"[{desc}]")


def run_xai(subset: str, graphs_dir: str = GRAPHS_DIR, detail: bool = False
            ) -> Tuple[pd.Series, np.ndarray, pd.DataFrame]:
    """
    XAI-прогон сабсета: консольный отчёт + вклад в сводные фигуры.

    SHAP по предотказным строкам train считается ОДИН раз; из него выводятся
    и нормированная важность признаков (сводный график 4), и блок точек для
    наложенного beeswarm (сводный график 5). При detail=True дополнительно
    строятся пер-сабсетные waterfall и beeswarm (материал приложения).

    Возвращает (importance, shap_matrix, X_ranked):
      importance  - Series долей mean|SHAP| по признакам;
      shap_matrix - SHAP-значения класса «Авария» (n x признаки);
      X_ranked    - перцентильные ранги значений признаков внутри сабсета
                    (честная цветовая шкала при наложении разных сабсетов).
    """

    print(f"\n{'='*62}\nXAI C-MAPSS: {subset}\n{'='*62}")
    train_df, test_df, pre = prepare_subset(subset, verbose=False)
    fc = pre.FEATURE_COLS
    xai = build_explainer(subset)

    # 1. Консольное локальное объяснение: двигатель на пороге отказа
    row, pid, cycle, rul = pick_pre_failure_row(test_df, fc)
    sv = xai.explain_prediction(row, pump_id=pid, timestamp=cycle, top_k=5)
    print_symptom_vector(sv, rul)

    # 2. SHAP по предотказным строкам train (один расчёт на сабсет)
    crit = cast(pd.DataFrame, train_df[train_df['target'] == 2])
    X_crit = cast(pd.DataFrame, crit[fc])
    if len(X_crit) > MAX_SHAP_ROWS:
        X_crit = X_crit.sample(MAX_SHAP_ROWS, random_state=42)
    shap_values = xai.explainer.shap_values(X_crit)
    shap_matrix = np.asarray(
        shap_values[xai.target_class_idx] if isinstance(shap_values, list)
        else shap_values[:, :, xai.target_class_idx])

    imp = np.abs(shap_matrix).mean(axis=0)
    importance = pd.Series(imp / max(float(imp.sum()), 1e-12),
                           index=list(X_crit.columns))
    X_ranked = X_crit.rank(pct=True)

    # 3. Пер-сабсетные графики - только в detail-режиме
    if detail:
        os.makedirs(graphs_dir, exist_ok=True)
        plot_shap_waterfall_readable(xai, row, pid, subset, graphs_dir)
        plot_shap_beeswarm(xai, X_crit, subset, graphs_dir)

    return importance, shap_matrix, X_ranked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XAI (SHAP) для C-MAPSS')
    parser.add_argument('subset', nargs='?', default='all',
                        choices=list(CMAPSS_SUBSETS) + ['all'])
    parser.add_argument('--detail', action='store_true',
                        help='дополнительно пер-сабсетные waterfall/beeswarm')
    args = parser.parse_args()

    subsets = list(CMAPSS_SUBSETS) if args.subset == 'all' else [args.subset]
    # одиночный сабсет: сводные фигуры невозможны - включаются детальные
    detail = args.detail or len(subsets) == 1

    importances, shap_blocks = {}, {}
    for s in subsets:
        importances[s], sv_m, x_r = run_xai(s, detail=detail)
        shap_blocks[s] = (sv_m, x_r)

    if len(subsets) > 1:
        # Сводная объяснимость: один и тот же набор физических параметров
        # ведёт прогноз на всех сабсетах
        plot_summary_shap_importance(importances, GRAPHS_DIR)
        plot_summary_shap_beeswarm(shap_blocks, GRAPHS_DIR)
    print('\n[OK] XAI-прогон завершён.')
