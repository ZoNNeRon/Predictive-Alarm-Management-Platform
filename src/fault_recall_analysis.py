"""
Анализ Recall по типам отказа — Fault Recall Analysis
======================================================
Доказывает, что XGBoost различает три физических сценария отказа,
а не работает по тривиальному правилу «все параметры выросли → авария».

Запускать ПОСЛЕ ml_pipeline.py (нужна обученная модель и preprocessed_pumps_dataset.csv).
fault_type сохраняется в preprocessed_pumps_dataset.csv (data_preprocessor.py его не удаляет),
если же он отсутствует — автоматически присоединяется из raw-данных.

Что доказывается:
  Recall(Critical) — доля строк state=4 данного типа, предсказанных как класс 2.
  Recall(Warning) — доля строк state=3 данного типа, предсказанных как ≥1.
  Тепловая карта сигнатур — средние значения датчиков в Critical-состоянии по типу отказа.
  Если Recall(Тип В) близок к Recall(Тип А), модель работает не по «всё выросло».
"""

import os
import sys
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config.settings import FAULT_TYPES, FAULT_LABELS
from data_preprocessor import DataPreprocessor
from visualisation_instruments import recall_plot


# Основная функция

def analyze_fault_recall(model, df_test: pd.DataFrame, feature_cols: list,
                         save_dir: str, raw_data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Считает recall модели отдельно для каждого типа отказа.

    Args:
        model:         Обученная модель (XGBoost или любая sklearn-совместимая).
        df_test:       Тестовый датасет (уже отфильтрован по pump_id).
        feature_cols:  Список признаков (FEATURE_COLS из DataPreprocessor).
        save_dir:      Директория для сохранения графика.
        raw_data_path: Путь к industrial_pumps_dataset.csv. Нужен если fault_type
                       отсутствует в df_test (автозагрузка для джойна).

    Returns:
        DataFrame с результатами по типам отказа.
    """
    # Проверка наличия fault_type
    if 'fault_type' not in df_test.columns:
        if raw_data_path and os.path.exists(raw_data_path):
            print("  [INFO] fault_type отсутствует в processed — присоединяю из raw...")
            raw_types = pd.read_csv(raw_data_path,
                                    usecols=['timestamp', 'pump_id', 'fault_type'])
            df_test = df_test.merge(raw_types, on=['timestamp', 'pump_id'], how='left')
        else:
            print("[ERROR] fault_type отсутствует и raw_data_path не указан. "
                  "Перезапустите data_preprocessor.py или передайте raw_data_path.")
            return pd.DataFrame()

    # Предсказания на тестовой выборке целиком
    preds = model.predict(df_test[feature_cols])
    df_test = df_test.copy()
    df_test['pred'] = preds

    # Вычисляем метрики по типам отказа
    results = []
    for ft in FAULT_TYPES:
        ft_mask = df_test['fault_type'] == ft

        # Critical recall (state=4 → target=2, верно если pred==2)
        crit_mask = ft_mask & (df_test['target'] == 2)
        n_crit = int(crit_mask.sum())
        recall_crit = float((df_test.loc[crit_mask, 'pred'] == 2).mean()) if n_crit else 0.0

        # Warning recall (state=3 → target=1, верно если pred>=1, т.е. угроза не пропущена)
        warn_mask = ft_mask & (df_test['target'] == 1)
        n_warn = int(warn_mask.sum())
        recall_warn = float((df_test.loc[warn_mask, 'pred'] >= 1).mean()) if n_warn else 0.0

        results.append({
            'fault_type': ft,
            'label': FAULT_LABELS[ft],
            'recall_critical': recall_crit,
            'n_critical': n_crit,
            'recall_warning': recall_warn,
            'n_warning': n_warn,
        })

    # Вывод в консоль
    print("\n" + "=" * 60)
    print("RECALL ПО ТИПАМ ОТКАЗА (тестовая выборка)")
    print("=" * 60)
    for r in results:
        det_c = int(r['recall_critical'] * r['n_critical'])
        det_w = int(r['recall_warning']  * r['n_warning'])
        print(f"\n  {r['label']:22s}:")
        print(f"    Critical  recall = {r['recall_critical']:.3f}  "
              f"({det_c}/{r['n_critical']} аварий обнаружено)")
        print(f"    Warning   recall = {r['recall_warning']:.3f}  "
              f"({det_w}/{r['n_warning']} предупреждений обнаружено)")

    res_df = pd.DataFrame(results)

    # Сигнатуры датчиков: средние значения в Critical-состоянии из df_test
    signatures = (df_test[df_test['target'] == 2]
                  .groupby('fault_type')[['vibration', 'temperature', 'current', 'pressure']]
                  .mean()
                  .reindex(FAULT_TYPES)
                  if all(c in df_test.columns for c in ['vibration', 'temperature', 'current', 'pressure'])
                  else None)

    # Если сырых значений нет в df_test, пробуем из raw
    if signatures is None and raw_data_path and os.path.exists(raw_data_path):
        pump_id = df_test['pump_id'].iloc[0] if 'pump_id' in df_test.columns else None
        raw_df = pd.read_csv(raw_data_path)
        raw_sub = raw_df[raw_df['pump_id'] == pump_id] if pump_id else raw_df
        signatures = (raw_sub[raw_sub['state'] == 4]
                      .groupby('fault_type')[['vibration', 'temperature', 'current', 'pressure']]
                      .mean()
                      .reindex(FAULT_TYPES))

    os.makedirs(save_dir, exist_ok=True)
    recall_plot(results, signatures, save_dir)

    res_df.to_csv(os.path.join(save_dir, 'ML_fault_recall_table.csv'), index=False)
    return res_df


# Точка входа (автономный запуск)

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, 'models', 'xgboost_pump_model.joblib')
    data_path = os.path.join(project_root, 'data', 'processed', 'preprocessed_pumps_dataset.csv')
    raw_path = os.path.join(project_root, 'data', 'raw', 'industrial_pumps_dataset.csv')
    save_dir = os.path.join(project_root, 'data', 'graphs')
    os.makedirs(save_dir, exist_ok=True)

    pre = DataPreprocessor(window_sizes=[15, 30, 60])
    df = pd.read_csv(data_path)
    df_test = df[df['pump_id'] == 'MNHV_005'].copy()
    model = joblib.load(model_path)

    analyze_fault_recall(model, df_test, pre.FEATURE_COLS, save_dir,    # type: ignore
                         raw_data_path=raw_path)