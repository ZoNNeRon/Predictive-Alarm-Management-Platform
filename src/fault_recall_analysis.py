"""
Анализ Recall по типам отказа — Fault Recall Analysis
======================================================
Доказывает, что XGBoost различает три физических сценария отказа,
а не работает по тривиальному правилу «все параметры выросли → авария».

Запускать ПОСЛЕ ml_pipeline.py (нужна обученная модель и processed_features.csv).
fault_type сохраняется в processed_features.csv (data_preprocessor.py его не удаляет),
если же он отсутствует — автоматически присоединяется из raw-данных.

Что доказывается:
  Recall(Critical)  — доля строк state=4 данного типа, предсказанных как класс 2.
  Recall(Warning)   — доля строк state=3 данного типа, предсказанных как ≥1.
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
from data_preprocessor import DataPreprocessor


# Константы

FAULT_TYPES  = ['overheat', 'cavitation', 'electrical']
FAULT_LABELS = {
    'overheat':   'Тип А: Перегрев',
    'cavitation': 'Тип Б: Кавитация',
    'electrical': 'Тип В: Электрика',
}
FAULT_COLORS = {
    'overheat':   '#C44E52',
    'cavitation': '#4C72B0',
    'electrical': '#55A868',
}


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
        raw_data_path: Путь к enterprise_pump_fleet.csv. Нужен если fault_type
                       отсутствует в df_test (автозагрузка для джойна).

    Returns:
        DataFrame с результатами по типам отказа.
    """
    # Гарантируем наличие fault_type
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
        n_crit      = int(crit_mask.sum())
        recall_crit = float((df_test.loc[crit_mask, 'pred'] == 2).mean()) if n_crit else 0.0

        # Warning recall (state=3 → target=1, верно если pred>=1, т.е. угроза не пропущена)
        warn_mask = ft_mask & (df_test['target'] == 1)
        n_warn      = int(warn_mask.sum())
        recall_warn = float((df_test.loc[warn_mask, 'pred'] >= 1).mean()) if n_warn else 0.0

        results.append({
            'fault_type':      ft,
            'label':           FAULT_LABELS[ft],
            'recall_critical': recall_crit,
            'n_critical':      n_crit,
            'recall_warning':  recall_warn,
            'n_warning':       n_warn,
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
        raw_df  = pd.read_csv(raw_data_path)
        raw_sub = raw_df[raw_df['pump_id'] == pump_id] if pump_id else raw_df
        signatures = (raw_sub[raw_sub['state'] == 4]
                      .groupby('fault_type')[['vibration', 'temperature', 'current', 'pressure']]
                      .mean()
                      .reindex(FAULT_TYPES))

    os.makedirs(save_dir, exist_ok=True)
    _plot(results, signatures, save_dir)

    res_df.to_csv(os.path.join(save_dir, 'fault_recall_table.csv'), index=False)
    return res_df


# Визуализация

def _plot(results: list, signatures, save_dir: str):
    """
    График 4 (две панели):
      Левая  — Recall по типам отказа (Critical и Warning)
      Правая — Тепловая карта средних значений датчиков (Critical state)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#FFFFFF')
    fig.suptitle(
        'Анализ Recall по типам отказа — XGBoost (тест: MNHV_005)\n'
        'Доказательство распознавания различных физических сигнатур',
        fontsize=13, fontweight='bold', y=1.02
    )

    # Панель 1 — Grouped bar chart

    ax    = axes[0]
    ax.set_facecolor('#FFFFFF')
    x     = np.arange(len(results))
    width = 0.32
    colors = [FAULT_COLORS[r['fault_type']] for r in results]
    labels = [r['label'] for r in results]

    bars1 = ax.bar(x - width / 2, [r['recall_critical'] for r in results], width,
                   color=colors, alpha=0.90, edgecolor='white',
                   label='Recall (Critical, target=2)')
    bars2 = ax.bar(x + width / 2, [r['recall_warning'] for r in results], width,
                   color=colors, alpha=0.45, edgecolor='white', hatch='//',
                   label='Recall (Warning, target=1)')

    for bars, key_n in [
        (bars1, 'n_critical'),
        (bars2, 'n_warning'),
    ]:
        for bar, r in zip(bars, results):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f'{h:.3f}\n(n={r[key_n]})',
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='#222222')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel('Recall', fontsize=11)
    ax.set_title('Recall по типам отказа\n(сплошные = Critical, штриховые = Warning)',
                 fontsize=11, fontweight='bold')
    ax.axhline(1.0, color='green', ls='--', lw=1.2, alpha=0.5, label='100% Recall')
    ax.legend(fontsize=9, framealpha=0.9, facecolor='#FFFFFF', edgecolor='#CCCCCC')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    for sp in ax.spines.values():
        sp.set_edgecolor('#CCCCCC')

    # Панель 2 — Тепловая карта сигнатур

    ax2 = axes[1]
    ax2.set_facecolor('#FFFFFF')

    if signatures is not None and not signatures.isna().all().all():
        sig_vals = signatures.fillna(0).values.astype(float)
        col_mean = sig_vals.mean(axis=0, keepdims=True)
        col_std  = sig_vals.std(axis=0,  keepdims=True) + 1e-9
        sig_norm = (sig_vals - col_mean) / col_std

        im = ax2.imshow(sig_norm, aspect='auto', cmap='RdYlBu_r', vmin=-2, vmax=2)

        col_names = ['Вибрация\n(мм/с)', 'Темп.\n(°C)', 'Ток\n(А)', 'Давление\n(МПа)']
        ax2.set_xticks(range(len(col_names)))
        ax2.set_xticklabels(col_names, fontsize=10)
        ax2.set_yticks(range(len(FAULT_TYPES)))
        ax2.set_yticklabels([FAULT_LABELS[ft] for ft in FAULT_TYPES], fontsize=10)

        for i in range(len(FAULT_TYPES)):
            for j in range(sig_vals.shape[1]):
                val      = signatures.iloc[i, j] if not pd.isna(signatures.iloc[i, j]) else 0
                z        = sig_norm[i, j]
                txt_clr  = 'white' if abs(z) > 1.0 else '#222222'
                ax2.text(j, i, f'{val:.1f}',
                         ha='center', va='center',
                         fontsize=12, fontweight='bold', color=txt_clr)

        plt.colorbar(im, ax=ax2, label='Z-score (отклонение от среднего по типам)',
                     shrink=0.8)
        ax2.set_title('Сигнатуры датчиков по типу отказа\n(Critical state, среднее по MNHV_005)',
                      fontsize=11, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Нет данных Critical\nдля построения сигнатур',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12)

    plt.tight_layout()
    path = os.path.join(save_dir, 'plot4_fault_recall_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"График 4 сохранён: {path}")


# Точка входа (автономный запуск)

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path   = os.path.join(project_root, 'models', 'xgboost_pump_model.joblib')
    data_path    = os.path.join(project_root, 'data', 'processed', 'processed_features.csv')
    raw_path     = os.path.join(project_root, 'data', 'raw', 'enterprise_pump_fleet.csv')
    save_dir     = os.path.join(project_root, 'data', 'graphs')
    os.makedirs(save_dir, exist_ok=True)

    pre      = DataPreprocessor(window_sizes=[15, 30, 60])
    df       = pd.read_csv(data_path)
    df_test  = df[df['pump_id'] == 'MNHV_005'].copy()
    model    = joblib.load(model_path)

    analyze_fault_recall(model, df_test, pre.FEATURE_COLS, save_dir,    # type: ignore
                         raw_data_path=raw_path)