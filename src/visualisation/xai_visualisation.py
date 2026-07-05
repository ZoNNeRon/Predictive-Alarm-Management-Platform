"""
Блок визуализации объяснимого ИИ (XAI)
======================================
src/visualisation/xai_visualisation.py

Строит SHAP-графики для двух моделей иерархического классификатора, обёрнутых
объектом XAIExplainer:

- Модель ТЯЖЕСТИ (xai.explainer - shap.TreeExplainer над XGBoost
  severity). Объясняет прогноз класса «Авария» (xai.target_class_idx);
- Модель ТИПА (xai.fault_explainer). Объясняет, почему классификатор
  отнёс инцидент к конкретному типу отказа (overheat / cavitation / electrical,
  config.settings.FAULT_TYPES / FAULT_LABELS). 

Два вида графиков:

- Waterfall - локальное объяснение одного предсказания (вклад каждого
  признака от базового значения);
- Beeswarm (summary) - глобальная важность и направление влияния признаков
  по выборке.

Бэкенд matplotlib принудительно Agg (без GUI) - для серверного прогона
скриптов и для рендера SHAP-вкладки в Streamlit (PlatformBackend).
"""

from typing import Optional
import numpy as np
import pandas as pd
import shap
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # без GUI - для серверного режима и Streamlit

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config.settings import FAULT_TYPES, FAULT_LABELS

# Визуализация 

def plot_severity_waterfall(xai, feature_row: pd.DataFrame,
                            pump_id: str = 'MNHV_Unknown',
                            save_path: Optional[str] = None) -> str:
    """
    Waterfall plot: объяснение одного конкретного предсказания.
    Показывает, как каждый признак смещает прогноз от базового значения.
    """

    shap_obj = xai.explainer(feature_row)
    values = np.asarray(shap_obj.values)
    base_values = np.asarray(shap_obj.base_values)
    explanation = shap.Explanation(
        values=values[0, :, xai.target_class_idx],
        base_values=base_values[0, xai.target_class_idx],
        data=feature_row.values[0],
        feature_names=feature_row.columns.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.sca(ax)
    shap.plots.waterfall(explanation, max_display=12, show=False)
    ax.set_title(
        f'SHAP Waterfall (модель ТЯЖЕСТИ) - «Авария»\n'
        f'Агрегат {pump_id} | Почему модель спрогнозировала аварию\n'
        f'Красные увеличивают вероятность аварии, синие - уменьшают',
        fontsize=14, pad=12
    )
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(_PROJECT_ROOT, 'artifacts', 'graphs',
                                 f'shap_severity_plot1_waterfall_{pump_id}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Waterfall plot сохранён: {save_path}")
    return save_path


def plot_severity_summary_by_fault_type(xai, df: pd.DataFrame, feature_cols: list,
                                        output_dir: str, max_display: int = 15) -> list[str]:
    """
    Строит отдельный SHAP Beeswarm для каждого типа аварии.

    Используются только аварийные строки (target == 2).

    ВАЖНО:
    Для статистической плотности допускается использование всего датасета,
    а не только тестового насоса. Beeswarm показывает структуру объяснений
    модели, а не качество её обобщения.
    """

    saved_files = []

    for idx, fault_type in enumerate(FAULT_TYPES, start=2):

        subset = df[(df["target"] == 2) & (df["fault_type"] == fault_type)]

        if len(subset) == 0:
            print(
                f"[WARNING] Нет аварийных строк для "
                f"fault_type='{fault_type}'"
            )
            continue

        X_subset = subset[feature_cols]
        shap_values = xai.explainer.shap_values(X_subset)
        shap_for_critical = (
            shap_values[xai.target_class_idx]
            if isinstance(shap_values, list)
            else shap_values[:, :, xai.target_class_idx]
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.sca(ax)
        shap.summary_plot(shap_for_critical, X_subset,
            max_display=max_display, plot_type="dot", show=False
        )
        plt.title(
            f'SHAP Beeswarm (модель ТЯЖЕСТИ) - {FAULT_LABELS[fault_type]}\n'
            f'Признаки прогноза «Авария» для этого типа | N={len(X_subset)}',
            fontsize=14,
            pad=12
        )
        plt.tight_layout()
        save_path = os.path.join(output_dir,
            f"shap_severity_plot{idx}_beeswarm_{fault_type}.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(
            f"Beeswarm для '{fault_type}' сохранён: "
            f"{save_path}"
        )
        saved_files.append(save_path)
    return saved_files


def plot_fault_waterfall(xai, feature_row: pd.DataFrame,
                         pump_id: str = 'MNHV_Unknown',
                         save_path: Optional[str] = None) -> Optional[str]:
    """
    Waterfall МОДЕЛИ ТИПА: объясняет, почему классификатор выбрал именно этот тип
    отказа для конкретного инцидента (SHAP по предсказанному классу типа).

    Дополняет waterfall модели тяжести: тот отвечает «почему авария»,
    этот - «почему именно перегрев/кавитация/электрика».
    """

    if getattr(xai, 'fault_explainer', None) is None:
        print("[WARN] Модель типа не загружена - waterfall типа пропущен.")
        return None

    # Индекс класса и сам fault_type (например, 'overheat')
    fault_idx = int(xai.fault_model.predict(feature_row)[0])
    fault_type = FAULT_TYPES[fault_idx]

    shap_obj = xai.fault_explainer(feature_row)
    values = np.asarray(shap_obj.values)
    base_values = np.asarray(shap_obj.base_values)
    explanation = shap.Explanation(
        values=values[0, :, fault_idx],
        base_values=base_values[0, fault_idx],
        data=feature_row.values[0],
        feature_names=feature_row.columns.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.sca(ax)
    shap.plots.waterfall(explanation, max_display=12, show=False)
    
    # Единообразный заголовок с использованием FAULT_LABELS[fault_type]
    ax.set_title(
        f'SHAP Waterfall (модель ТИПА) - {FAULT_LABELS[fault_type]}\n'
        f'Агрегат {pump_id} | Почему классификатор выбрал именно этот тип\n'
        f'Красные увеличивают вероятность этого типа, синие - уменьшают',
        fontsize=14, pad=12
    )
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(_PROJECT_ROOT, 'artifacts', 'graphs',
                                 f'shap_fault_plot1_waterfall_{pump_id}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Waterfall типа сохранён: {save_path}")
    return save_path


def plot_fault_summary_by_type(xai, df: pd.DataFrame, feature_cols: list,
                               output_dir: str, max_display: int = 15) -> list:
    """
    Для каждого типа отказа - SHAP Beeswarm МОДЕЛИ ТИПА по его собственному классу.
    Показывает, на какие признаки опирается классификатор, отделяя данный тип
    от остальных (глобальная сигнатура решающих признаков).

    Дополняет plot_summary_by_fault_type (которая объясняет модель ТЯЖЕСТИ):
    здесь объясняется модель ТИПА. Берутся все нештатные строки (target != 0),
    т.е. область определения модели типа.
    """

    if getattr(xai, 'fault_explainer', None) is None:
        print("[WARN] Модель типа не загружена - beeswarm типа пропущен.")
        return []

    saved_files = []
    for index, fault_type in enumerate(FAULT_TYPES, start=2):
        subset = df[(df['target'] != 0) & (df['fault_type'] == fault_type)]
        if len(subset) == 0:
            print(f"[WARNING] Нет нештатных строк для fault_type='{fault_type}'")
            continue

        X_subset = subset[feature_cols]
        cls_idx = FAULT_TYPES.index(fault_type)

        shap_values = xai.fault_explainer.shap_values(X_subset)
        shap_for_type = (shap_values[cls_idx] if isinstance(shap_values, list)
                         else shap_values[:, :, cls_idx])

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.sca(ax)
        shap.summary_plot(shap_for_type, X_subset, max_display=max_display,
                          plot_type='dot', show=False)
        plt.title(
            f'SHAP Beeswarm (модель ТИПА) - {FAULT_LABELS[fault_type]}\n'
            f'Признаки, по которым классификатор отделяет этот тип | N={len(X_subset)}',
            fontsize=14, pad=12
        )
        plt.tight_layout()

        save_path = os.path.join(output_dir, 
                                 f'shap_fault_plot{index}_beeswarm_{fault_type}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Beeswarm типа для '{fault_type}' сохранён: {save_path}")
        saved_files.append(save_path)

    return saved_files