"""
Блок визуализации объяснимого ИИ (XAI)
======================================
"""

import pandas as pd
import shap
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # без GUI — для серверного режима и Streamlit

# Визуализация 

def plot_waterfall(xai, feature_row: pd.DataFrame,
                    pump_id: str = 'MNHV_Unknown',
                    save_path: str = None) -> str: # type: ignore
    """
    Waterfall plot: объяснение одного конкретного предсказания.
    Показывает, как каждый признак смещает прогноз от базового значения.
    """

    shap_obj = xai.explainer(feature_row)
    explanation = shap.Explanation(
        values=shap_obj.values[0, :, xai.target_class_idx], # type: ignore
        base_values=shap_obj.base_values[0, xai.target_class_idx], # type: ignore
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
            '..', 'data', 'graphs', f'shap_plot1_waterfall_{pump_id}.png'
        )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Waterfall plot сохранён: {save_path}")
    return save_path

def plot_summary(xai, X_sample: pd.DataFrame,
                    save_path: str = None, # type: ignore
                    max_display: int = 15) -> str:
    """
    Summary (beeswarm) plot: глобальная важность признаков по всей выборке.
    Показывает не только важность (ось X), но и направление влияния
    (высокие значения признака → красный цвет → рост или падение прогноза).

    Для диплома: демонстрирует, что модель опирается на физически осмысленные
    признаки (temperature_mean_60, vibration_diff_30), а не на артефакты данных.
    """
    shap_values = xai.explainer.shap_values(X_sample) # список массивов по числу классов для XGBoost
    shap_for_critical = shap_values[xai.target_class_idx] \
        if isinstance(shap_values, list) else shap_values[:, :, xai.target_class_idx]

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
            '..', 'data', 'graphs', 'shap_plot2_summary_beeswarm.png'
        )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary plot сохранён: {save_path}")
    return save_path

def plot_summary_by_fault_type(xai, df: pd.DataFrame, feature_cols: list,
                               output_dir: str, max_display: int = 15) -> list[str]:
    """
    Строит отдельный SHAP Beeswarm для каждого типа аварии.

    Используются только аварийные строки (target == 2).

    ВАЖНО:
    Для статистической плотности допускается использование всего датасета,
    а не только тестового насоса. Beeswarm показывает структуру объяснений
    модели, а не качество её обобщения.
    """

    fault_types = [
        "overheat",
        "cavitation",
        "electrical"
    ]

    saved_files = []

    for idx, fault_type in enumerate(fault_types, start=2):

        subset = df[
            (df["target"] == 2) &
            (df["fault_type"] == fault_type)
        ]

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

        shap.summary_plot(
            shap_for_critical,
            X_subset,
            max_display=max_display,
            plot_type="dot",
            show=False
        )

        plt.title(
            f"SHAP Beeswarm — тип аварии: {fault_type}\n"
            f"Класс «Авария» | N={len(X_subset)}\n"
            f"Построено по всем аварийным строкам парка "
            f"для статистической плотности",
            fontsize=11,
            pad=12
        )

        plt.tight_layout()

        save_path = os.path.join(
            output_dir,
            f"shap_plot{idx}_beeswarm_{fault_type}.png"
        )

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Beeswarm для '{fault_type}' сохранён: "
            f"{save_path}"
        )

        saved_files.append(save_path)

    return saved_files