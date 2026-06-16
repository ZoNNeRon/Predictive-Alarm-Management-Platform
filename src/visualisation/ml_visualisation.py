"""
Блок визуализации полученных результатов тренировки ML-моделей
==============================================================
Строит и сохраняет три технических графика в светлой (академической) теме:
    1. Тепловые карты Confusion Matrix
    2. Grouped bar chart по метрикам
    3. PR-кривые
    4:
      Левая  — Recall по типам отказа (Critical и Warning)
      Правая — Тепловая карта средних значений датчиков (Critical state)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config.settings import FAULT_TYPES, FAULT_LABELS


# Палитра и стиль
PALETTE  = {'LogReg': '#4C72B0', 'RF': '#55A868', 'XGBoost': '#C44E52'}
LIGHT_BG = '#FFFFFF'
GRID_CLR = '#E5E5E5'
TEXT_CLR = '#222222'
FAULT_COLORS = {
    'overheat':   '#C44E52',
    'cavitation': '#4C72B0',
    'electrical': '#55A868',
}


def _apply_light_style(ax, title: str = ''):
    
    ax.set_facecolor(LIGHT_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.xaxis.label.set_color(TEXT_CLR)
    if title:
        ax.set_title(title, color=TEXT_CLR, fontsize=12, fontweight='bold', pad=10)


def plot_all(metrics_list: list, y_test: pd.Series, output_dir: str):

    os.makedirs(output_dir, exist_ok=True)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    # График 1: Confusion Matrices 
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(LIGHT_BG)
    fig.suptitle('Матрицы ошибок (тестовая выборка: насос MNHV_005)',
                 color=TEXT_CLR, fontsize=14, fontweight='bold', y=1.05)

    class_labels = ['Норма', 'Предупреждение', 'Авария']
    for ax, m in zip(axes, metrics_list):
        cm_norm = m['cm'].astype(float) / m['cm'].sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=m['cm'], fmt='d', ax=ax,
            cmap='Blues', linewidths=0.5, linecolor='#DDDDDD',
            xticklabels=class_labels, yticklabels=class_labels,
            cbar=False, annot_kws={'size': 11} # Убрали 'color': 'white', Seaborn настроит контраст
        )
        _apply_light_style(ax, m['label'])
        ax.set_xlabel('Предсказано', color=TEXT_CLR)
        ax.set_ylabel('Истинно', color=TEXT_CLR)
        
        # Выделяем диагональ (правильные предсказания)
        for i in range(3):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, # type: ignore
                                       edgecolor='#FF8C00', lw=2)) 

    plt.tight_layout()
    p1 = os.path.join(output_dir, 'ML_plot1_confusion_matrices.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    print(f"График 1 сохранён: {p1}")
    plt.close(fig)

    # График 2: Метрики — Grouped Bar Chart 
    metric_keys   = ['f1_macro', 'f1_critical', 'recall_critical', 'pr_auc_critical']
    metric_labels = ['F1 Macro', 'F1 (Авария)', 'Recall (Авария)', 'PR-AUC (Авария)']
    n_models  = len(metrics_list)
    n_metrics = len(metric_keys)
    x = np.arange(n_metrics)
    width = 0.22

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    colors = list(PALETTE.values())
    for i, m in enumerate(metrics_list):
        vals = [m[k] for k in metric_keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=m['label'],
                        color=colors[i], alpha=0.9, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=9, color=TEXT_CLR, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color=TEXT_CLR, fontsize=11)
    ax.set_ylim(0.35, 1.05) # Расширили верхний лимит для подписей
    ax.set_ylabel('Значение метрики', color=TEXT_CLR)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color=GRID_CLR, zorder=0)
    ax.set_title('Сравнение качества моделей по ключевым метрикам\n'
                 '(тестовая выборка: насос MNHV_005)',
                 color=TEXT_CLR, fontsize=13, fontweight='bold')
    ax.legend(framealpha=1.0, labelcolor=TEXT_CLR, fontsize=10,
              facecolor=LIGHT_BG, edgecolor='#CCCCCC')
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')

    plt.tight_layout()
    p2 = os.path.join(output_dir, 'ML_plot2_metrics_comparison.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    print(f"График 2 сохранён: {p2}")
    plt.close(fig)

    # График 3: PR-кривые для класса "Авария" 
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    for m, color in zip(metrics_list, colors):
        prec, rec, _ = precision_recall_curve(y_test_bin[:, 2], m['y_proba'][:, 2]) # type: ignore
        auc_val = m['pr_auc_critical']
        ax.plot(rec, prec, color=color, lw=2.0,
                label=f"{m['label']}  (PR-AUC = {auc_val:.4f})", zorder=3)
        # Точка при пороге 0.5
        idx_50 = np.argmin(np.abs(m['y_proba'][:, 2].mean() - 0.5))
        ax.scatter(rec[len(rec)//2], prec[len(prec)//2],
                   color=color, s=60, zorder=5, edgecolor=LIGHT_BG)

    # Базовая линия (случайный классификатор)
    baseline = y_test_bin[:, 2].mean() # type: ignore
    ax.axhline(baseline, color='#888888', lw=1.2, ls='--',
               label=f'Случайный классификатор (P = {baseline:.3f})', zorder=1)

    ax.set_xlabel('Recall (Полнота обнаружения аварий)', color=TEXT_CLR, fontsize=11)
    ax.set_ylabel('Precision (Точность предупреждений)', color=TEXT_CLR, fontsize=11)
    ax.set_title('PR-кривые для класса «Авария» (Class 2)\n'
                 'Ключевая метрика для несбалансированных промышленных данных',
                 color=TEXT_CLR, fontsize=12, fontweight='bold')
    ax.legend(framealpha=1.0, labelcolor=TEXT_CLR, fontsize=10,
              facecolor=LIGHT_BG, edgecolor='#CCCCCC')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color=GRID_CLR, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7, color=GRID_CLR, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
    ax.set_xlim([-0.02, 1.02]) # type: ignore
    ax.set_ylim([-0.02, 1.05]) # type: ignore

    plt.tight_layout()
    p3 = os.path.join(output_dir, 'ML_plot3_pr_curves_critical.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    print(f"График 3 сохранён: {p3}")
    plt.close(fig)

def recall_plot(results: list, signatures, save_dir: str):

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#FFFFFF')
    fig.suptitle(
        'Анализ Recall по типам отказа — XGBoost (тест: MNHV_005)\n'
        'Доказательство распознавания различных физических сигнатур',
        fontsize=13, fontweight='bold', y=1.02
    )

    # Панель 1 — Grouped bar chart

    ax = axes[0]
    ax.set_facecolor('#FFFFFF')
    x = np.arange(len(results))
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
        col_std = sig_vals.std(axis=0,  keepdims=True) + 1e-9
        sig_norm = (sig_vals - col_mean) / col_std

        im = ax2.imshow(sig_norm, aspect='auto', cmap='RdYlBu_r', vmin=-2, vmax=2)

        col_names = ['Вибрация\n(мм/с)', 'Темп.\n(°C)', 'Ток\n(А)', 'Давление\n(МПа)']
        ax2.set_xticks(range(len(col_names)))
        ax2.set_xticklabels(col_names, fontsize=10)
        ax2.set_yticks(range(len(FAULT_TYPES)))
        ax2.set_yticklabels([FAULT_LABELS[ft] for ft in FAULT_TYPES], fontsize=10)

        for i in range(len(FAULT_TYPES)):
            for j in range(sig_vals.shape[1]):
                val = signatures.iloc[i, j] if not pd.isna(signatures.iloc[i, j]) else 0
                z = sig_norm[i, j]
                txt_clr = 'white' if abs(z) > 1.0 else '#222222'
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
    path = os.path.join(save_dir, 'ML_plot4_fault_recall_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"График 4 сохранён: {path}")

def plot_fault_classifier(metrics_list: list, output_dir: str, best_label: str = None):  # type: ignore
    """
    Графики классификатора ТИПА отказа (вторая модель). Три фигуры:
      1. Матрицы ошибок 3x3 по моделям (overheat / cavitation / electrical).
      2. Сравнение моделей по macro-F1 и balanced accuracy (выбор модели для диплома).
      3. Раздельная точность по стадии: Recall @ Warning vs Recall @ Critical
         для лучшей модели — честно показывает, что раннее распознавание (Warning)
         менее уверенное, чем на аварии (Critical).

    Ожидает metrics_list из fault_classifier_pipeline.evaluate_fault_model():
      каждый элемент содержит 'label', 'cm', 'macro_f1', 'balanced_acc' и 'stage_recall'.
    """

    os.makedirs(output_dir, exist_ok=True)
    short_names = ['Перегрев', 'Кавитация', 'Электрика']  # компактные подписи осей

    # График 1: матрицы ошибок (нормировка по строкам, аннотация — абсолют)
    fig, axes = plt.subplots(1, len(metrics_list), figsize=(6 * len(metrics_list), 5))
    if len(metrics_list) == 1:
        axes = [axes]
    fig.patch.set_facecolor(LIGHT_BG)
    fig.suptitle('Матрицы ошибок классификатора типа отказа (тест: MNHV_005)',
                 color=TEXT_CLR, fontsize=14, fontweight='bold', y=1.05)
    for ax, m in zip(axes, metrics_list):
        cm = m['cm'].astype(float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(cm_norm, annot=m['cm'], fmt='d', ax=ax, cmap='Blues',
                    linewidths=0.5, linecolor='#DDDDDD',
                    xticklabels=short_names, yticklabels=short_names,
                    cbar=False, annot_kws={'size': 11})
        _apply_light_style(ax, m['label'])
        ax.set_xlabel('Предсказано', color=TEXT_CLR)
        ax.set_ylabel('Истинно', color=TEXT_CLR)
        for i in range(3):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,  # type: ignore
                                       edgecolor='#FF8C00', lw=2))
    plt.tight_layout()
    p1 = os.path.join(output_dir, 'ML_fault_plot1_confusion_matrix.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"График 1 (матрицы ошибок типа) сохранён: {p1}")

    # График 2: сравнение моделей (macro-F1, balanced accuracy)
    metric_keys = ['macro_f1', 'balanced_acc']
    metric_labels = ['macro-F1', 'Balanced Accuracy']
    x = np.arange(len(metric_keys))
    width = 0.8 / len(metrics_list)
    colors = list(PALETTE.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)
    for i, m in enumerate(metrics_list):
        vals = [m[k] for k in metric_keys]
        offset = (i - len(metrics_list) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=m['label'],
                      color=colors[i % len(colors)], alpha=0.9, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=TEXT_CLR)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color=TEXT_CLR, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Значение метрики', color=TEXT_CLR)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color=GRID_CLR, zorder=0)
    ax.set_title('Сравнение моделей классификации типа отказа\n(тест: насос MNHV_005)',
                 color=TEXT_CLR, fontsize=13, fontweight='bold')
    ax.legend(framealpha=1.0, labelcolor=TEXT_CLR, facecolor=LIGHT_BG, edgecolor='#CCCCCC')
    for sp in ax.spines.values():
        sp.set_edgecolor('#CCCCCC')
    plt.tight_layout()
    p2 = os.path.join(output_dir, 'ML_fault_plot2_model_comparison.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"График 2 (сравнение моделей) сохранён: {p2}")

    # График 3: раздельно Warning vs Critical (лучшая модель по macro-F1)
    if best_label is None:
        best = max(metrics_list, key=lambda m: m['macro_f1'])
    else:
        best = next((m for m in metrics_list if m['label'] == best_label), metrics_list[0])

    sr = best.get('stage_recall')
    if not sr:
        return
    x = np.arange(len(FAULT_TYPES))
    width = 0.38
    colors = [FAULT_COLORS[ft] for ft in FAULT_TYPES]
    warn_vals = [sr['warning'][ft] for ft in FAULT_TYPES]
    crit_vals = [sr['critical'][ft] for ft in FAULT_TYPES]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)
    b1 = ax.bar(x - width / 2, warn_vals, width, color=colors, alpha=0.45, hatch='//',
                edgecolor='white', label='Recall @ Warning (раннее предупреждение)')
    b2 = ax.bar(x + width / 2, crit_vals, width, color=colors, alpha=0.90,
                edgecolor='white', label='Recall @ Critical (авария)')
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if not np.isnan(h):
                ax.text(b.get_x() + b.get_width() / 2, h + 0.012, f'{h:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color=TEXT_CLR)
    ax.set_xticks(x)
    ax.set_xticklabels([FAULT_LABELS[ft] for ft in FAULT_TYPES], fontsize=10, color=TEXT_CLR)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel('Recall', color=TEXT_CLR)
    ax.axhline(1.0, color='green', ls='--', lw=1.0, alpha=0.5, label='100% Recall')
    ax.set_title(f'Точность распознавания типа по стадии — {best["label"]}\n'
                 'Раннее предупреждение (Warning) против аварии (Critical)',
                 color=TEXT_CLR, fontsize=13, fontweight='bold')
    ax.legend(framealpha=1.0, labelcolor=TEXT_CLR, facecolor=LIGHT_BG,
              edgecolor='#CCCCCC', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color=GRID_CLR, zorder=0)
    for sp in ax.spines.values():
        sp.set_edgecolor('#CCCCCC')
    plt.tight_layout()
    p3 = os.path.join(output_dir, 'ML_fault_plot3_stage_split.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"График 3 (стадии Warning/Critical) сохранён: {p3}")