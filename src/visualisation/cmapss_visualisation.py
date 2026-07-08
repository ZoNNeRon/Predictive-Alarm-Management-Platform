"""
Блок визуализации валидации на NASA C-MAPSS
===========================================
src/visualisation/cmapss_visualisation.py

Графики ветки CMAPSS-валидации (модель тяжести на реальных данных).
Отдельный модуль, а не расширение ml_visualisation: у насосных функций
определены подписи («тест: насос MNHV_005») и имена файлов ML_plot* -
переиспользование затёрло бы графики основного пайплайна. Стиль -светлая
академическая тема, палитра моделей.

Функции (имена PNG - cmapss_{fd}_plot*):

- plot_confusion_matrices(metrics_list, ...) - матрицы ошибок 3 моделей;
- plot_metrics_by_split(summary_df, ...) - метрики official vs holdout
  (прямой ответ на вопрос о дисбалансе официального теста);
- plot_pr_curves(metrics_list, y_test, ...) - PR-кривые класса «Авария»;
- plot_lead_time(lead_df, ...) - за сколько циклов до отказа
  модель впервые дала Предупреждение/Аварию (главный график валидации);
- plot_shap_beeswarm(xai, X, ...) - глобальная SHAP-картина признаков по 
  классу «Авария»;
- plot_shap_waterfall_readable(xai, row, ...) - локальное объяснение с
  физическими кодами сенсоров (Ps30, phi, T50 вместо s11/s12/s4).

СВОДНЫЙ СЛОЙ (главные фигуры для текста диплома, cmapss_summary_plot*) -
обобщение по ВСЕМ сабсетам в духе LOGO-CV-дашборда (mean ± std), чтобы
показывать комиссии подтверждённую усреднённую картину, а не 4 x 8 частных
графиков (те остаются материалом приложения):

- plot_summary_generalisation(summary_df) - сравнение LogReg/RF/XGBoost:
  mean ± std метрик по 4 сабсетам, панели holdout/official друг над другом;
- plot_summary_pr_curves(curves_df, summary_df) - усреднённые PR-кривые
  класса «Авария» (средняя precision по сабсетам, PR-AUC mean ± std
  в легенде) - аналог насосного ML_plot3;
- plot_summary_confusion(cm_df) - матрицы ошибок сплит x модель,
  агрегированные СУММОЙ счётчиков по сабсетам (не «средняя матрица»);
- plot_summary_lead_time({subset: lead_df}) - упреждение обнаружения по
  всем сабсетам (боксплоты) против границ разметки;
- plot_summary_shap_importance({subset: imp}) - усреднённая по сабсетам
  важность признаков: модель везде опирается на одну и ту же физику;
- plot_summary_shap_beeswarm({subset: (sv, X)}) - НАЛОЖЕННЫЙ beeswarm:
  SHAP-точки всех сабсетов на одном графике (цвет - перцентиль значения
  признака внутри своего сабсета, шкалы сабсетов не смешиваются).

Все входы - готовые объекты вызывающей стороны (metrics-словари
severity_classifier_pipeline.evaluate_model, XAIExplainer, DataFrame
lead-time) - модуль не ходит в данные сам (SRP, как у остальных vis-блоков).
Исключение - __main__: сводные фигуры 1-3 и 6 пересобираются из сохранённых
artifacts/tables/cmapss_*.csv без переобучения моделей.
"""

import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # без GUI - серверный прогон
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import shap
from sklearn.metrics import precision_recall_curve

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config.settings.settings_cmapss import (CMAPSS_GRAPHS_PREFIX,
                                             CMAPSS_SUBSET_INFO,
                                             CMAPSS_SENSOR_SHORT,
                                             CMAPSS_TABLES_PREFIX,
                                             RUL_WARNING, RUL_CRITICAL)

# Палитра и стиль - те же, что в ml_visualisation
PALETTE = {'LogReg': '#4C72B0', 'RF': '#55A868', 'XGBoost': '#C44E52'}
LIGHT_BG = '#FFFFFF'
GRID_CLR = '#E5E5E5'
TEXT_CLR = '#222222'
CLASS_LABELS = ['Норма', 'Предупреждение', 'Авария']

SPLIT_RU = {'official': 'официальный тест NASA',
            'holdout': 'holdout по двигателям'}

# СВОДНЫЙ СЛОЙ: обобщение по всем сабсетам (главные фигуры диплома)

# Палитра метрик - как в LOGO-CV-дашборде (та же сине-серая гамма)
C_F1, C_REC, C_PR = '#33506E', '#A9C0D6', '#6B8CAE'
SUMMARY_METRICS = (
    ('f1_macro', 'F1 Macro', C_F1),
    ('recall_critical', 'Recall (Авария)', C_REC),
    ('pr_auc_critical', 'PR-AUC (Авария)', C_PR),
)

MODEL_ORDER = ('LogReg', 'RF', 'XGBoost')


def _fname(subset: str, tail: str) -> str:
    """Название графика при сохранении его в папку."""
    return f'{CMAPSS_GRAPHS_PREFIX}_{subset.lower()}_{tail}.png'


def display_feature_name(col: str) -> str:
    """
    Техническое имя признака -> читаемое: 's11_mean_5' -> 'Ps30 | mean_5'.

    Физические коды сенсоров (Ps30, phi, T50...) - из CMAPSS_SENSOR_SHORT;
    без них SHAP-подписи s2/s11/s17 нечитаемы для не знакомого с датасетом.
    """

    sensor, _, stat = col.partition('_')
    return f"{CMAPSS_SENSOR_SHORT.get(sensor, sensor)} | {stat}"


def display_feature_names(cols) -> list:
    """Формирование списка признаков."""
    return [display_feature_name(c) for c in cols]


def _subset_title(subset: str) -> str:
    """Наименование сабсета данных."""
    return CMAPSS_SUBSET_INFO[subset]['label']


def _style_axis(ax):
    """Задание стиля для графиков."""
    ax.set_facecolor(LIGHT_BG)
    ax.tick_params(colors=TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')


# График 1: матрицы ошибок трёх моделей

def plot_confusion_matrices(metrics_list: List[dict], subset: str,
                            split: str, save_dir: str) -> str:
    """Нормированные по строкам матрицы ошибок, аннотация - абсолюты."""

    fig, axes = plt.subplots(1, len(metrics_list), figsize=(16, 5))
    fig.patch.set_facecolor(LIGHT_BG)
    fig.suptitle(f'Матрицы ошибок - {_subset_title(subset)}\n'
                 f'(реальные данные NASA C-MAPSS, {SPLIT_RU.get(split, split)})',
                 color=TEXT_CLR, fontsize=20, fontweight='bold', y=1.07)

    for ax, m in zip(np.atleast_1d(axes), metrics_list):
        cm = m['cm'].astype(float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(cm_norm, annot=m['cm'], fmt='d', ax=ax, cmap='Blues',
                    linewidths=0.5, linecolor='#DDDDDD',
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                    cbar=False, annot_kws={'size': 15, 'weight': 'bold'})
        _style_axis(ax)
        ax.set_title(m['label'], color=TEXT_CLR, fontsize=16, fontweight='bold')
        ax.set_xlabel('Предсказано', color=TEXT_CLR, fontsize=12)
        ax.set_ylabel('Истинно', color=TEXT_CLR, fontsize=12)
        ax.tick_params(labelsize=10)
        for i in range(3):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=False,
                                   edgecolor='#FF8C00', lw=2))

    plt.tight_layout()
    path = os.path.join(save_dir, _fname(subset, f'plot1_confusion_{split}'))
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"График 1 (confusion, {split}) сохранён: {path}")
    return path


# График 2: метрики official vs holdout

def plot_metrics_by_split(summary_df: pd.DataFrame, subset: str,
                          save_dir: str) -> Optional[str]:
    """
    Сгруппированные столбцы метрик по сплитам.

    Прямая иллюстрация к анализу дисбаланса: official-тест бенчмарка
    (обрезанные траектории, редкие позитивы) против сбалансированного
    holdout по двигателям. Ожидаемый вид: просадка на official при
    сохранении уровня на holdout.
    """

    d = summary_df[summary_df['subset'] == subset]
    splits = [s for s in ('official', 'holdout') if s in set(d['split'])]
    if not splits:
        print(f"[WARN] Нет строк сводки для {subset} - график 2 пропущен.")
        return None

    metric_keys = ['f1_macro', 'f1_critical', 'recall_critical', 'pr_auc_critical']
    metric_labels = ['F1 Macro', 'F1 (Авария)', 'Recall (Авария)', 'PR-AUC (Авария)']

    fig, axes = plt.subplots(1, len(splits), figsize=(7.5 * len(splits), 5.5),
                             sharey=True)
    fig.patch.set_facecolor(LIGHT_BG)
    fig.suptitle(f'Качество модели тяжести - {_subset_title(subset)}\n'
                 f'официальный тест NASA против сбалансированного holdout',
                 color=TEXT_CLR, fontsize=17, fontweight='bold', y=1.04)

    x = np.arange(len(metric_keys))
    for ax, split in zip(np.atleast_1d(axes), splits):
        _style_axis(ax)
        dd = d[d['split'] == split]
        models = list(dd['model'])
        width = 0.8 / max(len(models), 1)
        for i, (_, row) in enumerate(dd.iterrows()):
            vals = [row[k] for k in metric_keys]
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=row['model'],
                          color=PALETTE.get(row['model'], '#888888'),
                          alpha=0.9, zorder=3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=9,
                        fontweight='bold', color=TEXT_CLR, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, color=TEXT_CLR, fontsize=11)
        ax.set_ylim(0.4, 1.12)
        ax.set_title(SPLIT_RU.get(split, split), color=TEXT_CLR,
                     fontsize=14, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, color=GRID_CLR, zorder=0)
        ax.legend(framealpha=1.0, labelcolor=TEXT_CLR, fontsize=10,
                  facecolor=LIGHT_BG, edgecolor='#CCCCCC', loc='lower right')
    np.atleast_1d(axes)[0].set_ylabel('Значение метрики',
                                      color=TEXT_CLR, fontsize=12)

    plt.tight_layout()
    path = os.path.join(save_dir, _fname(subset, 'plot2_metrics_by_split'))
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"График 2 (метрики по сплитам) сохранён: {path}")
    return path


# График 3: PR-кривые класса «Авария»

def plot_pr_curves(metrics_list: List[dict], y_test: pd.Series, subset: str,
                   split: str, save_dir: str) -> str:
    """PR-кривые класса «Авария» трёх моделей + базовая линия."""

    y_bin = (np.asarray(y_test) == 2).astype(int)
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(LIGHT_BG)
    _style_axis(ax)

    for m in metrics_list:
        prec, rec, _ = precision_recall_curve(y_bin, m['y_proba'][:, 2])
        ax.plot(rec, prec, color=PALETTE.get(m['label'], '#888888'), lw=2.0,
                label=f"{m['label']}  (PR-AUC = {m['pr_auc_critical']:.4f})",
                zorder=3)

    baseline = float(y_bin.mean())
    ax.axhline(baseline, color='#888888', lw=1.2, ls='--',
               label=f'Случайный классификатор (P = {baseline:.3f})', zorder=1)

    ax.set_xlabel('Recall (полнота обнаружения предотказных строк)',
                  color=TEXT_CLR, fontsize=13)
    ax.set_ylabel('Precision (точность тревог)', color=TEXT_CLR, fontsize=13)
    ax.set_title(f'PR-кривые класса «Авария» - {_subset_title(subset)}\n'
                 f'({SPLIT_RU.get(split, split)})',
                 color=TEXT_CLR, fontsize=14, fontweight='bold')
    ax.legend(framealpha=1.0, labelcolor=TEXT_CLR, fontsize=10,
              facecolor=LIGHT_BG, edgecolor='#CCCCCC')
    ax.grid(True, linestyle='--', alpha=0.7, color=GRID_CLR, zorder=0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    path = os.path.join(save_dir, _fname(subset, f'plot3_pr_curves_{split}'))
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"График 3 (PR-кривые, {split}) сохранён: {path}")
    return path


# График 4: lead time - упреждение обнаружения

def plot_lead_time(lead_df: pd.DataFrame, subset: str, save_dir: str,
                   model_label: str = 'XGBoost') -> str:
    """
    За сколько циклов до реального отказа модель впервые подняла тревогу.

    Вход - DataFrame c колонками lead_warning / lead_critical (RUL в момент
    первого срабатывания соответствующей стадии) по каждому двигателю
    holdout-теста (траектории дожиты до отказа). Пунктиры - границы разметки
    RUL_WARNING/RUL_CRITICAL: столбик выше границы = модель сработала РАНЬШЕ,
    чем требует разметка.
    """

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(LIGHT_BG)
    fig.suptitle(f'Упреждение обнаружения деградации - {_subset_title(subset)}\n'
                 f'{model_label}, двигатели holdout-теста '
                 f'(n={len(lead_df)}, все дожиты до отказа)',
                 color=TEXT_CLR, fontsize=15, fontweight='bold', y=1.04)

    panels = (
        ('lead_warning', 'Первое «Предупреждение»', RUL_WARNING, '#E0A800'),
        ('lead_critical', 'Первая «Авария»', RUL_CRITICAL, '#C62828'),
    )
    for ax, (col, title, bound, color) in zip(axes, panels):
        _style_axis(ax)
        vals = lead_df[col].dropna()
        missed = len(lead_df) - len(vals)
        ax.hist(vals, bins=24, color=color, alpha=0.75,
                edgecolor='white', zorder=3)
        ax.axvline(bound, color=TEXT_CLR, ls='--', lw=1.6,
                   label=f'Граница разметки ({bound} циклов)', zorder=4)
        med = float(vals.median()) if len(vals) else float('nan')
        if len(vals):
            ax.axvline(med, color=color, ls='-', lw=2.0,
                       label=f'Медиана: {med:.0f} циклов до отказа', zorder=4)
        ax.set_title(title + (f'  (пропущено: {missed})' if missed else ''),
                     color=TEXT_CLR, fontsize=13, fontweight='bold')
        ax.set_xlabel('Циклов до отказа при первом срабатывании',
                      color=TEXT_CLR, fontsize=11)
        ax.set_ylabel('Двигателей', color=TEXT_CLR, fontsize=11)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, color=GRID_CLR, zorder=0)
        ax.legend(framealpha=1.0, labelcolor=TEXT_CLR, fontsize=9,
                  facecolor=LIGHT_BG, edgecolor='#CCCCCC')

    plt.tight_layout()
    path = os.path.join(save_dir, _fname(subset, 'plot4_lead_time'))
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"График 4 (lead time) сохранён: {path}")
    return path


# График 5: SHAP beeswarm по классу «Авария»

def plot_shap_beeswarm(xai, X: pd.DataFrame, subset: str, save_dir: str,
                       max_display: int = 15, max_rows: int = 2000) -> str:
    """
    Глобальная SHAP-картина: какие признаки двигают прогноз к «Аварии».

    Generic-аналог насосного beeswarm по типам отказа: C-MAPSS не даёт метку
    типа по двигателю, поэтому строится один сводный график по выборке
    предотказных строк (target == 2 у вызывающей стороны).
    """

    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=42)
    shap_values = xai.explainer.shap_values(X)
    shap_critical = (shap_values[xai.target_class_idx]
                     if isinstance(shap_values, list)
                     else shap_values[:, :, xai.target_class_idx])

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.sca(ax)
    # SHAP считается на исходных именах (контракт модели), подписи -
    # читаемые физические коды (Ps30 | mean_5)
    shap.summary_plot(shap_critical, X, max_display=max_display,
                      feature_names=display_feature_names(X.columns),
                      plot_type='dot', show=False)
    plt.title(f'SHAP Beeswarm (модель ТЯЖЕСТИ) - {_subset_title(subset)}\n'
              f'Вклад признаков в класс «Авария», N={len(X)} предотказных строк',
              fontsize=14, pad=12)
    plt.tight_layout()

    path = os.path.join(save_dir, _fname(subset, 'plot5_shap_beeswarm'))
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"График 5 (SHAP beeswarm) сохранён: {path}")
    return path


# График 6: SHAP waterfall с читаемыми именами признаков

def plot_shap_waterfall_readable(xai, feature_row: pd.DataFrame, pump_id: str,
                                 subset: str, save_dir: str) -> str:
    """
    Локальное SHAP-объяснение предотказной строки, подписи - физика.

    Аналог xai_visualisation.plot_severity_waterfall, но имена признаков
    заменены на читаемые коды сенсоров (SHAP считается на исходных именах -
    контракт обученной модели не трогается, подменяются только подписи).
    """

    shap_obj = xai.explainer(feature_row)
    values = np.asarray(shap_obj.values)
    base_values = np.asarray(shap_obj.base_values)
    explanation = shap.Explanation(
        values=values[0, :, xai.target_class_idx],
        base_values=base_values[0, xai.target_class_idx],
        data=feature_row.values[0],
        feature_names=display_feature_names(feature_row.columns))

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.sca(ax)
    shap.plots.waterfall(explanation, max_display=12, show=False)
    ax.set_title(
        f'SHAP Waterfall (модель ТЯЖЕСТИ) - «Авария» | {_subset_title(subset)}\n'
        f'Двигатель {pump_id}: почему модель считает состояние предотказным\n'
        f'Красные признаки увеличивают вероятность отказа, синие - уменьшают',
        fontsize=14, pad=12)
    plt.tight_layout()

    path = os.path.join(save_dir, _fname(subset, 'plot6_shap_waterfall'))
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"График 6 (SHAP waterfall) сохранён: {path}")
    return path


def plot_summary_generalisation(summary_df: pd.DataFrame, save_dir: str) -> str:
    """
    Сравнение моделей, обобщённое по ВСЕМ сабсетам (аналог LOGO-CV-дашборда).

    Верхняя панель - сбалансированный holdout, нижняя - официальный тест
    NASA: mean ± std каждой метрики по четырём сабсетам для LogReg / RF /
    XGBoost (видно, что baseline не уступает ансамблям). Разрез KPI по
    сабсетам вынесен в усреднённые PR-кривые (сводный график 2).
    """

    subsets = sorted(summary_df['subset'].unique())
    splits = [s for s in ('holdout', 'official')
              if s in set(summary_df['split'])]
    models = [m for m in MODEL_ORDER if m in set(summary_df['model'])]

    fig, axes = plt.subplots(len(splits), 1, figsize=(9, 4.9 * len(splits)))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(LIGHT_BG)
    fig.suptitle('Обобщение модели тяжести на реальных данных NASA C-MAPSS\n'
                 f'{len(models)} модели × {len(subsets)} подмножества '
                 '(1/6 полётных режимов × 1/2 типа отказа)',
                 color=TEXT_CLR, fontsize=15, fontweight='bold', y=0.99)

    for r, split in enumerate(splits):
        d = summary_df[summary_df['split'] == split]
        ax = axes[r]
        _style_axis(ax)
        x = np.arange(len(SUMMARY_METRICS))
        w = 0.8 / max(len(models), 1)
        for j, model in enumerate(models):
            dm = d[d['model'] == model]
            means = [float(dm[k].mean()) for k, _, _ in SUMMARY_METRICS]
            stds = [float(dm[k].std(ddof=0)) for k, _, _ in SUMMARY_METRICS]
            offset = (j - len(models) / 2 + 0.5) * w
            bars = ax.bar(x + offset, means, w, yerr=stds, capsize=4,
                          label=model, color=PALETTE.get(model, '#888'),
                          ecolor='#444444', alpha=0.92, zorder=3)
            for b, mval, sval in zip(bars, means, stds):
                ax.annotate(f'{mval:.2f}±{sval:.2f}',
                            xy=(b.get_x() + b.get_width() / 2, mval + sval),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=12,
                            fontweight='bold', color=TEXT_CLR, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace(' (', '\n(') for _, l, _ in
                            SUMMARY_METRICS],
                           color=TEXT_CLR, fontsize=13)
        ax.set_ylim(0.4, 1.25)
        ax.set_ylabel('Значение метрики', color=TEXT_CLR, fontsize=13)
        ax.set_title(f'Сравнение моделей, mean ± std по сабсетам - '
                     f'{SPLIT_RU.get(split, split)}',
                     color=TEXT_CLR, fontsize=14, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.7,
                      color=GRID_CLR, zorder=0)
        ax.legend(loc='lower left', framealpha=1.0, labelcolor=TEXT_CLR,
                  fontsize=12, facecolor=LIGHT_BG, edgecolor='#CCCCCC')

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(save_dir,
                        f'{CMAPSS_GRAPHS_PREFIX}_summary_plot1_generalisation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"Сводный график 1 (сравнение моделей) сохранён: {path}")
    return path


def plot_summary_confusion(cm_df: pd.DataFrame, save_dir: str) -> str:
    """
    Сводные матрицы ошибок: строки - сплиты (holdout / official), столбцы - модели.

    cm_df: содержимое cmapss_confusion.csv - счётчики ячеек матриц по разрезу
    сабсет x сплит x модель. Матрицы агрегированы СУММОЙ абсолютных счётчиков
    по всем сабсетам: каждая тестовая строка каждого сабсета учтена ровно один
    раз, «усреднения матриц» нет. Нормировка по строке после суммирования даёт
    recall класса на объединении всех тестов; аннотация - доля строки (%) +
    абсолютный счётчик.
    """

    splits = [s for s in ('holdout', 'official') if s in set(cm_df['split'])]
    models = [m for m in MODEL_ORDER if m in set(cm_df['model'])]
    n_sub = cm_df['subset'].nunique()

    fig, axes = plt.subplots(len(splits), len(models),
                             figsize=(4.9 * len(models), 4.4 * len(splits)))
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor(LIGHT_BG)
    fig.suptitle('Сводные матрицы ошибок модели тяжести - NASA C-MAPSS\n'
                 f'сумма абсолютных счётчиков по {n_sub} подмножествам'
                 '\nВ ячейке - доля истинного класса (recall) и число строк',
                 color=TEXT_CLR, fontsize=18, fontweight='bold', y=1.0)

    for r, split in enumerate(splits):
        for c, model in enumerate(models):
            ax = axes[r][c]
            d = cm_df[(cm_df['split'] == split) & (cm_df['model'] == model)]
            cm = (d.pivot_table(index='true_class', columns='pred_class',
                                values='count', aggfunc='sum')
                  .reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
                  .to_numpy(dtype=float))
            cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            annot = np.array([[f'{cm_norm[i, j] * 100:.1f}%\n{int(cm[i, j]):,}'
                               for j in range(3)] for i in range(3)])
            sns.heatmap(cm_norm, annot=annot, fmt='', ax=ax, cmap='Blues',
                        vmin=0.0, vmax=1.0, linewidths=0.5, linecolor='#DDDDDD',
                        xticklabels=CLASS_LABELS,
                        yticklabels=CLASS_LABELS if c == 0 else False,
                        cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
            _style_axis(ax)
            if r == 0:
                ax.set_title(model, color=PALETTE.get(model, TEXT_CLR),
                             fontsize=16, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'{SPLIT_RU.get(split, split)}\n\nИстинно',
                              color=TEXT_CLR, fontsize=14)
            else:
                ax.set_ylabel('')
            ax.set_xlabel('Предсказано' if r == len(splits) - 1 else '',
                          color=TEXT_CLR, fontsize=14)
            ax.tick_params(labelsize=10)
            for i in range(3):
                ax.add_patch(Rectangle((i, i), 1, 1, fill=False,
                                       edgecolor='#FF8C00', lw=2))

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(save_dir,
                        f'{CMAPSS_GRAPHS_PREFIX}_summary_plot6_confusion.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"Сводный график 6 (матрицы ошибок, сумма по сабсетам) сохранён: {path}")
    return path


def plot_summary_pr_curves(curves_df: pd.DataFrame, summary_df: pd.DataFrame,
                           save_dir: str) -> str:
    """
    Усреднённые PR-кривые класса «Авария» (аналог насосного ML_plot3).

    curves_df: содержимое cmapss_pr_curves.csv - PR-кривые каждой тройки
    (сабсет x сплит x модель), интерполированные ml-пайплайном на единую
    сетку recall. Кривая на графике - СРЕДНЯЯ precision по сабсетам в каждой
    точке recall, закрашенная область под кривой соответствует среднему
    PR-AUC. В легенде - PR-AUC mean ± std по сабсетам (значения из сводной
    таблицы, совпадают с графиком 1). Базовая линия - средняя доля класса
    «Авария» в тестах сабсетов (случайный классификатор).
    """

    splits = [s for s in ('holdout', 'official') if s in set(curves_df['split'])]
    models = [m for m in MODEL_ORDER if m in set(curves_df['model'])]

    fig, axes = plt.subplots(len(splits), 1, figsize=(9, 6.6 * len(splits)))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(LIGHT_BG)
    fig.suptitle('PR-кривые класса «Авария», усреднённые по сабсетам C-MAPSS\n'
                 'кривая - средняя precision по сабсетам'
                 '\nОбласть под кривой = средний PR-AUC',
                 color=TEXT_CLR, fontsize=14, fontweight='bold', y=1.0)

    for ax, split in zip(axes, splits):
        _style_axis(ax)
        dc = curves_df[curves_df['split'] == split]
        ds = summary_df[summary_df['split'] == split]
        n_sub = dc['subset'].nunique()

        for model in models:
            dm = dc[dc['model'] == model]
            mean_prec = dm.groupby('recall')['precision'].mean().sort_index()
            auc = ds[ds['model'] == model]['pr_auc_critical']
            ax.plot(mean_prec.index, mean_prec.to_numpy(),
                    color=PALETTE.get(model, '#888888'), lw=2.2,
                    label=f'{model}  (PR-AUC = {auc.mean():.3f} '
                          f'± {auc.std(ddof=0):.3f})',
                    zorder=3)
            ax.fill_between(mean_prec.index, 0.0, mean_prec.to_numpy(),
                            color=PALETTE.get(model, '#888888'),
                            alpha=0.10, zorder=2)

        base = dc.groupby('subset')['baseline'].first()
        ax.axhline(float(base.mean()), color='#888888', lw=1.2, ls='--',
                   label=f'Случайный классификатор (P = {base.mean():.3f} '
                         f'± {base.std(ddof=0):.3f})',
                   zorder=1)

        ax.set_xlabel('Recall (полнота обнаружения предотказных строк)',
                      color=TEXT_CLR, fontsize=12)
        ax.set_ylabel('Precision (точность тревог)', color=TEXT_CLR, fontsize=12)
        ax.set_title(f'Среднее по {n_sub} сабсетам - '
                     f'{SPLIT_RU.get(split, split)}',
                     color=TEXT_CLR, fontsize=13, fontweight='bold')
        ax.legend(framealpha=1.0, labelcolor=TEXT_CLR, fontsize=10,
                  facecolor=LIGHT_BG, edgecolor='#CCCCCC', loc='lower left')
        ax.grid(True, linestyle='--', alpha=0.7, color=GRID_CLR, zorder=0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(save_dir,
                        f'{CMAPSS_GRAPHS_PREFIX}_summary_plot2_pr_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"Сводный график 2 (усреднённые PR-кривые) сохранён: {path}")
    return path


def plot_summary_lead_time(lead_frames: dict, save_dir: str,
                           model_label: str = 'XGBoost') -> str:
    """
    Сводное упреждение обнаружения по всем сабсетам (боксплоты).

    lead_frames: {subset: DataFrame lead-time}. Пары боксов на сабсет
    (Предупреждение/Авария) против границ разметки RUL_WARNING/RUL_CRITICAL:
    медианы у границ = модель поднимает стадию сразу, как только деградация
    становится различимой, на любом подмножестве.
    """

    subsets = sorted(lead_frames)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(LIGHT_BG)
    _style_axis(ax)

    W_CLR, C_CLR = '#E0A800', '#C62828'
    positions_w = [i * 2.0 for i in range(len(subsets))]
    positions_c = [p + 0.7 for p in positions_w]
    total_engines = 0
    for pos, col, color in ((positions_w, 'lead_warning', W_CLR),
                            (positions_c, 'lead_critical', C_CLR)):
        data = [lead_frames[s][col].dropna().to_numpy() for s in subsets]
        bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
                        medianprops=dict(color='black', lw=1.6),
                        flierprops=dict(markersize=4, alpha=0.6))
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
    total_engines = sum(len(lead_frames[s]) for s in subsets)

    ax.axhline(RUL_WARNING, color=W_CLR, ls='--', lw=1.6,
               label=f'Граница разметки «Предупреждение» ({RUL_WARNING} циклов)')
    ax.axhline(RUL_CRITICAL, color=C_CLR, ls='--', lw=1.6,
               label=f'Граница разметки «Авария» ({RUL_CRITICAL} циклов)')

    ax.set_xticks([p + 0.35 for p in positions_w])
    ax.set_xticklabels(subsets, fontsize=14, color=TEXT_CLR)
    ax.set_ylabel('Циклов до отказа при\nпервом срабатывании',
                  color=TEXT_CLR, fontsize=14)
    ax.set_title('Упреждение обнаружения деградации на всех сабсетах C-MAPSS\n'
                 f'{model_label}, двигатели holdout-тестов, дожитые до отказа '
                 f'(всего n={total_engines})'
                 '\nЖёлтое - первое «Предупреждение», '
                 f'красное - первая «Авария»',
                 color=TEXT_CLR, fontsize=16, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color=GRID_CLR, zorder=0)
    ax.legend(framealpha=1.0, labelcolor=TEXT_CLR, fontsize=12,
              facecolor=LIGHT_BG, edgecolor='#CCCCCC', loc='upper right')

    plt.tight_layout()
    path = os.path.join(save_dir,
                        f'{CMAPSS_GRAPHS_PREFIX}_summary_plot3_lead_time.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"Сводный график 3 (lead time по сабсетам) сохранён: {path}")
    return path


def plot_summary_shap_importance(importances: dict, save_dir: str,
                                 top_n: int = 12) -> str:
    """
    Усреднённая по сабсетам SHAP-важность признаков (mean ± std).

    importances: {subset: Series mean|SHAP| по признакам, нормированная в
    доли}. Признаки с одинаковыми именами во всех сабсетах выравниваются;
    топ-N по среднему вкладу. Малый std = модель на всех подмножествах
    опирается на одну и ту же физику двигателя - аргумент объяснимости.
    """

    imp_df = pd.DataFrame(importances) # index=признак, columns=сабсеты
    mean_imp = imp_df.mean(axis=1)
    std_imp = imp_df.std(axis=1, ddof=0)
    top = mean_imp.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor(LIGHT_BG)
    _style_axis(ax)
    y = np.arange(len(top))
    ax.barh(y, top.to_numpy() * 100,
            xerr=std_imp[top.index].to_numpy() * 100,
            color=C_PR, ecolor='#444444', capsize=4, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(display_feature_names(top.index), fontsize=11,
                       color=TEXT_CLR)
    ax.set_xlabel('Средняя доля |SHAP|-вклада в класс «Авария», % '
                  '(mean ± std по 4 сабсетам)', color=TEXT_CLR, fontsize=11)
    ax.set_title('Ключевые признаки деградации по ВСЕМ сабсетам C-MAPSS\n'
                 'Стабильный набор физических параметров = '
                 'воспроизводимая объяснимость',
                 color=TEXT_CLR, fontsize=13, fontweight='bold')
    ax.xaxis.grid(True, linestyle='--', alpha=0.7, color=GRID_CLR, zorder=0)

    plt.tight_layout()
    path = os.path.join(save_dir,
                        f'{CMAPSS_GRAPHS_PREFIX}_summary_plot4_shap_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"Сводный график 4 (SHAP-важность по сабсетам) сохранён: {path}")
    return path


def plot_summary_shap_beeswarm(shap_blocks: dict, save_dir: str,
                               max_display: int = 15) -> str:
    """
    НАЛОЖЕННЫЙ beeswarm: SHAP-точки всех сабсетов на одном графике.

    shap_blocks: {subset: (shap_matrix, X_ranked)} - SHAP-значения класса
    «Авария» и ПЕРЦЕНТИЛЬНЫЕ ранги значений признаков внутри своего сабсета.

    Почему ранги, а не сырые значения: признаковое пространство едино
    (те же 140 колонок), но в FD001/FD003 сенсоры сырые, а в FD002/FD004 -
    z-нормированные по режимам; прямое смешение шкал сделало бы цветовую
    кодировку «high/low» бессмысленной. Перцентиль внутри сабсета сохраняет
    честную семантику цвета: «высокое/низкое значение признака у себя дома».
    SHAP-значения по оси X - в единой шкале log-odds своих моделей.
    """

    cols = None
    shap_parts, x_parts = [], []
    for subset in sorted(shap_blocks):
        sv, xr = shap_blocks[subset]
        if cols is None:
            cols = list(xr.columns)
        assert list(xr.columns) == cols, \
            f"{subset}: контракт признаков расходится с остальными сабсетами"
        shap_parts.append(np.asarray(sv))
        x_parts.append(xr)
    shap_all = np.vstack(shap_parts)
    X_all = pd.concat(x_parts, ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.sca(ax)
    shap.summary_plot(shap_all, X_all, max_display=max_display,
                      feature_names=display_feature_names(X_all.columns),
                      plot_type='dot', show=False)
    plt.title('Сводный SHAP Beeswarm - вклад признаков в класс «Авария»\n'
              f'Все сабсеты C-MAPSS вместе (N={len(X_all):,} предотказных '
              f'строк, {len(shap_blocks)} модели)'
              '\nЦвет - перцентиль значения '
              f'признака внутри своего сабсета',
              fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()

    path = os.path.join(save_dir,
                        f'{CMAPSS_GRAPHS_PREFIX}_summary_plot5_shap_beeswarm.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Сводный график 5 (наложенный beeswarm) сохранён: {path}")
    return path


# Точка входа: пересборка сводных фигур 1-3 и 6 из сохранённых CSV
# (без переобучения; сводные SHAP-фигуры 4-5 строятся из cmapss_xai_module)

if __name__ == '__main__':
    tables_dir = os.path.join(_PROJECT_ROOT, 'artifacts', 'tables')
    graphs_dir = os.path.join(_PROJECT_ROOT, 'artifacts', 'graphs')

    summary_csv = os.path.join(tables_dir, f'{CMAPSS_TABLES_PREFIX}_ml_summary.csv')
    curves_csv = os.path.join(tables_dir, f'{CMAPSS_TABLES_PREFIX}_pr_curves.csv')
    confusion_csv = os.path.join(tables_dir, f'{CMAPSS_TABLES_PREFIX}_confusion.csv')
    if os.path.isfile(summary_csv):
        summary = pd.read_csv(summary_csv)
        plot_summary_generalisation(summary, graphs_dir)
        if os.path.isfile(curves_csv):
            plot_summary_pr_curves(pd.read_csv(curves_csv), summary, graphs_dir)
        else:
            print(f"[WARN] Нет {curves_csv} - сводные PR-кривые пропущены.")
        if os.path.isfile(confusion_csv):
            plot_summary_confusion(pd.read_csv(confusion_csv), graphs_dir)
        else:
            print(f"[WARN] Нет {confusion_csv} - сводные матрицы ошибок "
                  f"пропущены (нужен прогон cmapss_ml_pipeline).")
    else:
        print(f"[WARN] Нет {summary_csv} - сначала прогоните cmapss_ml_pipeline.")

    lead_frames = {}
    for sub in sorted(CMAPSS_SUBSET_INFO):
        p = os.path.join(tables_dir,
                         f'{CMAPSS_TABLES_PREFIX}_{sub.lower()}_lead_time.csv')
        if os.path.isfile(p):
            lead_frames[sub] = pd.read_csv(p)
    if lead_frames:
        plot_summary_lead_time(lead_frames, graphs_dir)
    else:
        print("[WARN] Нет cmapss_*_lead_time.csv - сводный lead time пропущен.")
