"""
Блок визуализации бенчмарка LLM-агента
======================================
src/visualisation/ai_visualisation.py

Строит пять графиков, сравнивающих локальные LLM (Qwen 3.5 9B, Phi-4 14B,
YandexGPT-5 8B) в роли диагностического агента. Вынесены из бенчмарка
(experiments/llm_benchmark/ai_agent_benchmark.py) в отдельный модуль по
принципу SRP: логика бенчмарка не зависит от matplotlib/seaborn.

Источник данных - таблицы бенчмарка (см. artifacts/tables/):

- agent_benchmark_multi.csv - сырые прогоны (одна строка = один прогон
  сценария): скорость (gen_time_sec, tokens_per_sec), булевы метрики
  качества (format_ok, attribution_ok, toir_is_works,
  stage_appropriate), полнота предписания (action_steps) и стадия
  (stage).
- agent_summary_table.csv - сводка по моделям (index = модель).

Пять графиков (сохраняются с префиксом agent_plot1-5_*):

- plot_performance - скорость: среднее время генерации и пропускная
  способность (токенов/с).
- plot_quality_auto - профиль качества по дискриминирующим булевым метрикам
  плюс полнота предписания (число шагов).
- plot_summary_heatmap - сводный хитмап относительного качества с учётом
  направления каждой метрики (DIRECTIONS / FMT из настроек).
- plot_stage_breakdown - разрез по стадиям (Предупреждение / Авария):
  стадийная уместность и время генерации.
- plot_expert_radar - радар экспертных оценок (шкала 1–5, заполняется
  вручную).

Порядок и метки моделей берутся из config.settings (LLM_MODELS,
MODEL_LABELS); цвета - из локальной палитры PALETTE.

Бэкенд matplotlib принудительно Agg (без GUI) - для серверного прогона
скриптов. Вызывается из experiments/llm_benchmark/ai_agent_benchmark.py и
автономно из __main__ (строит графики из сохранённых CSV).
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # без GUI - для серверного прогона и сохранения в PNG
import matplotlib.pyplot as plt
import seaborn as sns

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config.settings import MODEL_LABELS, LLM_MODELS, DIRECTIONS, FMT


# Единые для всех графиков цвета моделей + русские метки и цвета стадий.
PALETTE = {'Qwen 3.5 9B': '#C44E52', 'Phi-4 14B': '#4C72B0',
           'YandexGPT-5 8B': '#55A868'}
STAGE_RU = {'warning': 'Предупреждение', 'critical': 'Авария'}
STAGE_CLR = {'Предупреждение': '#DD8452', 'Авария': '#C44E52'}


def ai_vis_order():
    """Отображаемые названия моделей в каноническом порядке (по LLM_MODELS)."""

    return [MODEL_LABELS[m] for m in LLM_MODELS]


# График 1 - производительность (сравнительная характеристика)

def plot_performance(df: pd.DataFrame, save_dir: str):
    """График 1: производительность - время генерации и пропускная способность."""

    agg = df.groupby('model').agg(
        t_mean=('gen_time_sec', 'mean'), t_std=('gen_time_sec', 'std'),
        tps_mean=('tokens_per_sec', 'mean'), tps_std=('tokens_per_sec', 'std'),
    ).reindex(ai_vis_order())
    colors = [PALETTE[m] for m in agg.index]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].bar(agg.index, agg['t_mean'], yerr=agg['t_std'], color=colors,
              alpha=0.85, capsize=5, edgecolor='black', lw=0.5)
    ax[0].set_ylabel('Время генерации, с', fontsize=14)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[0].grid(axis='y', alpha=0.3, ls='--')
    ax[0].set_title('Среднее время генерации\n(меньше - лучше)',
                    fontsize=16, fontweight='bold')
    # Подпись - над верхней «усикой» разброса (mean + std), чтобы cap не
    # перекрывал число; off - небольшой отступ сверху для читаемости.
    t_std = agg['t_std'].fillna(0)
    off0 = 0.03 * float((agg['t_mean'] + t_std).max())
    ax[0].set_ylim(0, float((agg['t_mean'] + t_std).max()) * 1.15)
    for i, (v, s) in enumerate(zip(agg['t_mean'], t_std)):
        ax[0].text(i, v + s + off0, f'{v:.1f}с', ha='center', va='bottom',
                   fontweight='bold', fontsize=14)

    ax[1].bar(agg.index, agg['tps_mean'], yerr=agg['tps_std'], color=colors,
              alpha=0.85, capsize=5, edgecolor='black', lw=0.5)
    ax[1].set_ylabel('Токенов в секунду', fontsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].grid(axis='y', alpha=0.3, ls='--')
    ax[1].set_title('Пропускная способность\n(больше - лучше)',
                    fontsize=16, fontweight='bold')
    tps_std = agg['tps_std'].fillna(0)
    off1 = 0.03 * float((agg['tps_mean'] + tps_std).max())
    ax[1].set_ylim(0, float((agg['tps_mean'] + tps_std).max()) * 1.15)
    for i, (v, s) in enumerate(zip(agg['tps_mean'], tps_std)):
        ax[1].text(i, v + s + off1, f'{v:.1f}', ha='center', va='bottom',
                   fontweight='bold', fontsize=14)

    plt.suptitle('Производительность локальных LLM',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, 'agent_plot1_performance.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"График 1 сохранён: {p}")


# График 2 - профиль качества (ПОЧЕМУ модель лучше/хуже)

def plot_quality_auto(df: pd.DataFrame, save_dir: str):
    """
    График 2: профиль качества предписаний по ДИСКРИМИНИРУЮЩИМ метрикам.
    Сгруппированные столбцы по моделям сразу показывают, чем модели различаются
    (атрибуция и стадийная уместность - главные разделители; формат и ТОиР, как
    правило, насыщены у всех). Это и есть обоснование выбора модели для диплома.
    """

    candidate = {'Формат': 'format_ok', 'Атрибуция': 'attribution_ok',
                 'ТОиР-работы': 'toir_is_works',
                 'Стадийная\nуместность': 'stage_appropriate'}
    bool_metrics = {k: v for k, v in candidate.items() if v in df.columns}

    order = ai_vis_order()
    agg = df.groupby('model')[list(bool_metrics.values())].mean().reindex(order) * 100
    steps = df.groupby('model')['action_steps'].mean().reindex(order)

    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5),
                           gridspec_kw={'width_ratios': [3, 1]})
    # Сгруппированные столбцы: n_m метрик на каждую модель; общая ширина группы
    # 0.8, ширина одного столбца w. Смещение (j*w - 0.4 + w/2) центрирует группу
    # вокруг позиции модели x.
    n_m = len(bool_metrics); x = np.arange(len(order)); w = 0.8 / max(n_m, 1)
    metric_colors = ['#4C72B0', '#C44E52', '#55A868', '#8172B2', '#CCB974']
    for j, (name, col) in enumerate(bool_metrics.items()):
        vals = agg[col].values
        ax[0].bar(x + j * w - 0.4 + w / 2, vals, w, label=name,
                  color=metric_colors[j % len(metric_colors)],
                  alpha=0.88, edgecolor='black', lw=0.4)
        for xi, v in zip(x + j * w - 0.4 + w / 2, vals):
            ax[0].text(xi, v + 1.5, f'{v:.0f}', ha='center',
                       fontsize=14, fontweight='bold')
    ax[0].set_xticks(x); ax[0].set_xticklabels(order, fontsize=14)
    ax[0].set_ylim(0, 115); ax[0].axhline(100, color='green', ls='--', alpha=0.4)
    ax[0].set_ylabel('% ответов', fontsize=14)
    ax[0].set_title('Профиль качества предписаний\n'
                    '(дискриминирующие метрики, % соответствия)',
                    fontsize=18, fontweight='bold')
    ax[0].legend(fontsize=14, ncol=2, loc='lower center')
    ax[0].grid(axis='y', alpha=0.3, ls='--')

    colors = [PALETTE[m] for m in order]
    ax[1].bar(order, steps.values, color=colors, alpha=0.85,
              edgecolor='black', lw=0.5)
    ax[1].set_ylim(0, 3.7)
    ax[1].set_ylabel('Среднее число шагов', fontsize=14)
    ax[1].set_title('Полнота\nпредписания', fontsize=18, fontweight='bold')
    ax[1].tick_params(axis='x', rotation=15, labelsize=14)
    ax[1].grid(axis='y', alpha=0.3, ls='--')
    for i, v in enumerate(steps.values):
        ax[1].text(i, v, f'{v:.1f}', ha='center', va='bottom',
                   fontweight='bold', fontsize=14)

    plt.suptitle('Автоматические метрики качества предписаний',
                 fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, 'agent_plot2_quality_auto.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"График 2 сохранён: {p}")


# График 3 - сводный хитмап (как раньше)

def plot_summary_heatmap(summary_df, directions, fmt_map, save_dir,
                         filename='agent_plot3_summary_heatmap.png'):
    """График 3: сравнительный хитмап по сводной таблице (относительное качество)."""

    cols = [c for c in directions if c in summary_df.columns]
    raw = summary_df[cols]

    # Мин-макс нормировка каждой метрики в [0, 1] с учётом её направления:
    # 'higher' - больше=лучше, иначе - меньше=лучше; так все столбцы становятся
    # сопоставимы по цвету (1 = лучший, 0 = худший). Нулевой размах → 0.5 (нейтраль).
    goodness = pd.DataFrame(index=raw.index, columns=cols, dtype=float)
    for c in cols:
        v = raw[c].astype(float)
        rng = float(v.max() - v.min())
        if rng == 0:
            goodness[c] = 0.5
        elif directions[c] == 'higher':
            goodness[c] = (v - v.min()) / rng
        else:
            goodness[c] = (v.max() - v) / rng

    # В ячейках показываются исходные значения (raw) в формате из fmt_map,
    # а цвет кодирует нормированное «качество» (goodness).
    annot = [[fmt_map.get(c, '{:.2f}').format(raw.loc[m, c]) for c in cols]
             for m in raw.index]

    LIGHT_BG, TEXT_CLR = '#FFFFFF', '#222222'
    fig, ax = plt.subplots(figsize=(1.7 * len(cols) + 3, 0.9 * len(raw) + 2.5))
    fig.patch.set_facecolor(LIGHT_BG)
    sns.heatmap(goodness.astype(float), annot=annot, fmt='', cmap='RdYlGn',
                vmin=0, vmax=1, linewidths=1.2, linecolor='#DDDDDD',
                cbar_kws={'label': 'Относительное качество (зелёный - лучше)'},
                annot_kws={'size': 15, 'weight': 'bold'}, ax=ax)
    # Кегль подписи и делений цветовой шкалы задаётся на самом объекте colorbar.
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.set_label('Относительное качество (зелёный - лучше)', size=12)
        cbar.ax.tick_params(labelsize=12)
    ax.set_title('Сравнение LLM по сводным метрикам\n'
                 'Цвет - относительное качество с учётом направления метрики',
                 color=TEXT_CLR, fontsize=15, fontweight='bold', pad=14)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.tick_params(axis='both', colors=TEXT_CLR, labelsize=14)
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    p = os.path.join(save_dir, filename)
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close()
    print(f"График 3 (хитмап) сохранён: {p}")


# График 4 - стадийный разрез: поведение по стадиям

def plot_stage_breakdown(df: pd.DataFrame, save_dir: str):
    """
    График 4: сравнение моделей в разрезе стадий (Предупреждение / Авария).
    Левая панель - стадийная уместность (доказывает, что упреждающий режим
    действительно отрабатывает, а не выдаёт аварийный останов на предупреждении).
    Правая - время генерации по стадиям.
    """

    if 'stage' not in df.columns:
        print("[WARN] нет колонки 'stage' - стадийный график пропущен.")
        return
    d = df.copy(); d['stage_ru'] = d['stage'].map(STAGE_RU)
    order = ai_vis_order(); stages = ['Предупреждение', 'Авария']

    appr = (d.groupby(['model', 'stage_ru'])['stage_appropriate'].mean()
            .unstack('stage_ru').reindex(order) * 100)
    tmean = (d.groupby(['model', 'stage_ru'])['gen_time_sec'].mean()
             .unstack('stage_ru').reindex(order))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(order)); w = 0.38

    for j, st in enumerate(stages):
        series = appr[st].values if st in appr.columns else np.zeros(len(order))
        ax[0].bar(x + (j - 0.5) * w, series, w, label=st,
                  color=STAGE_CLR[st], alpha=0.88, edgecolor='black', lw=0.4)
        for xi, v in zip(x + (j - 0.5) * w, series):
            ax[0].text(xi, v + 1.5, f'{v:.0f}', ha='center',
                       fontsize=14, fontweight='bold')
    ax[0].set_xticks(x); ax[0].set_xticklabels(order, fontsize=14)
    ax[0].set_ylim(0, 115); ax[0].axhline(100, color='green', ls='--', alpha=0.4)
    ax[0].set_ylabel('% уместных предписаний', fontsize=14)
    ax[0].set_title('Стадийная уместность по стадиям\n(больше - лучше)',
                    fontsize=16, fontweight='bold')
    ax[0].legend(fontsize=14); ax[0].grid(axis='y', alpha=0.3, ls='--')

    tmax = float(np.nanmax(tmean.values)) if tmean.size else 0.0
    off = 0.02 * tmax   # отступ подписи над баром, относительно шкалы времени
    for j, st in enumerate(stages):
        series = tmean[st].values if st in tmean.columns else np.zeros(len(order))
        ax[1].bar(x + (j - 0.5) * w, series, w, label=st,
                  color=STAGE_CLR[st], alpha=0.88, edgecolor='black', lw=0.4)
        for xi, v in zip(x + (j - 0.5) * w, series):
            if np.isnan(v):
                continue
            ax[1].text(xi, v + off, f'{v:.1f}', ha='center', va='bottom',
                       fontsize=14, fontweight='bold')
    ax[1].set_ylim(0, tmax * 1.15 if tmax else 1.0)
    ax[1].set_xticks(x); ax[1].set_xticklabels(order, fontsize=14)
    ax[1].set_ylabel('Время генерации, с', fontsize=14)
    ax[1].set_title('Время генерации по стадиям\n(меньше - лучше)',
                    fontsize=16, fontweight='bold')
    ax[1].legend(fontsize=14); ax[1].grid(axis='y', alpha=0.3, ls='--')

    plt.suptitle('Поведение моделей в разрезе стадий (Предупреждение / Авария)',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, 'agent_plot4_stage_breakdown.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"График 4 сохранён: {p}")


# График 5 - радар экспертных оценок (заполняется вручную)

def plot_expert_radar(expert_scores: dict, save_dir: str):
    """
    График 5: радар экспертных оценок по пяти критериям (шкала 1–5).
    В отличие от остальных графиков, оценки не считаются автоматически из CSV,
    а задаются вручную (экспертная разметка ответов агента) через expert_scores:
    {название_модели: {критерий: балл}}.
    """

    criteria = ['Корректность\nдиагноза', 'Полнота\nпредписания', 'Точность\nметрик',
                'Соответствие\nконтексту', 'Читаемость']
    angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
    angles += angles[:1]   # замыкание контура: последняя точка совпадает с первой

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for label, scores in expert_scores.items():
        # Критерии в criteria содержат переносы '\n'; ключи в scores - без них.
        vals = [scores.get(c.replace('\n', ' '), 0) for c in criteria]
        vals += vals[:1]   # замыкание полигона на радаре
        color = PALETTE.get(label, 'grey')
        ax.plot(angles, vals, color=color, lw=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.12)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(criteria, fontsize=14)
    ax.set_ylim(0, 5); ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title('Экспертная оценка качества предписаний\n(шкала 1–5)',
                 fontsize=16, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.1), fontsize=14)
    plt.tight_layout()
    p = os.path.join(save_dir, 'agent_plot5_expert_radar.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"График 5 сохранён: {p}")


if __name__ == "__main__":
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))))
    save_graphs_dir = os.path.join(project_root, 'artifacts', 'graphs')
    tables_dir = os.path.join(project_root, 'artifacts', 'tables')
    os.makedirs(save_graphs_dir, exist_ok=True)

    # Все графики строятся автономно из сохранённых CSV бенчмарка.
    csv_multi = os.path.join(tables_dir, 'agent_benchmark_multi.csv')
    csv_summary = os.path.join(tables_dir, 'agent_summary_table.csv')

    # Из сырых прогонов (multi): производительность, профиль качества, стадии.
    if os.path.exists(csv_multi):
        dfb = pd.read_csv(csv_multi)
        plot_performance(dfb, save_graphs_dir)
        plot_quality_auto(dfb, save_graphs_dir)
        plot_stage_breakdown(dfb, save_graphs_dir)
    else:
        print(f"[WARN] нет {csv_multi} - графики 1/2/4 пропущены.")

    # Из сводной таблицы (index = модель): сводный хитмап.
    if os.path.exists(csv_summary):
        dfb = pd.read_csv(csv_summary, index_col=0)
        plot_summary_heatmap(dfb, DIRECTIONS, FMT, save_graphs_dir)
    else:
        print(f"[WARN] нет {csv_summary} - сводный хитмап пропущен.")

    expert_scores = {
        'Qwen 3.5 9B': {'Корректность диагноза': 5, 
                        'Полнота предписания': 5,
                        'Точность метрик': 4.75, 
                        'Соответствие контексту': 4.83, 
                        'Читаемость': 5},
        'Phi-4 14B': {'Корректность диагноза': 5, 
                      'Полнота предписания': 4.125,
                      'Точность метрик': 4.58, 
                      'Соответствие контексту': 4.58, 
                      'Читаемость': 4.96},
        'YandexGPT-5 8B': {'Корректность диагноза': 5, 
                           'Полнота предписания': 4.21,
                           'Точность метрик': 4.21, 
                           'Соответствие контексту': 4.75, 
                           'Читаемость': 5},
    }
    plot_expert_radar(expert_scores, save_graphs_dir)