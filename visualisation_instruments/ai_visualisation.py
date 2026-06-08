import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config.settings import MODEL_LABELS, LLM_MODELS


PALETTE = {'Qwen 3.5 9B': '#C44E52', 'Phi-4 14B': '#4C72B0', 
           'YandexGPT-5 8B': '#55A868'}

# Визуализация

def ai_vis_order():
    return [MODEL_LABELS[m] for m in LLM_MODELS]


def plot_performance(df: pd.DataFrame, save_dir: str):
    """График 1: производительность — время и пропускная способность."""

    agg = df.groupby('model').agg(
        t_mean=('gen_time_sec', 'mean'), t_std=('gen_time_sec', 'std'),
        tps_mean=('tokens_per_sec', 'mean'), tps_std=('tokens_per_sec', 'std'),
    ).reindex(ai_vis_order())
    colors = [PALETTE[m] for m in agg.index]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].bar(agg.index, agg['t_mean'], yerr=agg['t_std'], color=colors,
              alpha=0.85, capsize=5, edgecolor='black', lw=0.5)
    ax[0].set_ylabel('Время генерации, с', fontsize=12)
    ax[0].tick_params(axis='both', labelsize=12)
    ax[0].grid(axis='y', alpha=0.3, ls='--')
    ax[0].set_title('Среднее время генерации\n(меньше — лучше)',
                    fontsize=12, fontweight='bold')
    for i, v in enumerate(agg['t_mean']):
        ax[0].text(i, v, f'{v:.1f}с', ha='center', va='bottom',
                   fontweight='bold', fontsize=12)

    ax[1].bar(agg.index, agg['tps_mean'], yerr=agg['tps_std'], color=colors,
              alpha=0.85, capsize=5, edgecolor='black', lw=0.5)
    ax[1].set_ylabel('Токенов в секунду', fontsize=12)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[1].grid(axis='y', alpha=0.3, ls='--')
    ax[1].set_title('Пропускная способность\n(больше — лучше)',
                    fontsize=12, fontweight='bold')
    for i, v in enumerate(agg['tps_mean']):
        ax[1].text(i, v, f'{v:.1f}', ha='center', va='bottom',
                   fontweight='bold', fontsize=12)

    plt.suptitle('Производительность локальных LLM', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, 'agent_plot1_performance.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"График 1 сохранён: {p}")


def plot_quality_auto(df: pd.DataFrame, save_dir: str):
    """График 2: автоматические метрики (формат, атрибуция, число шагов)."""

    agg = df.groupby('model').agg(
        fmt=('format_ok', 'mean'), attr=('attribution_ok', 'mean'),
        steps=('action_steps', 'mean'),
    ).reindex(ai_vis_order())
    colors = [PALETTE[m] for m in agg.index]

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax[0].bar(agg.index, agg['fmt']*100, color=colors, alpha=0.85,
              edgecolor='black', lw=0.5)
    ax[0].set_ylim(0, 115); ax[0].axhline(100, color='green', ls='--', alpha=0.5)
    ax[0].set_ylabel('% ответов', fontsize=12)
    ax[0].set_title('Соблюдение формата\n(4 раздела)', fontsize=12, fontweight='bold')
    for i, v in enumerate(agg['fmt']):
        ax[0].text(i, v*100+2, f'{v*100:.0f}%', ha='center',
                   fontweight='bold', fontsize=12)

    ax[1].bar(agg.index, agg['steps'], color=colors, alpha=0.85,
              edgecolor='black', lw=0.5)
    ax[1].set_ylabel('Среднее число шагов', fontsize=12)
    ax[1].set_title('Полнота предписания', fontsize=12, fontweight='bold')
    ax[1].grid(axis='y', alpha=0.3, ls='--')
    for i, v in enumerate(agg['steps']):
        ax[1].text(i, v, f'{v:.1f}', ha='center', va='bottom',
                   fontweight='bold', fontsize=12)

    for a in ax:
        a.tick_params(axis='x', rotation=15, labelsize=12)
        a.tick_params(axis='y', labelsize=12)
    plt.suptitle('Автоматические метрики качества предписаний', fontsize=14, 
                 fontweight='bold', y=1.03)
    plt.tight_layout()
    p = os.path.join(save_dir, 'agent_plot2_quality_auto.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"График 2 сохранён: {p}")

def plot_summary_heatmap(summary_df, directions, fmt_map, save_dir,
                         filename='agent_plot3_summary_heatmap.png'):
    """
    График 3: Сравнительный хитмап по сводной таблице моделей.
    Цвет = ОТНОСИТЕЛЬНОЕ качество в столбце (зелёный — лучше, красный — хуже)
    с учётом направления метрики (directions: 'higher'/'lower'); в ячейках —
    фактические значения (fmt_map). Это делает разнонаправленные метрики
    (Время↓ против Токен/с↑) визуально сопоставимыми.
    """

    cols = [c for c in directions if c in summary_df.columns]
    raw = summary_df[cols]

    goodness = pd.DataFrame(index=raw.index, columns=cols, dtype=float)
    for c in cols:
        v = raw[c].astype(float)
        rng = float(v.max() - v.min())
        if rng == 0:
            goodness[c] = 0.5
        elif directions[c] == 'higher':
            goodness[c] = (v - v.min()) / rng
        else:  # 'lower' — меньше=лучше
            goodness[c] = (v.max() - v) / rng

    annot = [[fmt_map.get(c, '{:.2f}').format(raw.loc[m, c]) for c in cols]
             for m in raw.index]

    LIGHT_BG, TEXT_CLR = '#FFFFFF', '#222222'
    fig, ax = plt.subplots(figsize=(1.7*len(cols)+3, 0.9*len(raw)+2.5))
    fig.patch.set_facecolor(LIGHT_BG)
    sns.heatmap(goodness.astype(float), annot=annot, fmt='', cmap='RdYlGn',
                vmin=0, vmax=1, linewidths=1.2, linecolor='#DDDDDD',
                cbar_kws={'label': 'Относительное качество (зелёный — лучше)'},
                annot_kws={'size': 13, 'weight': 'bold'}, ax=ax)
    ax.set_title('Сравнение LLM по сводным метрикам\n'
                 'Цвет — относительное качество с учётом направления метрики; ',
                 color=TEXT_CLR, fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.tick_params(colors=TEXT_CLR, labelsize=12)
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    p = os.path.join(save_dir, filename)
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close()
    print(f"График 3 (хитмап) сохранён: {p}")

def plot_expert_radar(expert_scores: dict, save_dir: str):
    """График 4: радар экспертных оценок (заполняется вручную)."""

    criteria = ['Корректность\nдиагноза', 'Полнота\nпредписания', 'Точность\nметрик',
                'Соответствие\nконтексту', 'Читаемость']
    angles = np.linspace(0, 2*np.pi, len(criteria), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for label, scores in expert_scores.items():
        vals = [scores.get(c.replace('\n', ' '), 0) for c in criteria]
        vals += vals[:1]
        color = PALETTE.get(label, 'grey')
        ax.plot(angles, vals, color=color, lw=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.12)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(criteria, fontsize=10)
    ax.set_ylim(0, 5); ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title('Экспертная оценка качества предписаний\n(шкала 1–5)',
                 fontsize=13, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.1), fontsize=11)
    plt.tight_layout()
    p = os.path.join(save_dir, 'agent_plot4_expert_radar.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"График 4 сохранён: {p}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_graphs_dir = os.path.join(root, 'data', 'graphs')

    # Экспертные оценки — заполняются значениями после ручной проверки
    expert_scores = {
        'Qwen 3.5 9B': {'Корректность диагноза': 5, 
                        'Полнота предписания': 5,
                        'Точность метрик': 4, 
                        'Соответствие контексту': 4, 
                        'Читаемость': 5},
        'Phi-4 14B': {'Корректность диагноза': 5, 
                      'Полнота предписания': 3,
                      'Точность метрик': 4, 
                      'Соответствие контексту': 4, 
                      'Читаемость': 4},
        'YandexGPT-5 8B': {'Корректность диагноза': 5, 
                           'Полнота предписания': 4,
                           'Точность метрик': 3, 
                           'Соответствие контексту': 4, 
                           'Читаемость': 5},
    }
    plot_expert_radar(expert_scores, save_graphs_dir)
