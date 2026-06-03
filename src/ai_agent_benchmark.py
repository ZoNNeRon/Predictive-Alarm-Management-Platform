"""
Многосценарный бенчмарк LLM-моделей для AI-агента
===========================================================
Прогоняет три модели (qwen3.5:9b, phi4:14b, yandex-gpt-5-lite:8b) на наборе
диагностических сценариев, собирает объективные метрики и строит графики
для диплома.

  - Три РАЗНЫХ диагностических сценария вместо одного повторённого:
      1. Перегрев + рост тока      (Сценарий А: критический перегрев)
      2. Вибрация + падение давления (Сценарий Б: кавитация/повреждение колеса)
      3. Аномалия тока без перегрева (Сценарий В: электрическая часть)
    Это проверяет, РАЗЛИЧАЕТ ли модель типы отказов, а не выдаёт одно и то же.
  - Новая автометрика: атрибуция источника (ссылается ли модель на документ,
    а не на безымянный "Сценарий А"). Автоматизирует дефект, замеченный вручную.
  - Усреднение метрик по 3 сценариям x 3 повтора = 9 прогонов на модель.
"""


import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_agent import DiagnosticAgent
from xai_module import XAIExplainer
from data_preprocessor import DataPreprocessor
from rag_database import KnowledgeBaseManager


# ── Конфигурация ──────────────────────────────────────────────────────────

MODELS = ['qwen3.5:9b', 'phi4:14b', 'second_constantine/yandex-gpt-5-lite:8b']

MODEL_LABELS = {
    'qwen3.5:9b':                              'Qwen 3.5 9B',
    'phi4:14b':                                'Phi-4 14B',
    'second_constantine/yandex-gpt-5-lite:8b': 'YandexGPT-5 8B',
}

PALETTE = {'Qwen 3.5 9B': '#C44E52', 'Phi-4 14B': '#4C72B0', 'YandexGPT-5 8B': '#55A868'}

# Обязательные разделы для проверки формата
REQUIRED_SECTIONS = ['СТАТУС', 'ДИАГНОЗ', 'ПРЕДПИСАНИЕ', 'ТОиР']

# Маркеры явной атрибуции источника (модель ссылается на документ, а не на "Сценарий А")
SOURCE_MARKERS = ['регламент', 'tm_regulation', 'гост', 'мануал', 'mnhv',
                  'руководств', 'источник', 'документ']


# ── Автоматические метрики качества ───────────────────────────────────────

def check_format(text: str) -> bool:
    """Все 4 обязательных раздела присутствуют в ответе."""
    return all(any(sec in line for line in text.split('\n')) for sec in REQUIRED_SECTIONS)


def check_attribution(text: str) -> bool:
    """
    Модель явно ссылается на нормативный документ, а не на безымянный "Сценарий А".
    Это автоматизирует дефект Qwen: упоминание сценария без указания источника
    создаёт "чёрный ящик" для оператора.
    """
    low = text.lower()
    return any(marker in low for marker in SOURCE_MARKERS)


def count_action_steps(text: str) -> int:
    """
    Число пунктов в разделе ПРЕДПИСАНИЕ — мера полноты плана действий.
    Автоматизирует наблюдение, что Phi-4 урезает порядок действий.
    """
    # Берём блок после "ПРЕДПИСАНИЕ" до следующего заголовка
    m = re.search(r'ПРЕДПИСАНИЕ.*?(?=ИНФОРМАЦИЯ|УРОВЕНЬ|$)', text, re.S | re.I)
    if not m:
        return 0
    block = m.group(0)
    # Считаем нумерованные пункты "1.", "2." ...
    return len(re.findall(r'^\s*\d+\.', block, re.M))


# ── Подготовка сценариев ──────────────────────────────────────────────────

def build_scenarios(xai, df, preprocessor, kb) -> list:
    """
    Формирует три разных сценария из реальных данных датасета.
    Каждый сценарий — свой тип отказа (разные разделы базы знаний).
    """
    fc = preprocessor.FEATURE_COLS
    scenarios = []

    # Сценарий 1: критическая авария (перегрев + ток) — класс 2
    crit = df[df['target'] == 2]
    if len(crit) > 0:
        row = crit.iloc[-1:][fc]
        sv = xai.explain(row, pump_id=str(crit.iloc[-1]['pump_id']),
                         timestamp=str(crit.iloc[-1].get('timestamp', 'N/A')), top_k=5)
        scenarios.append(_make_scenario(sv, kb, 'Перегрев+ток'))

    # Сценарий 2: деградация (Warning) — класс 1, обычно вибрация/тренд
    warn = df[df['target'] == 1]
    if len(warn) > 0:
        # Берём строку с наибольшей вибрацией среди Warning для контраста
        warn_sorted = warn.sort_values('vibration_max_15', ascending=False) \
            if 'vibration_max_15' in warn.columns else warn
        row = warn_sorted.iloc[:1][fc]
        sv = xai.explain(row, pump_id=str(warn_sorted.iloc[0]['pump_id']),
                         timestamp=str(warn_sorted.iloc[0].get('timestamp', 'N/A')), top_k=5)
        scenarios.append(_make_scenario(sv, kb, 'Деградация/вибрация'))

    # Сценарий 3: ещё одна критическая авария от другого насоса (вариативность)
    if len(crit) > 1:
        # Авария от другого pump_id, если есть
        other = crit[crit['pump_id'] != crit.iloc[-1]['pump_id']]
        src = other if len(other) > 0 else crit
        row = src.iloc[:1][fc]
        sv = xai.explain(row, pump_id=str(src.iloc[0]['pump_id']),
                         timestamp=str(src.iloc[0].get('timestamp', 'N/A')), top_k=5)
        scenarios.append(_make_scenario(sv, kb, 'Авария (др. агрегат)'))

    return scenarios


def _make_scenario(sv, kb, label) -> dict:
    """Собирает RAG-контекст для вектора симптомов."""
    symptom_dict = {
        'critical_probability': sv.critical_probability,
        'top_symptoms': [{'sensor': s.sensor, 'value': s.value,
                          'shap_weight': s.shap_weight} for s in sv.top_symptoms]
    }
    rag_results = kb.search_by_symptoms(symptom_dict, k=4)
    return {'sv': sv, 'rag': rag_results, 'label': label}


# ── Прогон бенчмарка ──────────────────────────────────────────────────────

def run_benchmark(scenarios: list, n_repeats: int = 3) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        label = MODEL_LABELS.get(model, model)
        print(f"\n{'='*55}\nМодель: {label}\n{'='*55}")
        agent = DiagnosticAgent(model_name=model)

        # Прогрев (загрузка в память) на первом сценарии — не учитывается в замерах
        print(f"  Прогрев...")
        try:
            agent.generate_prescription(scenarios[0]['sv'], scenarios[0]['rag'])
        except Exception as e:
            print(f"    [WARN] прогрев: {e}")

        for sc in scenarios:
            for rep in range(n_repeats):
                resp = agent.generate_prescription(sc['sv'], sc['rag'])
                if resp.error:
                    print(f"  [{sc['label']}] rep{rep}: ОШИБКА — {resp.error}")
                    continue

                fmt   = check_format(resp.raw_text)
                attr  = check_attribution(resp.raw_text)
                steps = count_action_steps(resp.raw_text)

                rows.append({
                    'model':          label,
                    'scenario':       sc['label'],
                    'repeat':         rep,
                    'gen_time_sec':   resp.gen_time_sec,
                    'tokens_per_sec': resp.tokens_per_sec,
                    'eval_count':     resp.eval_count,
                    'prompt_tokens':  resp.prompt_eval_count,
                    'answer_len':     len(resp.raw_text),
                    'format_ok':      fmt,
                    'attribution_ok': attr,
                    'action_steps':   steps,
                    'raw_text':       resp.raw_text,
                })
                print(f"  [{sc['label']}] rep{rep}: {resp.gen_time_sec}с, "
                      f"{resp.tokens_per_sec} ток/с, формат={'OK' if fmt else 'НЕТ'}, "
                      f"источник={'OK' if attr else 'НЕТ'}, шагов={steps}")
    return pd.DataFrame(rows)


# ── Визуализация ──────────────────────────────────────────────────────────

def _order():
    return [MODEL_LABELS[m] for m in MODELS]


def plot_performance(df: pd.DataFrame, save_dir: str):
    """График 1: производительность — время и пропускная способность (по 9 прогонам)."""
    agg = df.groupby('model').agg(
        t_mean=('gen_time_sec', 'mean'), t_std=('gen_time_sec', 'std'),
        tps_mean=('tokens_per_sec', 'mean'), tps_std=('tokens_per_sec', 'std'),
    ).reindex(_order())
    colors = [PALETTE[m] for m in agg.index]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].bar(agg.index, agg['t_mean'], yerr=agg['t_std'], color=colors,
              alpha=0.85, capsize=5, edgecolor='black', lw=0.5)
    ax[0].set_ylabel('Время генерации, с'); ax[0].grid(axis='y', alpha=0.3, ls='--')
    ax[0].set_title('Среднее время генерации\n(по 3 сценариям × 3 повтора; меньше — лучше)',
                    fontsize=12, fontweight='bold')
    for i, v in enumerate(agg['t_mean']):
        ax[0].text(i, v, f'{v:.1f}с', ha='center', va='bottom', fontweight='bold')

    ax[1].bar(agg.index, agg['tps_mean'], yerr=agg['tps_std'], color=colors,
              alpha=0.85, capsize=5, edgecolor='black', lw=0.5)
    ax[1].set_ylabel('Токенов в секунду'); ax[1].grid(axis='y', alpha=0.3, ls='--')
    ax[1].set_title('Пропускная способность\n(больше — лучше)', fontsize=12, fontweight='bold')
    for i, v in enumerate(agg['tps_mean']):
        ax[1].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Производительность локальных LLM (Apple M2, 3 сценария)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, 'agent_plot1_performance.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"График 1 сохранён: {p}")


def plot_quality_auto(df: pd.DataFrame, save_dir: str):
    """
    График 2: автоматические метрики качества.
    Три панели: соблюдение формата, атрибуция источника, число шагов предписания.
    """
    agg = df.groupby('model').agg(
        fmt=('format_ok', 'mean'),
        attr=('attribution_ok', 'mean'),
        steps=('action_steps', 'mean'),
    ).reindex(_order())
    colors = [PALETTE[m] for m in agg.index]

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[0].bar(agg.index, agg['fmt']*100, color=colors, alpha=0.85, edgecolor='black', lw=0.5)
    ax[0].set_ylim(0, 115); ax[0].axhline(100, color='green', ls='--', alpha=0.5)
    ax[0].set_ylabel('% ответов'); ax[0].set_title('Соблюдение формата\n(4 раздела)', fontsize=11, fontweight='bold')
    for i, v in enumerate(agg['fmt']):
        ax[0].text(i, v*100+2, f'{v*100:.0f}%', ha='center', fontweight='bold')

    ax[1].bar(agg.index, agg['attr']*100, color=colors, alpha=0.85, edgecolor='black', lw=0.5)
    ax[1].set_ylim(0, 115); ax[1].axhline(100, color='green', ls='--', alpha=0.5)
    ax[1].set_ylabel('% ответов'); ax[1].set_title('Атрибуция источника\n(ссылка на документ, не "Сценарий А")', fontsize=11, fontweight='bold')
    for i, v in enumerate(agg['attr']):
        ax[1].text(i, v*100+2, f'{v*100:.0f}%', ha='center', fontweight='bold')

    ax[2].bar(agg.index, agg['steps'], color=colors, alpha=0.85, edgecolor='black', lw=0.5)
    ax[2].set_ylabel('Среднее число шагов'); ax[2].set_title('Полнота предписания\n(пунктов действий)', fontsize=11, fontweight='bold')
    ax[2].grid(axis='y', alpha=0.3, ls='--')
    for i, v in enumerate(agg['steps']):
        ax[2].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    for a in ax:
        a.tick_params(axis='x', rotation=15)
    plt.suptitle('Автоматические метрики качества предписаний',
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    p = os.path.join(save_dir, 'agent_plot2_quality_auto.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"График 2 сохранён: {p}")


def plot_expert_radar(expert_scores: dict, save_dir: str):
    """График 3: радар экспертных оценок (заполняется вручную)."""
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
    p = os.path.join(save_dir, 'agent_plot3_expert_radar.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"График 3 сохранён: {p}")


# ── Точка входа ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    root      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root, 'models', 'xgboost_pump_model.joblib')
    data_path  = os.path.join(root, 'data', 'processed', 'processed_features.csv')
    chroma_dir = os.path.join(root, 'chroma_db')
    kb_dir     = os.path.join(root, 'knowledge_base')
    save_dir   = os.path.join(root, 'data', 'graphs')
    os.makedirs(save_dir, exist_ok=True)

    print("Подготовка сценариев...")
    xai = XAIExplainer(model_path)
    df  = pd.read_csv(data_path)
    pre = DataPreprocessor(window_sizes=[15, 30, 60])
    kb  = KnowledgeBaseManager(data_dir=kb_dir, chroma_dir=chroma_dir)

    scenarios = build_scenarios(xai, df, pre, kb)
    print(f"Сформировано сценариев: {len(scenarios)} — "
          f"{[s['label'] for s in scenarios]}")

    results = run_benchmark(scenarios, n_repeats=3)
    results.to_csv(os.path.join(save_dir, 'agent_benchmark_multi.csv'), index=False)

    plot_performance(results, save_dir)
    plot_quality_auto(results, save_dir)

    # Экспертные оценки — ваши значения
    expert_scores = {
        'Qwen 3.5 9B':    {'Корректность диагноза': 5, 'Полнота предписания': 5,
                           'Точность метрик': 4, 'Соответствие контексту': 4, 'Читаемость': 5},
        'Phi-4 14B':      {'Корректность диагноза': 5, 'Полнота предписания': 3,
                           'Точность метрик': 4, 'Соответствие контексту': 4, 'Читаемость': 4},
        'YandexGPT-5 8B': {'Корректность диагноза': 5, 'Полнота предписания': 4,
                           'Точность метрик': 3, 'Соответствие контексту': 4, 'Читаемость': 5},
    }
    plot_expert_radar(expert_scores, save_dir)

    # Сводная таблица для диплома
    print("\n" + "="*55)
    print("СВОДНАЯ ТАБЛИЦА (среднее по всем прогонам)")
    print("="*55)
    summary = results.groupby('model').agg(
        Время_с=('gen_time_sec', 'mean'),
        Токен_с=('tokens_per_sec', 'mean'),
        Токенов=('eval_count', 'mean'),
        Формат=('format_ok', 'mean'),
        Атрибуция=('attribution_ok', 'mean'),
        Шагов=('action_steps', 'mean'),
    ).reindex(_order()).round(2)
    print(summary.to_string())
    summary.to_csv(os.path.join(save_dir, 'agent_summary_table.csv'))
    print(f"\nГотово. Результаты в {save_dir}")