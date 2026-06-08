"""
Многосценарный бенчмарк LLM-моделей для AI-агента
=================================================
Прогоняет три модели (qwen3.5:9b, phi4:14b, yandex-gpt-5-lite:8b) на наборе
диагностических сценариев, собирает объективные метрики и строит графики
для диплома.

  - Три РАЗНЫХ диагностических сценария вместо одного повторённого:
      1. Перегрев + рост тока (Сценарий А: критический перегрев)
      2. Деградация по вибрации (Сценарий Б: ранняя кавитация, стадия Warning)
      3. Авария другого агрегата (вариативность)
    Это проверяет, РАЗЛИЧАЕТ ли модель типы отказов, а не выдаёт одно и то же.
  - Каждый сценарий несёт и контекст по симптомам (с подмешиванием типа отказа),
    и график ТОиР по pump_id — чтобы тестировать агента на полном входе.
  - Автометрики: соблюдение формата, атрибуция источника, число шагов предписания.
  - Усреднение метрик по сценариям x повторам.
"""

import os
import sys
import re
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ai_agent import DiagnosticAgent
from src.xai_module import XAIExplainer
from src.data_preprocessor import DataPreprocessor
from src.rag_database import KnowledgeBaseManager
from config.settings import (LLM_MODELS, MODEL_LABELS, TOIR_WORK_MARKERS,
                             REQUIRED_SECTIONS, SOURCE_MARKERS, WINDOW_SIZES,
                             DIRECTIONS, FMT)
from visualisation_instruments import (ai_vis_order, plot_performance, 
                                       plot_quality_auto, plot_summary_heatmap)


# Автоматические метрики качества 

def check_format(text: str) -> bool:
    """Все 4 обязательных раздела присутствуют в ответе."""

    return all(any(sec in line for line in text.split('\n')) 
               for sec in REQUIRED_SECTIONS)


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
    m = re.search(r'ПРЕДПИСАНИЕ.*?(?=РЕКОМЕНДАЦИИ|ПЛАНОВЫЙ|ИНФОРМАЦИЯ|УРОВЕНЬ|$)',
                  text, re.S | re.I)
    if not m:
        return 0
    block = m.group(0)
    # Считаем нумерованные пункты "1.", "2." ...
    return len(re.findall(r'^\s*\d+\.', m.group(0), re.M))

def check_toir_is_works(text: str) -> bool:
    """
    В разделе РЕКОМЕНДАЦИИ ТОиР должны быть РЕМОНТНЫЕ РАБОТЫ, 
    а не дата планового ремонта. Автоматизирует исходный дефект: 
    модель выводила в ТОиР график вместо работ.
    """

    m = re.search(r'РЕКОМЕНДАЦИИ ТОиР.*?(?=ПЛАНОВЫЙ|ИНФОРМАЦИЯ|$)', 
                  text, re.S | re.I)
    if not m:
        return False
    block = m.group(0).lower()
    return any(w in block for w in TOIR_WORK_MARKERS)


# Подготовка сценариев 

def build_scenarios(xai, df, preprocessor, kb) -> list:
    """
    Три РАЗНОТИПНЫХ сценария — по одному на каждый тип отказа и разные стадии,
    чтобы оценка не сводилась к одному типу:
      1) Авария + кавитация
      2) Предупреждение + перегрев
      3) Авария + электрика
    """

    fc = preprocessor.FEATURE_COLS
    for col in ('timestamp', 'fault_type', 'target'):
        if col not in df.columns:
            raise KeyError(f"В датасете нет колонки '{col}' — проверьте препроцессор.")

    plan = [
        (2, 'cavitation', 'Авария — кавитация'),
        (1, 'overheat',   'Предупреждение — перегрев'),
        (2, 'electrical', 'Авария — электрика'),
    ]

    scenarios = []
    for target, fault_type, label in plan:
        sub = df[(df['target'] == target) & (df['fault_type'] == fault_type)]
        if len(sub) == 0:
            print(f"  [WARN] нет строк для «{label}» "
                  f"(target={target}, fault_type={fault_type}) — пропущен.")
            continue
        rec = sub.iloc[-1]
        row = sub.iloc[-1:][fc]
        sv = xai.explain_prediction(row, pump_id=str(rec['pump_id']),
                                    timestamp=str(rec['timestamp']), top_k=5)
        scenarios.append(_make_scenario(sv, kb, label))

    return scenarios

def _make_scenario(sv, kb, label) -> dict:
    """Собирает RAG-контекст (с типом отказа), работы ТОиР и график ТОиР по pump_id."""

    symptom_dict = {
        'critical_probability': sv.critical_probability,
        'inferred_fault': sv.inferred_fault, # подмешивает тип в прескриптивный запрос
        'top_symptoms': [{'sensor': s.sensor, 'value': s.value,
                          'shap_weight': s.shap_weight} for s in sv.top_symptoms]
    }
    rag_results = kb.search_by_symptoms(symptom_dict, k=4)
    repair_results = kb.search_repair_works(sv.inferred_fault, k=2)
    schedule_results = kb.search_maintenance_schedule(sv.pump_id, k=2)
    operator_results = kb.search_operator_actions(sv.inferred_fault, k=2)
    return {'sv': sv, 'rag': rag_results, 'repair': repair_results,
            'schedule': schedule_results, 'label': label, 'operator': operator_results}


def run_benchmark(scenarios: list, n_repeats: int = 3) -> pd.DataFrame:
    rows = []
    for model in LLM_MODELS:
        label = MODEL_LABELS.get(model, model)
        print(f"\n{'='*55}\nМодель: {label}\n{'='*55}")
        agent = DiagnosticAgent(model_name=model)

        print("  Прогрев...")
        try:
            agent.generate_prescription(scenarios[0]['sv'], scenarios[0]['rag'],
                                        scenarios[0]['schedule'], scenarios[0]['repair'],
                                        scenarios[0]['operator'])
        except Exception as e:
            print(f"    [WARN] прогрев: {e}")

        for sc in scenarios:
            for rep in range(n_repeats):
                resp = agent.generate_prescription(sc['sv'], sc['rag'],
                                                   sc['schedule'], sc['repair'], 
                                                   sc['operator'])
                if resp.error:
                    print(f"  [{sc['label']}] rep{rep}: ОШИБКА — {resp.error}")
                    continue

                fmt = check_format(resp.raw_text)
                attr = check_attribution(resp.raw_text)
                steps = count_action_steps(resp.raw_text)
                toir = check_toir_is_works(resp.raw_text)

                rows.append({
                    'model': label,
                    'scenario': sc['label'],
                    'repeat': rep,
                    'gen_time_sec': resp.gen_time_sec,
                    'tokens_per_sec': resp.tokens_per_sec,
                    'eval_count': resp.eval_count,
                    'prompt_tokens': resp.prompt_eval_count,
                    'answer_len': len(resp.raw_text),
                    'format_ok': fmt,
                    'attribution_ok': attr,
                    'action_steps': steps,
                    'toir_is_works': toir,
                    'raw_text': resp.raw_text,
                })
                print(f"  [{sc['label']}] rep{rep}: {resp.gen_time_sec}с, "
                      f"{resp.tokens_per_sec} ток/с, формат={'OK' if fmt else 'НЕТ'}, "
                      f"источник={'OK' if attr else 'НЕТ'}, шагов={steps}, "
                      f"ТОиР-работы={'OK' if toir else 'НЕТ'}")
    return pd.DataFrame(rows)


# Точка входа 

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root, 'models', 'xgboost_pump_model.joblib')
    fault_model_path = os.path.join(root, 'models', 'fault_xgboost_model.joblib')
    data_path = os.path.join(root, 'data', 'processed', 'preprocessed_pumps_dataset.csv')
    chroma_dir = os.path.join(root, 'chroma_db')
    kb_dir = os.path.join(root, 'knowledge_base')
    save_graphs_dir = os.path.join(root, 'data', 'graphs')
    os.makedirs(save_graphs_dir, exist_ok=True)
    save_tables_dir = os.path.join(root, 'data', 'tables')
    os.makedirs(save_tables_dir, exist_ok=True)    

    print("Подготовка сценариев...")
    xai = XAIExplainer(model_path, fault_model_path=fault_model_path)
    df = pd.read_csv(data_path)
    pre = DataPreprocessor(window_sizes=WINDOW_SIZES)
    kb = KnowledgeBaseManager(data_dir=kb_dir, chroma_dir=chroma_dir)

    scenarios = build_scenarios(xai, df, pre, kb)
    print(f"Сформировано сценариев: {len(scenarios)} — {[s['label'] for s in scenarios]}")

    results = run_benchmark(scenarios, n_repeats=3)
    results.to_csv(os.path.join(save_tables_dir, 
                                'agent_benchmark_multi.csv'), index=False)

    plot_performance(results, save_graphs_dir)
    plot_quality_auto(results, save_graphs_dir)

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
        ТОиР_работы=('toir_is_works', 'mean'),
    ).reindex(ai_vis_order()).round(2)
    print(summary.to_string())
    summary.to_csv(os.path.join(save_tables_dir, 'agent_summary_table.csv'))

    plot_summary_heatmap(summary, DIRECTIONS, FMT, save_graphs_dir)

    print(f"\nГотово. Графики сохранены в {save_graphs_dir}")
    print(f"\n        Таблицы сохранены в {save_tables_dir}")