"""
Многосценарный бенчмарк LLM-моделей для AI-агента
=================================================
Прогоняет три модели (qwen3.5:9b, phi4:14b, yandex-gpt-5-lite:8b) покрывающих 
КАЖДЫЙ тип отказа на ОБЕИХ стадиях (Предупреждение / Авария), собирает объективные
метрики и строит графики.

  - План сценариев = {overheat, cavitation, electrical} x {warning, critical} (до 6 шт.),
    чтобы можно было СРАВНИВАТЬ поведение по стадиям, а не только по типам.
  - Каждый сценарий несёт и контекст по симптомам (с подмешиванием типа отказа),
    и график ТОиР по pump_id — чтобы тестировать агента на полном входе.
  - Автометрики: соблюдение формата, атрибуция источника, число шагов предписания.
  - Усреднение метрик по сценариям x повторам.
"""

import os
import sys
import re
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ai_agent import DiagnosticAgent
from src.xai_module import XAIExplainer
from src.data_preprocessor import DataPreprocessor
from src.rag_database import KnowledgeBaseManager, resolve_stage
from config.settings import (LLM_MODELS, MODEL_LABELS, TOIR_WORK_MARKERS,
                             REQUIRED_SECTIONS, SOURCE_MARKERS, WINDOW_SIZES,
                             DIRECTIONS, FMT, EMERGENCY_MARKERS, DECISIVE_MARKERS)
from visualisation_instruments import (ai_vis_order, plot_performance, 
                                       plot_quality_auto, plot_summary_heatmap,
                                       plot_stage_breakdown)

_EMERGENCY_MARKERS = EMERGENCY_MARKERS
_DECISIVE_MARKERS = DECISIVE_MARKERS


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


def _prescription_block(text: str) -> str:
    """Возвращает текст раздела ПРЕДПИСАНИЕ (до следующего заголовка)."""

    m = re.search(r'ПРЕДПИСАНИЕ.*?(?=РЕКОМЕНДАЦИИ|ПЛАНОВЫЙ|$)', text, re.S | re.I)

    return m.group(0) if m else ''


def count_action_steps(text: str) -> int:
    """Число нумерованных пунктов в разделе ПРЕДПИСАНИЕ."""

    block = _prescription_block(text)

    return len(re.findall(r'^\s*\d+\.', block, re.M))


def check_toir_is_works(text: str) -> bool:
    """В РЕКОМЕНДАЦИЯХ ТОиР — ремонтные работы, а не дата планового ремонта."""

    m = re.search(r'РЕКОМЕНДАЦИИ ТОиР.*?(?=ПЛАНОВЫЙ|ИНФОРМАЦИЯ|$)', text, re.S | re.I)

    if not m:
        return False
    block = m.group(0).lower()

    return any(w in block for w in TOIR_WORK_MARKERS)


def check_stage_appropriate(text: str, stage: str) -> bool:
    """
    Стадийная уместность предписания:
      - 'warning' → НЕ должно быть команды аварийного останова 
        (упреждение, а не реакция);
      - 'critical' → должно присутствовать решительное действие.
    Эта метрика напрямую измеряет успех стадийной дифференциации.
    """

    block = _prescription_block(text).lower()

    if stage == 'warning':
        return not any(w in block for w in _EMERGENCY_MARKERS)
    
    if stage == 'unknown':
        low = text.lower()
        return ('ручная инспекция' in low) and (count_action_steps(text) == 0)
    
    return any(w in block for w in _DECISIVE_MARKERS)


# Подготовка сценариев 

def build_scenarios(xai, df, preprocessor, kb) -> list:
    """Каждый тип отказа на обеих стадиях: до 6 сценариев."""

    fc = preprocessor.FEATURE_COLS
    for col in ('timestamp', 'fault_type', 'target'):
        if col not in df.columns:
            raise KeyError(f"В датасете нет колонки '{col}' — проверьте препроцессор.")
 
    faults = [('overheat', 'перегрев'), ('cavitation', 'кавитация'),
              ('electrical', 'электрика')]
    stages = [(1, 'Предупреждение'), (2, 'Авария')]
 
    plan = []
    for fault_type, fault_ru in faults:
        for target, stage_ru in stages:
            plan.append((target, fault_type, f"{stage_ru} — {fault_ru}"))
 
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
    """Собирает RAG-контекст (тип + СТАДИЯ), работы ТОиР и график ТОиР по pump_id."""

    symptom_dict = {
        'critical_probability': sv.critical_probability,
        'inferred_fault': sv.inferred_fault,
        'top_symptoms': [{'sensor': s.sensor, 'value': s.value,
                          'shap_weight': s.shap_weight} for s in sv.top_symptoms]
    }
    stage = resolve_stage(sv)
    operator_results = (kb.search_operator_actions(sv.inferred_fault, stage=stage, k=2)
                        if stage != 'unknown' else [])
 
    rag_results = kb.search_by_symptoms(symptom_dict, k=4)
    repair_results = kb.search_repair_works(sv.inferred_fault, k=2)
    schedule_results = kb.search_maintenance_schedule(sv.pump_id, k=2)
 
    return {'sv': sv, 'rag': rag_results, 'repair': repair_results,
            'schedule': schedule_results, 'operator': operator_results,
            'label': label, 'stage': stage}


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
                stage_ok = check_stage_appropriate(resp.raw_text, sc['stage'])
 
                rows.append({
                    'model': label,
                    'scenario': sc['label'],
                    'stage': sc['stage'],
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
                    'stage_appropriate': stage_ok,
                    'raw_text': resp.raw_text,
                })
                print(f"  [{sc['label']}] rep{rep}: {resp.gen_time_sec}с, "
                      f"формат={'OK' if fmt else 'НЕТ'}, источник={'OK' if attr else 'НЕТ'}, "
                      f"шагов={steps}, ТОиР={'OK' if toir else 'НЕТ'}, "
                      f"стадия={'OK' if stage_ok else 'НЕТ'}")
                
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
    results.to_csv(os.path.join(save_tables_dir, 'agent_benchmark_multi.csv'), index=False)
 
    plot_performance(results, save_graphs_dir)
    plot_quality_auto(results, save_graphs_dir)
    plot_stage_breakdown(results, save_graphs_dir)
 
    # Сводка по моделям (как раньше) + добавлена метрика стадийной уместности.
    print("\n" + "=" * 55)
    print("СВОДНАЯ ТАБЛИЦА (среднее по всем прогонам)")
    print("=" * 55)
    agg_spec = dict(
        Время_с=('gen_time_sec', 'mean'),
        Токен_с=('tokens_per_sec', 'mean'),
        Токенов=('eval_count', 'mean'),
        Формат=('format_ok', 'mean'),
        Атрибуция=('attribution_ok', 'mean'),
        Шагов=('action_steps', 'mean'),
        ТОиР_работы=('toir_is_works', 'mean'),
        Стадия_уместность=('stage_appropriate', 'mean'),
    )
    summary = results.groupby('model').agg(**agg_spec).reindex(ai_vis_order()).round(2)
    print(summary.to_string())
    summary.to_csv(os.path.join(save_tables_dir, 'agent_summary_table.csv'))
 
    # НОВОЕ: разбиение метрик по стадиям (model × stage) — для стадийных графиков диплома.
    print("\n" + "=" * 55)
    print("СВОДКА ПО СТАДИЯМ (model x stage)")
    print("=" * 55)
    stage_summary = (results.groupby(['stage', 'model']).agg(**agg_spec)
                     .round(2).sort_index())
    print(stage_summary.to_string())
    stage_summary.to_csv(os.path.join(save_tables_dir, 'agent_summary_by_stage.csv'))
 
    plot_summary_heatmap(summary, DIRECTIONS, FMT, save_graphs_dir)
 
    print(f"\nГотово. Графики: {save_graphs_dir}")
    print(f"        Таблицы: {save_tables_dir}")