"""
Регрессионный guard (v2): целостность сценарной И СТАДИЙНОЙ привязки RAG
========================================================================
Расширяет прежний тест на измерение СТАДИИ. После ввода двух операторных блоков
на сценарий («Предупреждение»/«Авария») поиск действий обязан возвращать блок
именно нужного типа отказа И нужной стадии.

Что проверяется:
  1. Чанки регламента размечены fault_type; у operator-чанков есть stage из
     {warning, critical}; для КАЖДОГО типа отказа присутствуют ОБА стадийных блока.
  2. Для каждого типа × каждой стадии search_operator_actions(fault, stage):
       - непустой результат;
       - каждый чанк: fault_type == fault, sop_part == 'operator', stage == запрошенной;
       - содержит характерную лексику своей (fault, stage) и НЕ содержит лексику
         другой стадии того же типа (защита от перепутывания Предупреждение↔Авария).
  3. search_repair_works по-прежнему заперт по сценарию (работы стадийно-независимы).

Запуск:
    pytest tests/test_rag_scenario_integrity.py -v
  или:  python tests/test_rag_scenario_integrity.py
Предусловие: база собрана со стадийной разметкой (kb.build_database(reset=True)).
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import FAULT_TYPES, DOC_TYPE_MAP
from src.rag.rag_database import KnowledgeBaseManager
from langchain_chroma import Chroma

STAGES = ('warning', 'critical')

# Лексика, уникальная для каждой пары (тип отказа, стадия) в tm_regulation.md.
# Используется и как позитивная проверка, и (для другой стадии) как негативная.
STAGE_OPERATOR_KW = {
    'overheat':   {'warning': ['смотров', 'резервн'],          'critical': ['заклинивания', 'картере']},
    'cavitation': {'warning': ['подпор', 'npsha'],             'critical': ['напорную задвижку']},
    'electrical': {'warning': ['энергодиспетчер', 'наблюдат'], 'critical': ['обесточить']},
}

_CHROMA_DIR = os.path.join(_PROJECT_ROOT, 'artifacts', 'chroma_db')
_KB_DIR = os.path.join(_PROJECT_ROOT, 'knowledge_base')
_kb_cache = {}


def _get_kb() -> KnowledgeBaseManager:

    if 'kb' not in _kb_cache:
        if not os.path.isdir(_CHROMA_DIR) or not os.listdir(_CHROMA_DIR):
            raise RuntimeError(
                f"Векторная база не найдена в {_CHROMA_DIR}. "
                f"Сначала соберите её со стадийной разметкой: kb.build_database(reset=True).")
        _kb_cache['kb'] = KnowledgeBaseManager(
            data_dir=_KB_DIR, chroma_dir=_CHROMA_DIR, doc_type_map=DOC_TYPE_MAP)
    return _kb_cache['kb']


def _clean(text: str) -> str:

    return text.replace('passage: ', '', 1).lower()


# Тест 1: разметка fault_type + stage; оба стадийных блока на каждый тип
def test_sop_chunks_tagged_with_fault_and_stage():

    kb = _get_kb()
    db = Chroma(persist_directory=kb.chroma_dir, embedding_function=kb.embeddings)
    meta = db._collection.get().get('metadatas') or []
    sop = [m for m in meta if m.get('doc_type') == 'sop']
    assert sop, "В базе нет чанков регламента (doc_type='sop')."

    tagged = [m for m in sop if m.get('fault_type')]
    assert tagged, ("SOP-чанки не размечены fault_type — проверьте заголовки сценариев "
                    "и пересоберите базу (reset=True).")
    present_faults = {m['fault_type'] for m in tagged}  # type: ignore
    assert not (set(FAULT_TYPES) - present_faults), f"В разметке отсутствуют типы: {sorted(set(FAULT_TYPES) - present_faults)}."    # type: ignore

    op = [m for m in sop if m.get('sop_part') == 'operator']
    assert op, "Нет operator-чанков SOP — проверьте разметку sop_part."
    for m in op:
        assert m.get('stage') in STAGES, \
            f"operator-чанк без валидной стадии: stage={m.get('stage')!r}."

    for fault in FAULT_TYPES:
        stages = {m.get('stage') for m in op if m.get('fault_type') == fault}   # type: ignore
        assert set(STAGES) <= stages, \
            f"У типа '{fault}' отсутствует операторный блок стадии: {set(STAGES) - stages}."    # type: ignore


# Тест 2: действия оператора заперты по сценарию И по стадии
def test_operator_actions_locked_by_fault_and_stage():

    kb = _get_kb()
    for fault in FAULT_TYPES:
        for stage in STAGES:
            res = kb.search_operator_actions(fault, stage=stage, k=2)
            assert res, f"search_operator_actions('{fault}', stage='{stage}') вернул пусто."

            own = STAGE_OPERATOR_KW[fault][stage]
            other = 'critical' if stage == 'warning' else 'warning'
            foreign = STAGE_OPERATOR_KW[fault][other]

            for doc, _score in res:
                md = doc.metadata
                assert md.get('fault_type') == fault, (
                    f"[{fault}/{stage}] действия из чужого сценария "
                    f"(fault_type={md.get('fault_type')!r}).")
                assert md.get('sop_part') == 'operator', (
                    f"[{fault}/{stage}] возвращён не операторный подраздел "
                    f"(sop_part={md.get('sop_part')!r}).")
                assert md.get('stage') == stage, (
                    f"[{fault}/{stage}] возвращён блок чужой стадии "
                    f"(stage={md.get('stage')!r}). Стадийная привязка нарушена.")

                text = _clean(doc.page_content)
                assert any(kw in text for kw in own), (
                    f"[{fault}/{stage}] нет характерной лексики {own} — "
                    f"возможно, вернулся не тот блок.")
                leaked = [kw for kw in foreign if kw in text]
                assert not leaked, (
                    f"[{fault}/{stage}] просочилась лексика стадии '{other}': {leaked}.")


# Тест 3: работы ТОиР заперты по сценарию (стадийно-независимы)
def test_repair_works_locked_by_fault():
    
    kb = _get_kb()
    for fault in FAULT_TYPES:
        res = kb.search_repair_works(fault, k=2)
        assert res, f"search_repair_works('{fault}') вернул пусто."
        for doc, _score in res:
            assert doc.metadata.get('fault_type') == fault, (
                f"Работы ТОиР для '{fault}' из чужого сценария "
                f"(fault_type={doc.metadata.get('fault_type')!r}).")


if __name__ == "__main__":
    tests = [test_sop_chunks_tagged_with_fault_and_stage,
             test_operator_actions_locked_by_fault_and_stage,
             test_repair_works_locked_by_fault]
    failed = 0
    print("=" * 59)
    print("Регрессионный guard: сценарная + стадийная привязка RAG")
    print("=" * 59)
    for t in tests:
        try:
            t()
            print(f"  [OK]   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {t.__name__}\n         {e}")
        except Exception as e:
            failed += 1
            print(f"  [ERR]  {t.__name__}: {type(e).__name__}: {e}")
    print("=" * 59)
    print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ" if failed == 0 else f"ПРОВАЛЕНО ПРОВЕРОК: {failed}")
    sys.exit(1 if failed else 0)