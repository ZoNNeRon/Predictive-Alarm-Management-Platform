"""
Smoke-тест отладочного отката UI: ProtoBackend
==============================================
`app.py` при сбое инициализации боевого `PlatformBackend` (нет Ollama/ChromaDB/
моделей) молча падает на `ProtoBackend`. Если интерфейс прототипа отстал от того,
что дёргает UI, дашборд рушится вместо graceful-демо. Этот тест ЗАМОРАЖИВАЕТ
контракт: какие методы у backend и какие поля у возвращаемого `SymptomVector`
читает интерфейс — и проверяет, что прототип их все отдаёт.

Контракт собран из app.py (держать в синхроне при правках UI):
  • backend.<...>: process_tick, explain, prescription_stream, retrieval_trace,
    shap_figures; атрибут preproc с .feature_cols / .reset();
  • sv.<...>: probabilities[1] (drill-down предупреждения!), critical_probability,
    inferred_fault, fault_confidence, fault_probabilities, top_symptoms,
    fault_top_symptoms; у элемента симптома — feature/sensor/value/shap_weight;
  • retrieval_trace → list[dict] с ключами для таблицы инженера.

Тест headless: НЕ импортирует app.py (там st.set_page_config на верхнем уровне),
а кодирует контракт явными списками.

Запуск:
    pytest experiments/validation/protobackend_smoke_test.py -v
  или:  python experiments/validation/protobackend_smoke_test.py
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from src.runtime.platform_backend import ProtoBackend
from src.runtime.online_preprocessor import WARMUP_ROWS

# --- контракт, который UI ожидает от backend и SymptomVector ------------------
BACKEND_METHODS = ["process_tick", "explain", "prescription_stream",
                   "retrieval_trace", "shap_figures"]
SV_REQUIRED_ATTRS = ["pump_id", "predicted_class", "probabilities",
                     "critical_probability", "inferred_fault", "fault_confidence",
                     "fault_probabilities", "top_symptoms", "fault_top_symptoms"]
SYMPTOM_ATTRS = ["feature", "sensor", "value", "shap_weight"]
TRACE_KEYS = {"block", "doc", "distance", "fault_type", "sop_part", "stage"}

_cache = {}


def _proto() -> ProtoBackend:
    if "b" not in _cache:
        _cache["b"] = ProtoBackend()
    return _cache["b"]


def _row(pid: str, i: int, state: int, fault) -> pd.Series:
    return pd.Series({
        "pump_id": pid,
        "timestamp": f"2026-04-01 {i // 60:02d}:{i % 60:02d}:00",
        "state": state, "fault_type": fault,
        # лёгкая вариация, чтобы std != 0 на полном окне
        "vibration": 1.8 + i * 1e-3, "temperature": 70.0 + i * 1e-3,
        "current": 50.0 + i * 1e-3, "pressure": 1.5,
    })


def _drive_until_ready(backend, pid, state, fault, n=WARMUP_ROWS + 5):
    """Прогнать n строк через process_tick; вернуть последний (готовый) tick."""
    tick = None
    for i in range(n):
        tick = backend.process_tick(pid, _row(pid, i, state, fault))
    return tick


# Тест 1: у прототипа есть все методы и атрибут preproc, которые дёргает app.py
def test_protobackend_has_app_interface():
    b = _proto()
    for m in BACKEND_METHODS:
        assert callable(getattr(b, m, None)), f"ProtoBackend.{m} отсутствует/не вызываем."
    assert hasattr(b, "preproc"), "ProtoBackend.preproc отсутствует."
    assert hasattr(b.preproc, "feature_cols"), "preproc.feature_cols отсутствует."
    assert callable(getattr(b.preproc, "reset", None)), "preproc.reset отсутствует."


# Тест 2: process_tick отдаёт TickResult с полями, которые читает advance_stream
def test_process_tick_contract():
    b = ProtoBackend()                       # чистый инстанс — без общего буфера
    pid = "MNHV_TEST"
    # прогрев: первые WARMUP_ROWS тиков НЕ готовы
    first = b.process_tick(pid, _row(pid, 0, 2, "none"))
    assert first.ready is False, "Первый тик не должен быть готов (идёт прогрев)."
    tick = _drive_until_ready(b, pid, 3, "overheat")   # Degradation + перегрев
    for fld in ("ready", "severity", "raw_severity", "suppressed",
                "severity_proba", "fault_type", "fault_proba"):
        assert hasattr(tick, fld), f"TickResult.{fld} отсутствует."
    assert tick.ready is True
    assert tick.severity == 1, "Degradation должен маппиться в Предупреждение (1)."
    assert tick.fault_type == "overheat"


# Тест 3: explain отдаёт SymptomVector с ПОЛНЫМ набором полей, читаемых UI
def test_explain_symptomvector_contract():
    b = ProtoBackend()
    pid = "MNHV_W"
    _drive_until_ready(b, pid, 3, "overheat")          # предупреждение
    sv = b.explain(pid, "2026-04-01 01:05:00", 1)

    missing = [a for a in SV_REQUIRED_ATTRS if not hasattr(sv, a)]
    assert not missing, f"SymptomVector прототипа без полей, нужных UI: {missing}"

    # Точное воспроизведение опасного доступа из view_operator (drill-down пред-я):
    #   pw = sv.probabilities[1] * 100 if sv and len(sv.probabilities) > 2 else None
    assert len(sv.probabilities) > 2, "probabilities должен быть из 3 элементов [P0,P1,P2]."
    _ = sv.probabilities[1] * 100                       # не должно бросать

    # элементы симптомов — с полями для таблиц инженера
    for bag in (sv.top_symptoms, sv.fault_top_symptoms):
        assert bag, "top_symptoms / fault_top_symptoms не должны быть пустыми на инциденте."
        s0 = bag[0]
        miss = [a for a in SYMPTOM_ATTRS if not hasattr(s0, a)]
        assert not miss, f"Элемент симптома без полей {miss}."
    # view_operator: sv.fault_top_symptoms[0].feature
    assert isinstance(sv.fault_top_symptoms[0].feature, str)


# Тест 4: prescription_stream / retrieval_trace / shap_figures — формат для UI
def test_stream_trace_shap_contract():
    b = ProtoBackend()
    pid = "MNHV_A"
    _drive_until_ready(b, pid, 4, "overheat")          # авария
    sv = b.explain(pid, "2026-04-01 02:00:00", 2)

    text = "".join(b.prescription_stream(sv, "critical"))
    assert isinstance(text, str) and "ПРЕДПИСАНИЕ" in text, \
        "prescription_stream должен отдавать поток токенов с разделом ПРЕДПИСАНИЕ."

    trace = b.retrieval_trace(sv, "critical")
    assert isinstance(trace, list) and trace, "retrieval_trace должен вернуть непустой list."
    assert TRACE_KEYS.issubset(trace[0].keys()), \
        f"Запись трассы без ключей: {TRACE_KEYS - set(trace[0].keys())}"

    figs = b.shap_figures(pid)
    assert isinstance(figs, tuple) and len(figs) == 2, \
        "shap_figures должен возвращать 2-кортеж (sev, fault)."


if __name__ == "__main__":
    tests = [test_protobackend_has_app_interface,
             test_process_tick_contract,
             test_explain_symptomvector_contract,
             test_stream_trace_shap_contract]
    failed = 0
    print("=" * 59)
    print("Smoke-тест отката UI: ProtoBackend")
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
