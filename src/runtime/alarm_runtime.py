"""
Runtime-слой управления тревогами поверх предсказаний ML:
  * PumpAlarmFSM   — конечный автомат тревог с дебаунсом (анти-дребезг);
  * Incident       — жизненный цикл инцидента и кеш предписаний;
  * AlarmJournal   — журнал всех событий, включая подавленные
                     state-based фильтрацией сигналы (требование
                     ФЗ № 116-ФЗ / ГОСТ Р 22.1.12-2005: скрытые сигналы
                     сохраняются в архиве).

Ключевой принцип — ГЕЙТИНГ LLM: тяжёлая цепочка XAI -> RAG -> агент
запускается ТОЛЬКО на подтверждённом переходе состояния
(0->1, 1->2, 0->2), один раз на стадию инцидента. Инференс ML-моделей
на каждый тик дёшев; генерация предписания (~20 с) — нет.
"""

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional

SEVERITY_LABELS = {0: "Норма", 1: "Предупреждение", 2: "Авария"}
STAGE_BY_SEVERITY = {1: "warning", 2: "critical"}

FAULT_LABELS = {
    "overheat": "Тип А — Перегрев",
    "cavitation": "Тип Б — Кавитация",
    "electrical": "Тип В — Электрика",
}


@dataclass
class JournalEvent:
    ts: str
    pump_id: str
    kind: str                 # 'transition' | 'suppressed' | 'ack' | 'reset'
    from_state: Optional[int] = None
    to_state: Optional[int] = None
    fault_type: Optional[str] = None
    note: str = ""

    @property
    def label(self) -> str:
        if self.kind == "transition":
            return (
                f"{SEVERITY_LABELS.get(self.from_state, '?')} -> "  # type: ignore
                f"{SEVERITY_LABELS.get(self.to_state, '?')}"        # type: ignore
            )
        if self.kind == "suppressed":
            return "Сигнал подавлен (state-based)"
        if self.kind == "ack":
            return "Квитировано оператором"
        return self.kind


class AlarmJournal:
    def __init__(self, maxlen: int = 2000):
        self.events: List[JournalEvent] = []
        self._maxlen = maxlen

    def add(self, ev: JournalEvent) -> None:
        self.events.append(ev)
        if len(self.events) > self._maxlen:
            self.events = self.events[-self._maxlen:]

    def last(self, n: int = 10, kinds: Optional[List[str]] = None) -> List[JournalEvent]:
        evs = self.events if not kinds else [e for e in self.events if e.kind in kinds]
        return list(reversed(evs[-n:]))

    def count(self, kind: str) -> int:
        return sum(1 for e in self.events if e.kind == kind)


@dataclass
class Incident:
    """Один инцидент: от первого подтверждённого Предупреждения до сброса."""

    incident_id: int
    pump_id: str
    opened_ts: str                       # время первого перехода в >=1
    stage: int = 1                       # текущая стадия: 1 или 2
    stage_ts: str = ""                   # время входа в текущую стадию
    fault_type: Optional[str] = None
    acknowledged: bool = False
    # кеш предписаний по стадиям: {1: текст, 2: текст}
    prescriptions: Dict[int, str] = field(default_factory=dict)
    # трассировка извлечения по стадиям (для инженерной вкладки)
    retrieval_traces: Dict[int, list] = field(default_factory=dict)
    # SymptomVector по стадиям (объект xai_module)
    symptom_vectors: Dict[int, object] = field(default_factory=dict)

    def needs_prescription(self) -> bool:
        return self.stage not in self.prescriptions

    @property
    def stage_label(self) -> str:
        return SEVERITY_LABELS[self.stage]

    @property
    def fault_label(self) -> str:
        return FAULT_LABELS.get(self.fault_type or "", "определяется…")


class PumpAlarmFSM:
    """Дебаунс-автомат тревог для одного потока предсказаний по парку.

    Переход состояния подтверждается только после confirm_ticks
    одинаковых предсказаний подряд (защита от мерцающих алармов,
    практика рационализации по ISA 18.2 / EEMUA 191).

    Эскалация (вверх) подтверждается быстрее, чем деэскалация (вниз):
    пропустить аварию дороже, чем подержать предупреждение лишнюю минуту.
    """

    def __init__(self, confirm_up: int = 2, confirm_down: int = 5,
                 max_closed: int = 200):
        self.confirm_up = confirm_up
        self.confirm_down = confirm_down
        self._state: Dict[str, int] = {}
        self._pending: Dict[str, tuple] = {}   # pump_id -> (candidate, count)
        self._incidents: Dict[str, Incident] = {}
        # Закрытые инциденты копятся для истории. Кап (> кап ss.events в UI с
        # запасом) защищает ядро от роста памяти на длинном live-прогоне.
        self._closed: List[Incident] = []
        self._max_closed = max_closed
        self._ids = itertools.count(1)
        self.journal = AlarmJournal()

    def state(self, pump_id: str) -> int:
        return self._state.get(pump_id, 0)

    def incident(self, pump_id: str) -> Optional[Incident]:
        return self._incidents.get(pump_id)

    def closed_incidents(self) -> List[Incident]:
        return list(self._closed)

    def all_incidents(self) -> List[Incident]:
        """Все инциденты (закрытые + активные), новейшие сверху.

        Источник для истории предписаний в выдвижной створке.
        """
        merged = list(self._closed) + list(self._incidents.values())
        return sorted(merged, key=lambda i: i.incident_id, reverse=True)

    def active_alarm_count(self) -> int:
        return sum(1 for s in self._state.values() if s == 2)

    def active_warning_count(self) -> int:
        return sum(1 for s in self._state.values() if s == 1)

    def update(
        self,
        pump_id: str,
        ts: str,
        predicted: int,
        suppressed: bool = False,
        fault_type: Optional[str] = None,
    ) -> Optional[Incident]:
        """Обработать один тик предсказания.

        Возвращает Incident, если на этом тике произошла подтверждённая
        ЭСКАЛАЦИЯ и для новой стадии требуется предписание (триггер для
        запуска XAI -> RAG -> агент). Иначе None.
        """
        
        if suppressed:
            # сигнал подавлен контекстной фильтрацией (пуск/простой):
            # на дашборд не идёт, в архив — обязательно
            self.journal.add(JournalEvent(ts, pump_id, "suppressed",
                                          note="режим пуска/простоя"))
            return None

        cur = self.state(pump_id)
        if predicted == cur:
            self._pending.pop(pump_id, None)
            return None

        cand, cnt = self._pending.get(pump_id, (predicted, 0))
        cnt = cnt + 1 if cand == predicted else 1
        self._pending[pump_id] = (predicted, cnt)

        need = self.confirm_up if predicted > cur else self.confirm_down
        if cnt < need:
            return None

        # подтверждённый переход 
        self._pending.pop(pump_id, None)
        self._state[pump_id] = predicted
        self.journal.add(JournalEvent(ts, pump_id, "transition",
                                      from_state=cur, to_state=predicted,
                                      fault_type=fault_type))

        if predicted == 0:
            inc = self._incidents.pop(pump_id, None)
            if inc is not None:
                self._closed.append(inc)
                if len(self._closed) > self._max_closed:
                    self._closed = self._closed[-self._max_closed:]
            self.journal.add(JournalEvent(ts, pump_id, "reset",
                                          note="возврат в норму"))
            return None

        # эскалация: открыть или продвинуть инцидент
        inc = self._incidents.get(pump_id)
        if inc is None:
            inc = Incident(
                incident_id=next(self._ids),
                pump_id=pump_id,
                opened_ts=ts,
                stage=predicted,
                stage_ts=ts,
                fault_type=fault_type,
            )
            self._incidents[pump_id] = inc
        else:
            inc.stage = predicted
            inc.stage_ts = ts
            inc.acknowledged = False
            if fault_type:
                inc.fault_type = fault_type

        return inc if inc.needs_prescription() else None

    def acknowledge(self, pump_id: str, ts: str) -> None:
        inc = self._incidents.get(pump_id)
        if inc is not None:
            inc.acknowledged = True
            self.journal.add(JournalEvent(ts, pump_id, "ack"))