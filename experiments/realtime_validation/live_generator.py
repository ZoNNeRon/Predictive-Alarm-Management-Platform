"""
live_generator.py — живой мультинасосный генератор телеметрии (реальное время).
================================================================================
Назначение: валидация системы в режиме реального времени. В отличие от
`data_generator.py` (пакетная генерация всего датасета) и `demo_stream.py`
(воспроизведение готового эпизода из CSV), этот модуль генерирует данные
ПОШАГОВО, «здесь и сейчас»: на каждый тик — по одной строке на каждый насос
парка, синхронно по единым модельным часам.

ВЕРНОСТЬ АЛГОРИТМУ. Поминутные формулы AR(1), сигнатуры трёх типов отказа,
startup-всплеск, аномалии датчиков и State Machine (Off/Startup/Healthy/
Degradation/Critical = 0..4) перенесены ОДИН-В-ОДИН из `src/data/data_generator.py`
(те же μ, φ, σ, linspace-тренды). Отличия — только структурные:
  • поминутный шаг вместо пакетных циклов (нужно для real-time и квитирования);
  • независимый RNG-поток на насос (`np.random.default_rng(seed)`) вместо
    глобального `np.random.seed` — распределение тика идентично датасету,
    значения не байт-в-байт (это и не требуется: модели обучены на распределении);
  • ускоренные тайминги/вероятности (RealtimeConfig) — чтобы валидация шла
    минуты, а не дни.

ВАЖНО про distribution. Деградация ускорена умеренно (90–150 sim-мин): при
слишком короткой деградации linspace-тренд становится круче обучающего и
`diff_30` уходит за обучающий диапазон (модель детектит раньше/резче). Ручка —
`RealtimeConfig.degradation_min/max`.

ЛАВИНА ТРЕВОГ. Аномалии датчиков (p из конфига, в реальном времени поднята до
видимой частоты) и пусковые всплески — это материал «лавины», которую система
гасит окнами/состоянием. Здесь они ПРОИЗВОДЯТСЯ и помечаются флагами
`anomaly_vibration/temperature/current`; подсчёт «сколько бы сработало наивно
против подтверждённых» — задача слоя метрик (следующий модуль).

ПРОГРЕВ. Каждый насос стартует в устоявшейся Норме, а первый цикл принудительно
нормальный (без отказа): первая возможная деградация наступает заметно позже
прогрева препроцессора — отказ в «слепом окне» первых минут не возникает.

КВИТИРОВАНИЕ (обратная связь от UI), действует по ТЕКУЩЕМУ физическому состоянию:
  • Degradation + квитирование → с вер. 50% перевод в Healthy (AR(1) сам стянет
    значения к норме), иначе деградация продолжается;
  • Critical + квитирование → 100% останов: Off (затухание) → Startup (всплеск,
    глушится состоянием) → Healthy («режим предотвращённых сигналов»).

Схема строки (12 колонок) идентична датасету — `process_tick`/`push_history`
работают без правок:
  timestamp, pump_id, state, state_name, fault_type,
  vibration, temperature, current, pressure,
  anomaly_vibration, anomaly_temperature, anomaly_current
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import (FAULT_TYPES, FAULT_WEIGHTS, PUMP_SEEDS, PUMPS)

# Имена состояний State Machine (как в data_generator.py)
_STATE_NAMES = {0: "Off", 1: "Startup", 2: "Healthy",
                3: "Degradation", 4: "Critical"}
OFF, STARTUP, HEALTHY, DEGRADATION, CRITICAL = 0, 1, 2, 3, 4

AMBIENT_TEMP = 20.0

# Устоявшиеся значения Healthy-AR(1) (неподвижные точки): для старта в Норме
#   temp: x=0.9x+70*0.1   → 70 ;  vib: x=0.88x+1.8*0.12 → 1.8
#   curr: x=0.9x+50*0.1   → 50 ;  press: x=0.85x+1.5*0.15 → 1.5
_HEALTHY_FIXED = {"temperature": 70.0, "vibration": 1.8,
                  "current": 50.0, "pressure": 1.5}


@dataclass
class RealtimeConfig:
    """Ускоренные тайминги/вероятности для валидации (sim-минуты).

    Ускоряем «скучные» части и частоту отказов; крутизну деградации держим
    умеренной, чтобы diff-признаки не ушли за обучающий диапазон.
    Все длительности — диапазоны [min, max) для rng.integers.
    """
    # длительности состояний (sim-мин)
    off_min: int = 5;          off_max: int = 16          # был 60..180
    healthy_norm_min: int = 130; healthy_norm_max: int = 221  # короче старого: насыщеннее событиями (пик переходов ~4/ч)
    healthy_pre_fault_min: int = 20; healthy_pre_fault_max: int = 41  # перед отказом
    degradation_min: int = 90;  degradation_max: int = 151  # умеренно (distribution!)
    critical_min: int = 10;     critical_max: int = 31      # был 10..60
    # startup как в оригинале (2..3 мин)
    startup_min: int = 2;       startup_max: int = 4

    # вероятность того, что цикл — отказной. Подобрана под горизонт 480 мин /
    # 5 насосов так, чтобы предупреждений было немного (≈3 на парк), а аварий ≤2.
    p_fault: float = 0.22
    # частота аппаратной аномалии датчика за минуту в Healthy (был 0.001).
    # Поднята, чтобы «лавина» была видимой за короткий прогон валидации.
    anomaly_rate: float = 0.0075

    # квитирование предупреждения: вероятность реального устранения
    warning_recovery_prob: float = 0.5
    # короткий останов после квитирования аварии перед рестартом (sim-мин)
    ack_off_min: int = 3;       ack_off_max: int = 8

    # воспроизводимость: общий мастер-сид; на насос — производный поток
    master_seed: int = 2026


@dataclass
class _PumpRuntime:
    """Состояние одного насоса: AR(1)-память + позиция в цикле State Machine."""
    pump_id: str
    rng: np.random.Generator
    # AR(1)-состояние (храним НЕокруглённым — как self.last_* в оригинале)
    temp: float
    vib: float
    curr: float
    press: float
    state: int = HEALTHY
    remaining: int = 0          # минут до конца текущего состояния
    cycle_is_failure: bool = False
    fault: Optional[str] = None
    # для startup: индекс/длительность (нужно для спадающего пускового тока)
    seg_i: int = 0
    seg_len: int = 0
    # для degradation: предвычисленные linspace-тренды
    deg_trends: Dict[str, np.ndarray] = field(default_factory=dict)
    # #5: насос остановлен по аварии и НЕ перезапускается до квитирования
    halted: bool = False
    # учёт исходов квитирования (для будущих метрик)
    ack_recovered: int = 0
    ack_restarted: int = 0
    # первый рабочий цикл после старта симуляции — гарантированно без отказа
    first_cycle: bool = False


class LiveMultiPumpGenerator:
    """Пошаговый генератор парка: step() → по строке на каждый насос за тик."""

    def __init__(self,
                 pump_ids: Optional[List[str]] = None,
                 config: Optional[RealtimeConfig] = None,
                 start_date: str = "2026-04-01 00:00:00"):
        self.cfg = config or RealtimeConfig()
        self.pump_ids = list(pump_ids if pump_ids is not None else PUMPS)
        self.clock = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        self.tick = 0
        self._master = np.random.default_rng(self.cfg.master_seed)
        self.pumps: Dict[str, _PumpRuntime] = {}
        for pid in self.pump_ids:
            self.pumps[pid] = self._make_pump(pid)

    # инициализация насоса в устоявшейся Норме (первый цикл — без отказа)

    def _make_pump(self, pid: str) -> _PumpRuntime:
        # производный сид: из PUMP_SEEDS, если есть, иначе из мастер-потока
        base = PUMP_SEEDS.get(pid)
        seed = int(base) if base is not None else int(self._master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed ^ self.cfg.master_seed)
        p = _PumpRuntime(
            pump_id=pid, rng=rng,
            temp=_HEALTHY_FIXED["temperature"] + rng.normal(0, 0.4),
            vib=_HEALTHY_FIXED["vibration"] + rng.normal(0, 0.08),
            curr=_HEALTHY_FIXED["current"] + rng.normal(0, 0.8),
            press=_HEALTHY_FIXED["pressure"] + rng.normal(0, 0.018),
        )
        # старт симуляции — с ПУСКА оборудования (а не из середины работы):
        # Пуск → Норма. Параметры не «появляются изнеоткуда». Первый рабочий
        # цикл гарантированно без отказа (флаг снимается в STARTUP→HEALTHY).
        p.state = STARTUP
        p.seg_len = int(rng.integers(self.cfg.startup_min, self.cfg.startup_max))
        p.seg_i = 0
        p.remaining = p.seg_len
        p.cycle_is_failure = False
        p.fault = None
        p.first_cycle = True
        return p

    def reset(self) -> None:
        self.tick = 0
        for pid in self.pump_ids:
            self.pumps[pid] = self._make_pump(pid)

    # ---- переходы State Machine (тайминги/вероятности — из RealtimeConfig) ----

    def _enter_next_state(self, p: _PumpRuntime) -> None:
        cfg, rng = self.cfg, p.rng
        cur = p.state
        if cur == OFF:
            p.state = STARTUP
            p.seg_len = int(rng.integers(cfg.startup_min, cfg.startup_max))
            p.seg_i = 0
            p.remaining = p.seg_len
        elif cur == STARTUP:
            p.state = HEALTHY
            # Выход на режим: снимаем стартовый переходный хвост AR(1). При
            # ускоренных коротких Healthy он доминировал бы и смещал распределение
            # Нормы вверх (ток/вибрация). Сброс к устоявшимся точкам = насос
            # мгновенно вышел на режим. Плавную релаксацию оставляем ТОЛЬКО для
            # восстановления после квитирования (там значения не сбрасываются).
            p.temp = _HEALTHY_FIXED["temperature"] + rng.normal(0, 0.4)
            p.vib = _HEALTHY_FIXED["vibration"] + rng.normal(0, 0.08)
            p.curr = _HEALTHY_FIXED["current"] + rng.normal(0, 0.8)
            p.press = _HEALTHY_FIXED["pressure"] + rng.normal(0, 0.018)
            if p.first_cycle:
                p.first_cycle = False
                p.cycle_is_failure = False        # первый рабочий цикл — без отказа
            else:
                p.cycle_is_failure = bool(rng.random() < cfg.p_fault)
            if p.cycle_is_failure:
                p.fault = str(rng.choice(np.array(FAULT_TYPES, dtype=object),
                                         p=FAULT_WEIGHTS))
                p.remaining = int(rng.integers(cfg.healthy_pre_fault_min,
                                               cfg.healthy_pre_fault_max))
            else:
                p.fault = None
                p.remaining = int(rng.integers(cfg.healthy_norm_min,
                                               cfg.healthy_norm_max))
        elif cur == HEALTHY:
            if p.cycle_is_failure and p.fault is not None:
                self._begin_degradation(p)
            else:
                p.state = OFF
                p.remaining = int(rng.integers(cfg.off_min, cfg.off_max))
        elif cur == DEGRADATION:
            p.state = CRITICAL
            p.remaining = int(rng.integers(cfg.critical_min, cfg.critical_max))
        elif cur == CRITICAL:
            p.state = OFF
            p.cycle_is_failure = False
            p.fault = None
            p.remaining = int(rng.integers(cfg.off_min, cfg.off_max))

    def _begin_degradation(self, p: _PumpRuntime) -> None:
        """Готовит linspace-тренды деградации (один-в-один с data_generator)."""
        cfg, rng = self.cfg, p.rng
        dur = int(rng.integers(cfg.degradation_min, cfg.degradation_max))
        p.state = DEGRADATION
        p.seg_i = 0
        p.seg_len = dur
        p.remaining = dur
        ln = np.linspace
        if p.fault == "overheat":
            p.deg_trends = {
                "temp": ln(p.temp, 91.0, dur), "vib": ln(2.0, 5.5, dur),
                "curr": ln(50.0, 66.0, dur), "press": ln(1.5, 1.4, dur)}
        elif p.fault == "cavitation":
            p.deg_trends = {"vib": ln(2.5, 7.5, dur), "press": ln(1.5, 0.9, dur)}
        else:  # electrical
            p.deg_trends = {"curr": ln(50.0, 72.0, dur)}

    # ---- эмиссия одной минуты по текущему состоянию (формулы из оригинала) ----

    def _emit(self, p: _PumpRuntime, ts: datetime) -> dict:
        rng = p.rng
        ft = "none"
        a_vib = a_temp = a_curr = 0
        # выбросы-аномалии: ТОЛЬКО в выводе строки, AR(1)-память не трогаем
        vib_out = temp_out = curr_out = None

        if p.state == HEALTHY:
            p.temp = p.temp * 0.9 + 70.0 * 0.1 + rng.normal(0, 0.4)
            p.vib = p.vib * 0.88 + 1.8 * 0.12 + rng.normal(0, 0.08)
            p.curr = p.curr * 0.9 + 50.0 * 0.1 + rng.normal(0, 0.8)
            p.press = p.press * 0.85 + 1.5 * 0.15 + rng.normal(0, 0.018)
            # Аппаратный сбой датчика: разовый выброс ТОЛЬКО в выводе строки,
            # AR(1)-память не трогаем — иначе при поднятой частоте аномалий
            # базовая линия Healthy «уплывает» (ток/вибрация выше номинала).
            # Это и физичнее: глитч датчика — мгновенное ложное показание.
            if rng.random() <= self.cfg.anomaly_rate:
                sensor = rng.choice(np.array(["vib", "temp", "curr"], dtype=object))
                if sensor == "vib":
                    vib_out = rng.normal(9.5, 0.5); a_vib = 1
                elif sensor == "temp":
                    temp_out = rng.normal(97.0, 1.5); a_temp = 1
                else:
                    curr_out = rng.normal(90.0, 3.0); a_curr = 1

        elif p.state == OFF:
            p.temp = AMBIENT_TEMP + (p.temp - AMBIENT_TEMP) * 0.95
            p.vib = p.vib * 0.5 + rng.normal(0, 0.04)
            p.curr = p.curr * 0.5 + rng.normal(0, 0.08)
            p.press = p.press * 0.5 + rng.normal(0, 0.008)

        elif p.state == STARTUP:
            i, dur = p.seg_i, max(1, p.seg_len)
            p.temp = p.temp + rng.normal(4.0, 0.8)
            p.vib = rng.normal(4.5, 0.5)
            # пусковой ток как в датасете (инраш ~150 А, спадает). Глушится
            # состоянием STARTUP + сбросом окна + warming-фильтром, поэтому
            # ложной аварии не даёт, но на графике совпадает с обучающими данными.
            p.curr = rng.normal(150 * (1 - 0.6 * i / dur), 7)
            p.press = rng.normal(1.8, 0.12)
            p.seg_i += 1

        elif p.state == DEGRADATION:
            i = min(p.seg_i, p.seg_len - 1)
            ft = p.fault or "none"
            t = p.deg_trends
            if p.fault == "overheat":
                p.temp = t["temp"][i] + rng.normal(0, 0.5)
                p.vib = t["vib"][i] + rng.normal(0, 0.18)
                p.curr = t["curr"][i] + rng.normal(0, 2.0)
                p.press = p.press * 0.85 + t["press"][i] * 0.15 + rng.normal(0, 0.03)
            elif p.fault == "cavitation":
                p.temp = p.temp * 0.9 + 70.0 * 0.1 + rng.normal(0, 0.5)
                p.vib = t["vib"][i] + rng.normal(0, 0.3)
                p.curr = p.curr * 0.9 + 50.0 * 0.1 + rng.normal(0, 1.0)
                p.press = t["press"][i] + rng.normal(0, 0.08)
            else:  # electrical
                p.temp = p.temp * 0.9 + 72.0 * 0.1 + rng.normal(0, 0.6)
                p.vib = p.vib * 0.88 + 2.0 * 0.12 + rng.normal(0, 0.15)
                spike = rng.normal(0, 6.0) if rng.random() < 0.1 else 0.0
                p.curr = t["curr"][i] + rng.normal(0, 3.0) + spike
                p.press = p.press * 0.85 + 1.45 * 0.15 + rng.normal(0, 0.04)
            p.seg_i += 1

        elif p.state == CRITICAL:
            ft = p.fault or "none"
            if p.fault == "overheat":
                p.temp = p.temp * 0.8 + 96.0 * 0.2 + rng.normal(0, 0.8)
                p.vib = p.vib * 0.8 + 8.5 * 0.2 + rng.normal(0, 0.4)
                p.curr = p.curr * 0.8 + 78.0 * 0.2 + rng.normal(0, 2.0)
                p.press = p.press * 0.8 + 1.3 * 0.2 + rng.normal(0, 0.05)
            elif p.fault == "cavitation":
                p.temp = p.temp * 0.9 + 74.0 * 0.1 + rng.normal(0, 0.8)
                p.vib = p.vib * 0.8 + 9.0 * 0.2 + rng.normal(0, 0.5)
                p.curr = p.curr * 0.9 + 52.0 * 0.1 + rng.normal(0, 1.0)
                p.press = p.press * 0.8 + 0.7 * 0.2 + rng.normal(0, 0.06)
            else:  # electrical
                p.temp = p.temp * 0.9 + 75.0 * 0.1 + rng.normal(0, 0.8)
                p.vib = p.vib * 0.85 + 2.2 * 0.15 + rng.normal(0, 0.2)
                p.curr = p.curr * 0.8 + 95.0 * 0.2 + rng.normal(0, 3.5)
                p.press = p.press * 0.85 + 1.4 * 0.15 + rng.normal(0, 0.05)

        vib_value = p.vib if vib_out is None else vib_out
        temp_value = p.temp if temp_out is None else temp_out
        curr_value = p.curr if curr_out is None else curr_out
        return {
            "timestamp": ts,
            "pump_id": p.pump_id,
            "state": p.state,
            "state_name": _STATE_NAMES[p.state],
            "fault_type": ft,
            "vibration": round(max(0.0, vib_value), 3),
            "temperature": round(max(AMBIENT_TEMP, temp_value), 2),
            "current": round(max(0.0, curr_value), 2),
            "pressure": round(max(0.0, p.press), 3),
            "anomaly_vibration": a_vib,
            "anomaly_temperature": a_temp,
            "anomaly_current": a_curr,
        }

    # ---- публичное API ----

    def step(self) -> List[dict]:
        """Один тик: по строке на каждый насос (единый timestamp), затем +1 мин."""
        ts = self.clock
        rows: List[dict] = []
        for pid in self.pump_ids:
            p = self.pumps[pid]
            if not p.halted:                       # #5: остановленный по аварии
                if p.remaining <= 0:               # не переходит дальше — держим Off
                    self._enter_next_state(p)
            rows.append(self._emit(p, ts))
            if not p.halted:
                p.remaining -= 1
        self.clock += timedelta(minutes=1)
        self.tick += 1
        return rows

    def trip(self, pump_id: str) -> None:
        """Защитный останов по ПОДТВЕРЖДЕНИЮ аварии моделью. Вызывается приложением
        при срабатывании FSM stage=2 — то есть ПОСЛЕ того, как модель увидела
        аварию (а не по внутренней истине генератора). Насос уходит в Off и
        держится там до квитирования, не перезапускаясь автоматически (#5)."""
        p = self.pumps.get(pump_id)
        if p is None or p.halted:
            return
        p.state = OFF
        p.halted = True
        p.fault = None
        p.deg_trends = {}
        p.remaining = 0

    def acknowledge(self, pump_id: str) -> Optional[str]:
        """Квитирование по текущему физическому состоянию насоса.

        Возвращает 'recovered' / 'continues' / 'restarted' / None — для метрик.
        """
        p = self.pumps.get(pump_id)
        if p is None:
            return None
        rng = p.rng
        if p.halted or p.state == CRITICAL:
            # квитирование аварии: снимаем останов → перезапуск
            # (Off→Startup→Healthy естественным ходом FSM)
            p.halted = False
            p.state = OFF
            p.cycle_is_failure = False
            p.fault = None
            p.deg_trends = {}
            p.first_cycle = True       # рестарт = чистый пуск: первый цикл гарантированно
                                       # без отказа (как старт симуляции и как в датасете)
            p.remaining = int(rng.integers(self.cfg.ack_off_min, self.cfg.ack_off_max))
            p.ack_restarted += 1
            return "restarted"
        if p.state == DEGRADATION:
            if rng.random() < self.cfg.warning_recovery_prob:
                # реальное устранение: возврат в Норму, AR(1) сам стянет значения
                p.state = HEALTHY
                p.cycle_is_failure = False
                p.fault = None
                p.deg_trends = {}
                p.remaining = int(rng.integers(self.cfg.healthy_norm_min,
                                               self.cfg.healthy_norm_max))
                p.ack_recovered += 1
                return "recovered"
            return "continues"            # ложное квитирование — деградация идёт
        return None


# Самотест верности распределению (headless)
if __name__ == "__main__":
    import pandas as pd

    cfg = RealtimeConfig()
    gen = LiveMultiPumpGenerator(config=cfg)
    print(f"Парк: {gen.pump_ids}")
    print(f"Тик: {len(gen.pump_ids)} строк; деградация {cfg.degradation_min}–"
          f"{cfg.degradation_max - 1} мин; p_fault={cfg.p_fault}; "
          f"anomaly={cfg.anomaly_rate}")

    N_TICKS = 20000
    rows = []
    for _ in range(N_TICKS):
        rows.extend(gen.step())
    df = pd.DataFrame(rows)
    print(f"\nСтрок сгенерировано: {len(df):,}  ({N_TICKS} тиков × {len(gen.pump_ids)})")

    print("\nРаспределение состояний:")
    for name, cnt in df["state_name"].value_counts().items():
        print(f"  {name:<12}: {cnt:>7,} ({100*cnt/len(df):5.1f}%)")

    print("\nСредние Healthy (ожидаем ~ temp70 / vib1.8 / curr50 / press1.5):")
    h = df[df["state"] == HEALTHY]
    print(h[["vibration", "temperature", "current", "pressure"]].mean().round(2).to_dict())

    crit = df[df["state"] == CRITICAL]
    if not crit.empty:
        print("\nСредние Critical по типам (сверка с data_generator):")
        print(crit.groupby("fault_type")[
            ["vibration", "temperature", "current", "pressure"]].mean().round(1))

    print("\nАномалий (лавина-материал):",
          int(df[["anomaly_vibration", "anomaly_temperature",
                  "anomaly_current"]].sum().sum()))

    # проверка квитирования: ставим насос в Degradation/Critical и квитируем
    gp = gen.pumps[gen.pump_ids[0]]
    gp.state = DEGRADATION; gp.fault = "overheat"; gp.remaining = 50
    outcomes = {gen.acknowledge(gen.pump_ids[0]) for _ in range(1)}
    print("\nКвитирование Degradation →", outcomes, "| state теперь:",
          _STATE_NAMES[gp.state])
    gp.state = CRITICAL; gp.fault = "overheat"; gp.remaining = 20
    print("Квитирование Critical →", gen.acknowledge(gen.pump_ids[0]),
          "| state теперь:", _STATE_NAMES[gp.state])