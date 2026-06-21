"""
realtime_player.py — драйвер тика для режима реального времени.
================================================================
Тонкая обёртка над LiveMultiPumpGenerator, повторяющая интерфейс
ScenarioPlayer (next_rows / finished / progress / pos / __len__ / skip_warmup),
поэтому встаёт НА МЕСТО плеера в app.py без переписывания UI: весь существующий
контур (advance_stream, прогресс-бар, ▶/⏸/⏭, графики, тосты) работает как есть.

Отличия от ScenarioPlayer:
  • данные не вырезаются из CSV, а генерируются на лету по всему парку;
  • за один тик — по строке на КАЖДЫЙ насос (общий timestamp), поэтому
    next_rows(n) отдаёт n тиков × N насосов; pos считает sim-минуты (тики);
  • acknowledge(pump_id) пробрасывает квитирование в генератор (обратная связь
    50% восстановление / 100% останов-рестарт).

ПРОГРЕВ. Строгий OnlinePreprocessor выдаёт признаки только после WARMUP_ROWS
(60) строк истории на насос. Поэтому skip_warmup() прогоняет WARMUP_ROWS тиков
форсированно-нормальной Нормы и заполняет буферы — значения видны на графиках
(populated-старт, а не «телепорт в пустоту»), и модель готова с первого видимого
тика. Когда подключим прогрессивный realtime-препроцессор (следующий модуль),
плеер создаётся с warmup_rows=0 → честный «холодный старт» с 0-й минуты.
"""

from __future__ import annotations

import os
import sys
from typing import Iterator, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.realtime_validation.live_generator import (  # noqa: E402
    LiveMultiPumpGenerator, RealtimeConfig)

try:
    from src.runtime.online_preprocessor import WARMUP_ROWS  # noqa: E402
except Exception:
    WARMUP_ROWS = 60  # фолбэк = max(WINDOWS)


class RealtimePlayer:
    """Drop-in вместо ScenarioPlayer поверх живого мультинасосного генератора."""

    def __init__(self,
                 generator: Optional[LiveMultiPumpGenerator] = None,
                 horizon_minutes: int = 1440,
                 warmup_rows: int = WARMUP_ROWS,
                 config: Optional[RealtimeConfig] = None):
        self.gen = generator or LiveMultiPumpGenerator(config=config)
        self.horizon = int(horizon_minutes)
        self.warmup_rows = int(warmup_rows)
        self.pos = 0  # пройдено sim-минут (тиков) ВИДИМОГО потока

    # ---- интерфейс, совпадающий со ScenarioPlayer ----

    def __len__(self) -> int:
        return self.horizon

    @property
    def finished(self) -> bool:
        return self.pos >= self.horizon

    @property
    def progress(self) -> float:
        return min(1.0, self.pos / self.horizon) if self.horizon else 1.0

    def skip_warmup(self) -> Iterator[dict]:
        """Заполнить буферы препроцессора: warmup_rows тиков Нормы.

        Строки отдаются вызывающему (app.py пушит их в историю и process_tick),
        но в pos НЕ засчитываются — это подготовка, не видимый прогон.
        При warmup_rows=0 (прогрессивный препроцессор) не делает ничего.
        """
        for _ in range(self.warmup_rows):
            for row in self.gen.step():
                yield row

    def next_rows(self, n: int) -> Iterator[dict]:
        """n sim-минут потока: на каждый тик — по строке на насос."""
        for _ in range(int(n)):
            if self.finished:
                return
            for row in self.gen.step():
                yield row
            self.pos += 1

    # ---- расширение: обратная связь квитирования (UI → генератор) ----

    def acknowledge(self, pump_id: str) -> Optional[str]:
        """Квитирование из UI → генератор. 'recovered'/'continues'/'restarted'/None."""
        return self.gen.acknowledge(pump_id)

    def trip(self, pump_id: str) -> None:
        """Защитный останов из UI → генератор: насос уходит в Off и держится там
        до квитирования (вызывается приложением при подтверждении аварии моделью)."""
        self.gen.trip(pump_id)


# Самотест: интерфейс-совместимость и поток
if __name__ == "__main__":
    player = RealtimePlayer(horizon_minutes=120, warmup_rows=60)
    n_pumps = len(player.gen.pump_ids)
    print(f"Парк: {player.gen.pump_ids}  | горизонт {len(player)} мин")

    warm = list(player.skip_warmup())
    print(f"Прогрев: {len(warm)} строк ({len(warm) // n_pumps} тиков × {n_pumps}); "
          f"pos после прогрева = {player.pos} (ожидаем 0)")
    assert hasattr(warm[0], "get"), "строки должны поддерживать .get() (как ждёт push_history)"

    step = list(player.next_rows(1))
    one_ts = len({str(r["timestamp"]) for r in step}) == 1
    print(f"⏭ один тик: {len(step)} строк (ожидаем {n_pumps}); pos={player.pos}; "
          f"единый timestamp на тик: {one_ts}")

    total = len(step)
    while not player.finished:
        total += len(list(player.next_rows(20)))
    print(f"Прогон до конца: pos={player.pos}/{len(player)}, finished={player.finished}; "
          f"видимых строк ≈ {total} (ожидаем ~{player.horizon * n_pumps})")

    pid = player.gen.pump_ids[0]
    gp = player.gen.pumps[pid]
    gp.state = 4; gp.fault = "overheat"; gp.remaining = 20
    print(f"acknowledge({pid}) в Critical →", player.acknowledge(pid))