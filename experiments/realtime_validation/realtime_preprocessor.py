"""
realtime_preprocessor.py — прогрессивный потоковый препроцессор (реальное время).
================================================================================
Назначение: в режиме реального времени выдавать вектор признаков FEATURE_COLS
С ПЕРВОЙ ЖЕ строки, а не после 60-минутного прогрева. Это снимает «слепое окно»
на старте демо: модель оценивает состояние сразу, на частичных (расширяющихся)
окнах, а по мере накопления истории окна становятся полноразмерными.

ОТЛИЧИЕ ОТ СТРОГОГО OnlinePreprocessor — ТОЛЬКО политика прогрева. Наследуемся
от него, чтобы переиспользовать __init__/reset/feature_cols и НЕ трогать его
паритет-контракт и regression-тест verify_parity (offline == online на полном
окне остаётся в силе для обучающего пайплайна).

Прогрессивные окна (mean/max/std/diff) считаются так:
  • mean, max — по min(len_буфера, w) последним точкам (расширяющееся → скользящее
    окно). Пока строк меньше 15, окна 15/30/60 численно совпадают — это и есть
    поведение «считаем по тому, что есть»;
  • std — 0 при <2 точках (волатильности ещё нет), иначе по доступным (ddof=1,
    как pandas .rolling().std());
  • diff_30 — 0 при <31 строке (нет 30-шаговой истории → нет тренда), иначе
    точный x[T-1] − x[T-31].

СОХРАНЯЕТСЯ:
  • shift(1): признаки момента T считаются ТОЛЬКО по строкам [T-w .. T-1];
    текущая сырая строка в окно не входит (защита от утечки, как в offline);
  • порядок и имена колонок строго по FEATURE_COLS (контракт признаков);
  • ПАРИТЕТ НА ПОЛНОМ ОКНЕ: начиная с 60-й строки результат бит-в-бит совпадает
    со строгим OnlinePreprocessor (и, следовательно, с offline-обучением).

Компромисс (осознанный, согласован): первые ~15 минут признаки построены на
частичных окнах и слегка вне обучающего распределения. Поскольку Норма стартует
с устоявшихся значений, mean/max почти не отличаются от полных; шумят std/diff,
но diff на старте обнулён. На плитках первые минуты помечаются «прогрев»
(см. rows_seen) — оценка идёт, но честно отмечена как предварительная.
"""

from collections import deque
from typing import Dict, Optional

import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.runtime.online_preprocessor import (OnlinePreprocessor, PARAMS,
                                             WINDOWS, WARMUP_ROWS)


class RealtimeProgressivePreprocessor(OnlinePreprocessor):
    """OnlinePreprocessor с прогрессивным прогревом (признаки с 1-й строки)."""

    def push(self, pump_id: str, raw_row: Dict[str, float]) -> Optional[pd.DataFrame]:
        buf = self._buffers.setdefault(pump_id, deque(maxlen=WARMUP_ROWS + 1))
        # shift(1): считаем ПО буферу ДО добавления текущей строки.
        # Отличие от строгого — порог: с 1-й строки истории, а не с 60-й.
        features = self._compute(buf) if len(buf) >= 1 else None
        buf.append({p: float(raw_row[p]) for p in PARAMS})
        return features

    def _compute(self, buf: deque) -> pd.DataFrame:
        hist = {p: np.array([r[p] for r in buf], dtype=float) for p in PARAMS}
        n = len(buf)
        out: Dict[str, float] = {}
        for p in PARAMS:
            series = hist[p]
            for w in WINDOWS:
                window = series[-w:]                       # min(n, w) точек
                out[f"{p}_mean_{w}"] = float(window.mean())
                out[f"{p}_max_{w}"] = float(window.max())
                # std требует >=2 точек; иначе волатильности ещё нет
                out[f"{p}_std_{w}"] = (float(window.std(ddof=1))
                                       if window.size >= 2 else 0.0)
            # diff_30: точный 30-шаговый градиент только при >=31 строке
            out[f"{p}_diff_30"] = (float(series[-1] - series[-31])
                                   if n >= 31 else 0.0)
        row = pd.DataFrame([out])
        return row[self.feature_cols]   # жёсткий порядок колонок по контракту # type: ignore

    def rows_seen(self, pump_id: str) -> int:
        """Накоплено строк истории по насосу — для бейджа «прогрев» на плитке."""
        buf = self._buffers.get(pump_id)
        return len(buf) if buf else 0


# Самотест: паритет на полном окне + поведение на частичных окнах
if __name__ == "__main__":
    sensors = list(PARAMS)
    fc = []
    for c in sensors:
        for w in WINDOWS:
            fc += [f"{c}_mean_{w}", f"{c}_std_{w}", f"{c}_max_{w}"]
        fc.append(f"{c}_diff_30")
    print(f"FEATURE_COLS: {len(fc)} признаков")

    rng = np.random.default_rng(0)
    mu = np.array([1.8, 70.0, 50.0, 1.5])
    sg = np.array([0.1, 0.4, 0.8, 0.02])
    rows = [{p: float(v) for p, v in zip(sensors, rng.normal(mu, sg))}
            for _ in range(220)]

    strict = OnlinePreprocessor(fc)
    prog = RealtimeProgressivePreprocessor(fc)

    prog_first = strict_first = None
    mism = 0
    for i, r in enumerate(rows):
        fs = strict.push("P", dict(r))
        fp = prog.push("P", dict(r))
        if strict_first is None and fs is not None:
            strict_first = i
        if prog_first is None and fp is not None:
            prog_first = i
        if fs is not None:  # где строгий даёт результат — прогрессивный обязан совпасть
            d = np.abs(fs[fc].values - fp[fc].values)   # type: ignore
            if not np.all(d <= 1e-9):
                mism += 1
    print(f"первый признак: progressive с тика {prog_first}, strict с тика {strict_first}")
    print(f"расхождений на ПОЛНОМ окне (после прогрева): {mism}  (ожидаем 0)")

    prog2 = RealtimeProgressivePreprocessor(fc)
    print("\nЧастичные окна на старте (vibration):")
    for i, r in enumerate(rows[:4] + [rows[16]] + [rows[32]]):
        f = prog2.push("P", dict(r))
        if f is None:
            print(f"  тик {i}: None (буфер пуст)")
            continue
        print(f"  строк в окне≈{prog2.rows_seen('P'):>2}: "
              f"mean_15={f['vibration_mean_15'].iloc[0]:.3f}  "
              f"mean_60={f['vibration_mean_60'].iloc[0]:.3f}  "
              f"std_15={f['vibration_std_15'].iloc[0]:.3f}  "
              f"diff_30={f['vibration_diff_30'].iloc[0]:.3f}")