"""
Потоковый stateful-препроцессор реального времени (Online Preprocessor)
======================================================================
src/runtime/online_preprocessor.py

Назначение: принимать сырые строки телеметрии по одной и вычислять
тот же самый вектор признаков FEATURE_COLS, что и offline-модуль
data_preprocessor.py, без пересчёта всего датасета.

КОНТРАКТ ПАРИТЕТА С OFFLINE-ПАЙПЛАЙНОМ (не менять без regression-теста):
  1. shift(1): признаки для момента T считаются ТОЛЬКО по строкам
     [T-w .. T-1]. Текущая сырая строка в окно не входит.
  2. min_periods = w: пока в буфере меньше w предыдущих строк,
     признак не считается (offline там NaN -> строка отброшена).
     Здесь push() возвращает None до полного прогрева (60 строк).
  3. diff_30 = x[T-1] - x[T-31] (ровно 30 шагов, как в offline:
     shift(1) - shift(31)).
  4. Порядок и имена колонок берутся из data_preprocessor.FEATURE_COLS,
     что исключает дрейф контракта признаков.

Прогрев: 60 предыдущих строк (максимальное окно). Это и есть причина,
по которой демо-сценарий (demo_stream.py) включает 60-минутный
warmup-префикс до содержательной части.
"""

from collections import deque
from typing import Dict, List, Optional, Sequence, cast

import numpy as np
import pandas as pd

PARAMS: Sequence[str] = ("vibration", "temperature", "current", "pressure")
WINDOWS: Sequence[int] = (15, 30, 60)
WARMUP_ROWS: int = 60  # max(WINDOWS); также покрывает diff_30 (нужна 31 строка)


class OnlinePreprocessor:
    """Кольцевой буфер истории + расчёт оконных статистик «на лету».

    Один экземпляр обслуживает весь парк: история ведётся отдельно
    по каждому pump_id.
    """

    def __init__(self, feature_cols: Sequence[str]):
        # FEATURE_COLS импортируется вызывающей стороной из
        # data_preprocessor - единый источник истины для train и inference.
        self.feature_cols: List[str] = list(feature_cols)
        # buffer хранит ТОЛЬКО предыдущие строки (история до момента T)
        self._buffers: Dict[str, deque] = {}

    def reset(self, pump_id: Optional[str] = None) -> None:
        if pump_id is None:
            self._buffers.clear()
        else:
            self._buffers.pop(pump_id, None)

    def warmup_progress(self, pump_id: str) -> float:
        buf = self._buffers.get(pump_id)
        return min(1.0, (len(buf) if buf else 0) / WARMUP_ROWS)

    def push(self, pump_id: str, raw_row: Dict[str, float]) -> Optional[pd.DataFrame]:
        """Принять одну сырую строку; вернуть строку признаков или None.

        Семантика shift(1): признаки считаются ПО СОСТОЯНИЮ БУФЕРА ДО
        добавления текущей строки, затем строка кладётся в историю.
        """

        buf = self._buffers.setdefault(pump_id, deque(maxlen=WARMUP_ROWS + 1))

        features: Optional[pd.DataFrame] = None
        if len(buf) >= WARMUP_ROWS:
            features = self._compute(buf)

        buf.append({p: float(raw_row[p]) for p in PARAMS})
        return features

    def _compute(self, buf: deque) -> pd.DataFrame:
        hist = {p: np.array([r[p] for r in buf], dtype=float) for p in PARAMS}
        out: Dict[str, float] = {}
        for p in PARAMS:
            series = hist[p]
            for w in WINDOWS:
                window = series[-w:]
                out[f"{p}_mean_{w}"] = float(window.mean())
                # ddof=1 - как pandas .rolling().std() по умолчанию
                out[f"{p}_std_{w}"] = float(window.std(ddof=1))
                out[f"{p}_max_{w}"] = float(window.max())
            # diff_30: x[T-1] - x[T-31]
            out[f"{p}_diff_30"] = float(series[-1] - series[-31])
        row = pd.DataFrame([out])
        # Жёсткая фиксация порядка колонок по контракту FEATURE_COLS.
        # cast: pandas-stubs выводят DataFrame[list] как Series - здесь это DataFrame.
        return cast(pd.DataFrame, row[self.feature_cols])


# Регрессионная проверка паритета online == offline.
# Запускать один раз при изменении любого из препроцессоров:
#   python experiments/validation/online_parity_test.py
def verify_parity(
    raw_df: pd.DataFrame,
    offline_features: pd.DataFrame,
    feature_cols: Sequence[str],
    pump_col: str = "pump_id",
    atol: float = 1e-8,
    max_rows_per_pump: int = 500,
) -> bool:
    """Сверяет потоковый расчёт с offline-матрицей признаков.

    raw_df            - сырой датасет (industrial_pumps_dataset.csv)
    offline_features  - результат offline data_preprocessor для тех же строк,
                        с сохранённым исходным индексом raw_df.
    Возвращает True при совпадении; иначе бросает AssertionError
    с первой расходящейся колонкой.
    """

    proc = OnlinePreprocessor(feature_cols)
    checked = 0
    for pump_id, grp in raw_df.groupby(pump_col, sort=False):
        proc.reset(pump_id)
        n = 0
        for idx, raw in grp.iterrows():
            feats = proc.push(pump_id, cast(Dict[str, float], raw.to_dict()))
            if feats is None or idx not in offline_features.index:
                continue
            ref = offline_features.loc[idx, feature_cols].astype(float).values
            got = feats.iloc[0].values
            diff = np.abs(ref - got)
            if not np.all(diff <= atol):
                bad = feature_cols[int(np.argmax(diff))]
                raise AssertionError(
                    f"Паритет нарушен: pump={pump_id}, index={idx}, "
                    f"колонка={bad}, offline={ref[np.argmax(diff)]:.10f}, "
                    f"online={got[np.argmax(diff)]:.10f}"
                )
            checked += 1
            n += 1
            if n >= max_rows_per_pump:
                break
    if checked == 0:
        raise AssertionError("Паритет не проверен: нет пересечения индексов.")
    print(f"[OK] Паритет online/offline подтверждён на {checked} строках.")
    return True
