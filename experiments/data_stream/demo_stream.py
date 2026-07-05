"""
Детерминированный демо-сценарий (Data Stream)
=============================================
experiments/data_stream/demo_stream.py

Демо-сценарий для валидации платформы (раздел 3.9).

Подход: сценарий НЕ генерируется заново, а вырезается из уже
существующего размеченного датасета industrial_pumps_dataset.csv.
Это даёт три свойства, важных для защиты:
  1. Воспроизводимость - один и тот же сегмент на каждом запуске;
  2. Честность - данные те же, на которых валидированы модели
     (тестовый насос MNHV_005 по умолчанию -> unseen для моделей);
  3. Гарантированный сюжет - Норма -> Предупреждение -> Авария
     выбранного физического типа отказа.

Структура сценария:
  [warmup 60 мин Healthy] -> [healthy_minutes Healthy]
      -> [полная фаза Degradation] -> [critical_minutes Critical]

Warmup-префикс нужен потоковому препроцессору для заполнения окон
(см. online_preprocessor.WARMUP_ROWS) - на дашборде он проигрывается
ускоренно либо скрыто.
"""

from dataclasses import dataclass, field
from typing import List, Optional, cast

import pandas as pd

# Коды конечного автомата генератора (см. data_generator.py)
STATE_NAMES = {"Off": 0, "Startup": 1, "Healthy": 2, "Degradation": 3, "Critical": 4}
HEALTHY, DEGRADATION, CRITICAL = 2, 3, 4

WARMUP_MINUTES = 60


def _state_series(df: pd.DataFrame) -> pd.Series:
    """Столбец состояния FSM как int-Series: из числового 'state', иначе
    маппингом текстового 'state_name' через STATE_NAMES."""

    if "state" in df.columns:
        return df["state"].astype(int)
    return df["state_name"].map(STATE_NAMES).astype(int)


def extract_demo_scenario(
    dataset_path: str,
    fault_type: str,                      # 'overheat' | 'cavitation' | 'electrical'
    pump_id: Optional[str] = "MNHV_005",  # None -> искать по всему парку
    healthy_minutes: int = 25,
    critical_minutes: int = 12,
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """Найти и вырезать первый подходящий эпизод отказа заданного типа.

    Критерий: непрерывный блок Degradation(fault_type), перед которым
    стоит не менее (WARMUP + healthy_minutes) строк Healthy, и за
    которым следует Critical того же типа.
    """

    df = pd.read_csv(dataset_path)
    pumps = [pump_id] if pump_id else sorted(df["pump_id"].unique())

    for pid in pumps:
        grp = cast(pd.DataFrame, df[df["pump_id"] == pid]).reset_index(drop=True)
        st = _state_series(grp).to_numpy()
        ft = grp["fault_type"].to_numpy()

        i = 0
        need_prefix = WARMUP_MINUTES + healthy_minutes
        while i < len(grp):
            if st[i] == DEGRADATION and ft[i] == fault_type:
                # начало деградации нужного типа
                start_deg = i
                # проверка Healthy-префикса
                if start_deg < need_prefix or not (
                    (st[start_deg - need_prefix:start_deg] == HEALTHY).all()
                ):
                    # промотать до конца этого блока деградации
                    while i < len(grp) and st[i] == DEGRADATION:
                        i += 1
                    continue
                # конец деградации
                j = start_deg
                while j < len(grp) and st[j] == DEGRADATION:
                    j += 1
                # за деградацией должна идти авария того же типа
                if j >= len(grp) or st[j] != CRITICAL or ft[j] != fault_type:
                    i = j
                    continue
                k = j
                while k < len(grp) and st[k] == CRITICAL and (k - j) < critical_minutes:
                    k += 1
                scenario = grp.iloc[start_deg - need_prefix:k].copy()
                scenario.reset_index(drop=True, inplace=True)
                scenario["demo_phase"] = (
                    ["warmup"] * WARMUP_MINUTES
                    + ["healthy"] * healthy_minutes
                    + ["degradation"] * (j - start_deg)
                    + ["critical"] * (k - j)
                )
                if out_path:
                    scenario.to_csv(out_path, index=False)
                return scenario
            i += 1

    raise ValueError(
        f"Эпизод '{fault_type}' с достаточным Healthy-префиксом не найден "
        f"(pump_id={pump_id}). Попробуй pump_id=None или другой тип."
    )


@dataclass
class ScenarioPlayer:
    """Плеер потока: выдаёт строки по одной, хранит позицию.

    Состояние плеера живёт в st.session_state - интерфейс при каждом
    rerun продолжает с того же места.
    """

    scenario: pd.DataFrame
    pos: int = 0
    finished: bool = field(default=False)

    def __len__(self) -> int:
        """Длина сценария в строках (sim-минутах)."""

        return len(self.scenario)

    @property
    def progress(self) -> float:
        """Доля пройденного сценария [0..1] для прогресс-бара."""

        return self.pos / max(1, len(self.scenario))

    def next_rows(self, n: int = 1) -> List[pd.Series]:
        """Выдать до n следующих строк сценария (по одной на sim-минуту).

        По достижении конца выставляет finished=True и возвращает уже накопленное."""

        rows: List[pd.Series] = []
        for _ in range(n):
            if self.pos >= len(self.scenario):
                self.finished = True
                break
            rows.append(self.scenario.iloc[self.pos])
            self.pos += 1
        return rows

    def skip_warmup(self) -> List[pd.Series]:
        """Прогнать warmup-префикс одним пакетом (для быстрого старта демо)."""
        
        rows: List[pd.Series] = []
        if "demo_phase" not in self.scenario.columns:
            return rows
        while (
            self.pos < len(self.scenario)
            and self.scenario.iloc[self.pos]["demo_phase"] == "warmup"
        ):
            rows.append(self.scenario.iloc[self.pos])
            self.pos += 1
        return rows
