"""
Регрессионный тест паритета препроцессоров: online == offline
=============================================================


Гарантирует, что вектор признаков FEATURE_COLS, посчитанный ПОТОКОВО (по одной
строке, как в реальном времени), бит-в-бит совпадает с ОФЛАЙН-расчётом, на
котором обучены модели. Любое расхождение - это рассинхрон train/inference:
модель в проде увидела бы не те числа, что в обучении.

Что проверяется:
  1. verify_parity: строгий OnlinePreprocessor == offline DataPreprocessor
     (тот же shift(1), min_periods=w, diff_30, std ddof=1) на каждой строке
     после прогрева - иначе AssertionError с первой расходящейся колонкой.
  2. Прогрессивный RealtimeProgressivePreprocessor (живой режим, признаки с 1-й
     строки) == строгий препроцессор на ПОЛНОМ окне (>= WARMUP_ROWS): ускоренный
     прогрев на старте не ломает паритет с обучением.
  3. На частичном окне (во время прогрева) прогрессивный ВЫДАЁТ признаки, а
     строгий молчит - подтверждение разной политики прогрева (это фича, не баг).

Выравнивание индексов (важно): verify_parity сопоставляет строки по индексу
сырого датасета, поэтому offline считается через process(is_training=False) -
этот режим НЕ делает dropna/reset_index и сохраняет исходный индекс.

Скорость: берутся первые HEAD_ROWS строк КАЖДОГО насоса (непрерывная ранняя
история → окна корректны), а не весь датасет на 648k строк.

Запуск:
    pytest experiments/validation/online_parity_test.py -v
  или:  python experiments/validation/online_parity_test.py
Предусловие: сгенерирован сырой датасет (python -m src.data.data_generator).
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

from config.settings import WINDOW_SIZES
from src.data.data_preprocessor import DataPreprocessor
from src.runtime.online_preprocessor import (OnlinePreprocessor, WARMUP_ROWS,
                                             verify_parity)
from experiments.realtime_validation.realtime_preprocessor import (
    RealtimeProgressivePreprocessor)

_RAW = os.path.join(_PROJECT_ROOT, 'data', 'raw', 'industrial_pumps_dataset.csv')
HEAD_ROWS = 800 # на насос: WARMUP_ROWS(60) прогрева + запас проверочных строк
ATOL = 1e-8

_pre = DataPreprocessor(window_sizes=WINDOW_SIZES)
FEATURE_COLS = _pre.FEATURE_COLS

_cache = {}


def _raw_head() -> pd.DataFrame:
    """Первые HEAD_ROWS строк каждого насоса в хронологическом порядке.

    Непрерывная ранняя история → скользящие окна считаются корректно; объём мал
    → тест проходит за секунды. Индекс сбрасывается в чистый RangeIndex, чтобы
    offline и сырой датасет ссылались на одни и те же метки строк."""

    if 'raw' not in _cache:
        if not os.path.isfile(_RAW):
            raise FileNotFoundError(
                f"Не найден сырой датасет: {_RAW}. Сначала запустите "
                f"python -m src.data.data_generator")
        df = pd.read_csv(_RAW)
        df = (df.sort_values(['pump_id', 'timestamp'])
                .groupby('pump_id', sort=False).head(HEAD_ROWS)
                .reset_index(drop=True))
        _cache['raw'] = df
    return _cache['raw']


def _offline_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Offline-матрица признаков с СОХРАНЁННЫМ индексом raw.

    is_training=False не делает dropna/reset_index - индекс совпадает с raw,
    поэтому verify_parity может сопоставить строки по метке."""
    
    return _pre.process(raw.copy(), is_training=False)


# Тест 1: строгий потоковый препроцессор == offline бит-в-бит
def test_online_strict_parity():
    raw = _raw_head()
    offline = _offline_features(raw)
    assert verify_parity(raw, offline, FEATURE_COLS, atol=ATOL) is True


# Тест 2: прогрессивный realtime-препроцессор == строгий на ПОЛНОМ окне
def test_progressive_matches_strict_on_full_window():
    raw = _raw_head()
    strict = OnlinePreprocessor(FEATURE_COLS)
    prog = RealtimeProgressivePreprocessor(FEATURE_COLS)
    checked = 0
    for pid, grp in raw.groupby('pump_id', sort=False):
        strict.reset(pid)
        prog.reset(pid)
        for _, row in grp.iterrows():
            d = {str(k): float(v) for k, v in row.to_dict().items()
                 if k in ('vibration', 'temperature', 'current', 'pressure')}
            fs = strict.push(pid, dict(d))
            fp = prog.push(pid, dict(d))
            if fs is None: # строгий ещё прогревается - не сравниваем
                continue
            assert fp is not None, (
                f"pump={pid}: строгий выдал признаки, прогрессивный - None.")
            diff = np.abs(fs[FEATURE_COLS].to_numpy(dtype=float)
                          - fp[FEATURE_COLS].to_numpy(dtype=float))
            if not np.all(diff <= ATOL):
                bad = FEATURE_COLS[int(np.argmax(diff))]
                raise AssertionError(
                    f"pump={pid}: расхождение на полном окне, колонка={bad}, "
                    f"strict={fs[bad].iloc[0]:.10f}, prog={fp[bad].iloc[0]:.10f}")
            checked += 1
    assert checked > 0, "Нет строк на полном окне - проверка не выполнена."
    print(f"[OK] progressive == strict на полном окне: {checked} строк.")


# Тест 3: на частичном окне (прогрев) прогрессивный выдаёт признаки, строгий - нет
def test_progressive_emits_during_warmup():
    raw = _raw_head()
    pid = raw['pump_id'].iloc[0]
    grp = raw[raw['pump_id'] == pid]
    strict = OnlinePreprocessor(FEATURE_COLS)
    prog = RealtimeProgressivePreprocessor(FEATURE_COLS)
    emitted_during_warmup = 0
    for i, (_, row) in enumerate(grp.iterrows()):
        if i >= WARMUP_ROWS:
            break
        d = {str(k): float(v) for k, v in row.to_dict().items()
             if k in ('vibration', 'temperature', 'current', 'pressure')}
        fs = strict.push(pid, dict(d))
        fp = prog.push(pid, dict(d))
        assert fs is None, f"строгий выдал признаки во время прогрева (строка {i})."
        if fp is not None:
            emitted_during_warmup += 1
    assert emitted_during_warmup > 0, (
        "Прогрессивный препроцессор не выдал признаки в окне прогрева.")
    print(f"[OK] progressive выдал {emitted_during_warmup} строк в прогреве "
          f"(строгий - 0).")


if __name__ == "__main__":
    tests = [test_online_strict_parity,
             test_progressive_matches_strict_on_full_window,
             test_progressive_emits_during_warmup]
    failed = 0
    print("=" * 59)
    print("Паритет препроцессоров: online == offline")
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
