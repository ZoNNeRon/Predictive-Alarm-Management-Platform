"""
Адаптер реального датасета NASA C-MAPSS (Turbofan Engine Degradation)
=====================================================================
src/data/cmapss_dataset.py

Мост между реальными run-to-failure данными NASA PCoE и аналитическим ядром
платформы. Отвечает за:

  1. ЗАГРУЗКУ датасета из сети (официальное S3-зеркало NASA) в
     data/cmapss_dataset/ - если все файлы на месте, сеть не трогается. 
     Архив содержит вложенный CMAPSSData.zip - распаковка рекурсивная. 
     Датасет в git не публикуется (.gitignore).
  2. ПАРСИНГ 26-колоночных txt (юнит, цикл, 3 режимные настройки, 21 сенсор)
     в схему платформы: unit -> pump_id ('FD001_E001'), cycle -> timestamp.
  3. РАЗМЕТКУ: RUL (остаток циклов до отказа) по каждому двигателю
     (train - от последнего цикла траектории; test - от известного финального
     RUL из RUL_FD00X.txt) и piecewise-маппинг RUL -> target 0/1/2
     (Норма / Предупреждение / Авария) - той же трёхклассовой постановки,
     что и у насосного пайплайна.
  4. РЕЖИМНУЮ НОРМАЛИЗАЦИЮ для FD002/FD004 (6 полётных режимов): z-score
     сенсоров внутри режима, статистики фитуются ТОЛЬКО на train (нет утечки).
     Режим определяется детерминированно - по ближайшему якорю op1.
  5. ПРЕПРОЦЕССИНГ: CmapssPreprocessor наследует DataPreprocessor - вся
     механика rolling-признаков (shift(1) против утечки, groupby по агрегату,
     min_periods) переиспользуется без изменений. Переопределены только
     список сенсоров, окна (в ЦИКЛАХ, не минутах) и лаг diff-признака.

Точка входа prepare_subset(subset) возвращает готовые для ML train/test
DataFrame с target и FEATURE_COLS - дальше работают немодифицированные
severity-модели, XAI и визуализация.
"""

import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from typing import Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings.settings_cmapss import (
    CMAPSS_URLS, CMAPSS_DATA_SUBDIR, CMAPSS_SUBSETS, CMAPSS_SUBSET_INFO,
    CMAPSS_REQUIRED_FILES, CMAPSS_RAW_COLUMNS, ENGINE_ID_FMT,
    CMAPSS_SENSORS, RUL_WARNING, RUL_CRITICAL, CMAPSS_STAGE_LABELS,
    CMAPSS_WINDOW_SIZES, CMAPSS_DIFF_LAG, CMAPSS_MULTI_REGIME_SUBSETS,
    CMAPSS_REGIME_OP1_ANCHORS)
from src.data.data_preprocessor import DataPreprocessor

DATA_DIR = os.path.join(_PROJECT_ROOT, *CMAPSS_DATA_SUBDIR)


# Загрузка датасета из сети

def download_cmapss(data_dir: str = DATA_DIR, force: bool = False) -> str:
    """
    Гарантирует наличие всех файлов C-MAPSS в data_dir; возвращает data_dir.

    Идемпотентно: если полный комплект txt уже лежит - сеть не трогается.
    Иначе архив скачивается с первого доступного URL (CMAPSS_URLS),
    вложенные zip распаковываются рекурсивно, нужные txt раскладываются
    в data_dir, временные файлы удаляются.
    """

    os.makedirs(data_dir, exist_ok=True)
    missing = [f for f in CMAPSS_REQUIRED_FILES
               if not os.path.isfile(os.path.join(data_dir, f))]
    if not missing and not force:
        return data_dir

    print(f"C-MAPSS: отсутствуют файлы ({len(missing)} шт.) - загрузка из сети...")
    last_err: Optional[Exception] = None
    for url in CMAPSS_URLS:
        try:
            with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
                archive = os.path.join(tmp, 'cmapss.zip')
                print(f"  Скачивание: {url}")
                urllib.request.urlretrieve(url, archive)
                size_mb = os.path.getsize(archive) / 1048576
                print(f"  Получено {size_mb:.1f} МБ, распаковка...")
                _extract_recursive(archive, tmp)
                found = _collect_txt(tmp, data_dir)
                if found < len(CMAPSS_REQUIRED_FILES):
                    raise RuntimeError(
                        f"в архиве найдено только {found} из "
                        f"{len(CMAPSS_REQUIRED_FILES)} файлов")
            print(f"  Готово: {found} файлов в {data_dir}")
            return data_dir
        except Exception as e:
            last_err = e
            print(f"  [WARN] Источник недоступен ({type(e).__name__}: {e})")

    raise RuntimeError(
        f"Не удалось загрузить C-MAPSS ни с одного источника "
        f"({len(CMAPSS_URLS)} URL). Последняя ошибка: {last_err}")


def _extract_recursive(archive_path: str, out_dir: str) -> None:
    """Распаковывает zip и рекурсивно вложенные zip (CMAPSSData.zip)."""

    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(out_dir)
    os.remove(archive_path)
    # Пока в дереве появляются новые zip - распаковывает и их
    while True:
        inner_zips = [os.path.join(root, name)
                      for root, _dirs, files in os.walk(out_dir)
                      for name in files if name.lower().endswith('.zip')]
        if not inner_zips:
            return
        for inner in inner_zips:
            with zipfile.ZipFile(inner) as zf:
                zf.extractall(inner + '_unpacked')
            os.remove(inner)


def _collect_txt(search_dir: str, data_dir: str) -> int:
    """
    Переносит нужные txt (по именам из CMAPSS_REQUIRED_FILES) в data_dir.

    Возвращает число файлов комплекта, лежащих в data_dir после переноса.
    """

    wanted = set(CMAPSS_REQUIRED_FILES) | {'readme.txt'}
    for root, _dirs, files in os.walk(search_dir):
        for name in files:
            if name in wanted:
                shutil.move(os.path.join(root, name),
                            os.path.join(data_dir, name))
    return sum(os.path.isfile(os.path.join(data_dir, f))
               for f in CMAPSS_REQUIRED_FILES)


# Парсинг и разметка

def load_subset(subset: str, split: str = 'train',
                data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Читает train_/test_FD00X.txt в DataFrame схемы платформы.

    Колонки: pump_id ('FD001_E001'), timestamp (= cycle, int), unit, cycle,
    op1..op3, s1..s21. Файл - 26 колонок через пробел, без заголовка.
    """

    if subset not in CMAPSS_SUBSETS:
        raise ValueError(f"Неизвестный сабсет: {subset} (ожидается {CMAPSS_SUBSETS})")
    path = os.path.join(data_dir, f'{split}_{subset}.txt')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Нет файла {path} - сначала вызовите download_cmapss().")

    df = pd.read_csv(path, sep=r'\s+', header=None, names=CMAPSS_RAW_COLUMNS)
    df['pump_id'] = [ENGINE_ID_FMT.format(subset=subset, unit=int(u))
                     for u in df['unit']]
    # timestamp = номер цикла: препроцессор сортирует по ['pump_id','timestamp']
    df['timestamp'] = df['cycle'].astype(int)
    return df


def compute_rul(df: pd.DataFrame, subset: str, split: str,
                data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Добавляет колонку RUL (остаток циклов до отказа) для каждой строки.

    train: траектория доживает до отказа -> RUL = max(cycle) - cycle.
    test:  траектория обрезана; истинный финальный RUL берётся из
           RUL_FD00X.txt (строка i = юнит i) -> RUL = final + max(cycle) - cycle.
    """

    df = df.copy()
    last_cycle = df.groupby('unit')['cycle'].transform('max')
    if split == 'train':
        df['RUL'] = last_cycle - df['cycle']
        return df

    rul_path = os.path.join(data_dir, f'RUL_{subset}.txt')
    finals = pd.read_csv(rul_path, header=None).iloc[:, 0].astype(int)
    final_by_unit = {unit: int(finals.iloc[unit - 1])
                     for unit in df['unit'].unique()}
    df['RUL'] = df['unit'].map(final_by_unit) + (last_cycle - df['cycle'])
    return df


def add_severity_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    target 0/1/2 из RUL: piecewise-пороги RUL_WARNING / RUL_CRITICAL.

    Та же трёхклассовая постановка, что у насосной модели тяжести:
    0 Норма, 1 Предупреждение (деградация обнаружима), 2 Авария (предотказ).
    """

    df = df.copy()
    df['target'] = np.select(
        [df['RUL'] <= RUL_CRITICAL, df['RUL'] <= RUL_WARNING],
        [2, 1], default=0).astype(int)
    return df


# Режимная нормализация (FD002 / FD004: 6 полётных режимов)

def regime_ids(df: pd.DataFrame) -> pd.Series:
    """
    Id полётного режима (0..5) по ближайшему якорю op1.

    Шесть режимов C-MAPSS однозначно разделяются высотной настройкой op1
    (якоря 0/10/20/25/35/42 при шуме ~1e-3) - кластеризация не нужна,
    привязка детерминированна и воспроизводима.
    """

    anchors = np.asarray(CMAPSS_REGIME_OP1_ANCHORS)
    return pd.Series(
        np.abs(df['op1'].to_numpy()[:, None] - anchors[None, :]).argmin(axis=1),
        index=df.index, name='regime')


def fit_regime_normalizer(
        train_df: pd.DataFrame) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """
    Статистики z-score сенсоров по режимам, ТОЛЬКО на train (нет утечки).

    Возвращает {regime_id: {sensor: (mean, std)}}; std==0 заменяется на 1
    (константный в режиме сенсор превращается в ноль, а не в NaN).
    """

    reg = regime_ids(train_df)
    stats: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for r, grp in train_df.groupby(reg):
        stats[int(r)] = {}
        for s in CMAPSS_SENSORS:
            mean = float(grp[s].mean())
            std = float(grp[s].std(ddof=1))
            stats[int(r)][s] = (mean, std if std > 1e-12 else 1.0)
    return stats


def apply_regime_normalization(
        df: pd.DataFrame,
        stats: Dict[int, Dict[str, Tuple[float, float]]]) -> pd.DataFrame:
    """Приводит сенсоры к z-score внутри полётного режима по train-статистикам."""

    df = df.copy()
    reg = regime_ids(df)
    for r in sorted(stats):
        mask = (reg == r).to_numpy()
        if not mask.any():
            continue
        for s in CMAPSS_SENSORS:
            mean, std = stats[r][s]
            df.loc[mask, s] = (df.loc[mask, s] - mean) / std
    return df


# Препроцессор: наследует всю rolling-механику насосного пайплайна

class CmapssPreprocessor(DataPreprocessor):
    """
    DataPreprocessor для C-MAPSS: та же механика, другие сенсоры и окна.

    От родителя без изменений: shift(1) против data leakage, groupby по
    pump_id (окна не перетекают между двигателями), min_periods=w для
    mean/max, min_periods=2 для std, dropna при обучении. Переопределено:
      - sensors: 14 информативных сенсоров C-MAPSS вместо 4 насосных;
      - window_sizes: ЦИКЛЫ (CMAPSS_WINDOW_SIZES) вместо минут;
      - лаг diff-признака: CMAPSS_DIFF_LAG (родительский diff_30 требовал бы
        31 цикл прогрева - самые короткие обрезанные test-траектории
        потеряли бы все строки).
    """

    def __init__(self):
        super().__init__(window_sizes=list(CMAPSS_WINDOW_SIZES))
        self.sensors = list(CMAPSS_SENSORS)
        self.diff_lag = int(CMAPSS_DIFF_LAG)
        self.FEATURE_COLS = self._build_feature_cols()

    def _build_feature_cols(self) -> list:
        """Контракт признаков: {sensor}_{stat}_{w} + {sensor}_diff_{lag}."""

        # При первом вызове из DataPreprocessor.__init__ собственные атрибуты
        # ещё не выставлены (getattr-дефолт) - финальный список пересобирается
        # в конце __init__ этого класса, когда sensors/diff_lag уже свои.
        diff_lag = getattr(self, 'diff_lag', 30)
        cols = []
        for col in self.sensors:
            for w in self.window_sizes:
                cols += [f'{col}_mean_{w}', f'{col}_std_{w}', f'{col}_max_{w}']
            cols.append(f'{col}_diff_{diff_lag}')
        return cols

    def _calculate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Родительская механика с параметрическим лагом diff.

        Дублирует _calculate_rolling_features родителя осознанно: контракт
        родителя (жёсткий diff_30) заморожен паритет-тестом
        online_parity_test.py и не параметризуется, чтобы не трогать боевой
        насосный пайплайн.
        """

        lag = self.diff_lag

        def apply_rolling(group: pd.DataFrame) -> pd.DataFrame:
            # Колонки копятся в dict и склеиваются одним concat: 140 признаков,
            # повставочная запись в DataFrame фрагментирует его и сыплет
            # PerformanceWarning на каждый столбец
            feats: Dict[str, pd.Series] = {}
            for col in self.sensors:
                shifted = group[col].shift(1)  # текущий цикл исключён из окна
                for w in self.window_sizes:
                    feats[f'{col}_mean_{w}'] = shifted.rolling(w, min_periods=w).mean()
                    feats[f'{col}_max_{w}'] = shifted.rolling(w, min_periods=w).max()
                    feats[f'{col}_std_{w}'] = shifted.rolling(w, min_periods=2).std()
                # Градиент за lag циклов: (T-1) - (T-1-lag), ровно lag шагов
                feats[f'{col}_diff_{lag}'] = (group[col].shift(1)
                                              - group[col].shift(lag + 1))
            # cast: pandas-stubs выводят concat(Mapping) как Series;
            # при axis=1 это DataFrame
            return cast(pd.DataFrame, pd.concat(feats, axis=1))

        parts = [apply_rolling(group) for _, group in df.groupby('pump_id')]
        rolling_features = pd.concat(parts).reindex(df.index)
        return pd.concat([df, rolling_features], axis=1)


# Оркестрация: сабсет -> готовые train/test для ML

def prepare_subset(subset: str, data_dir: str = DATA_DIR,
                   verbose: bool = True
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, CmapssPreprocessor]:
    """
    Полная подготовка сабсета: загрузка -> RUL/target -> нормализация ->
    rolling-признаки. Возвращает (train_df, test_df, preprocessor).

    Обучение - официальный train (run-to-failure), оценка - официальный
    test с истинным RUL из RUL_FD00X.txt: сплит по агрегатам задан самим
    бенчмарком (двигатели test не встречаются в train - тот же принцип
    Group Split, что и у насосного пайплайна).
    """

    download_cmapss(data_dir)

    frames = {}
    for split in ('train', 'test'):
        df = load_subset(subset, split, data_dir)
        df = compute_rul(df, subset, split, data_dir)
        df = add_severity_target(df)
        frames[split] = df

    if subset in CMAPSS_MULTI_REGIME_SUBSETS:
        stats = fit_regime_normalizer(frames['train'])
        frames = {split: apply_regime_normalization(df, stats)
                  for split, df in frames.items()}
        if verbose:
            print(f"  {subset}: z-нормализация сенсоров по {len(stats)} "
                  f"полётным режимам (статистики - только train)")

    pre = CmapssPreprocessor()
    out = {}
    for split, df in frames.items():
        n_raw, engines_raw = len(df), df['pump_id'].nunique()
        # is_training=True: dropna по FEATURE_COLS (прогрев окон на двигатель);
        # маппинг state->target внутри process() пропускается - колонки state
        # нет, target уже посчитан из RUL
        out[split] = pre.process(df, is_training=True)
        if verbose:
            kept, engines = len(out[split]), out[split]['pump_id'].nunique()
            print(f"  {subset} {split}: {n_raw:,} строк / {engines_raw} двиг. "
                  f"-> после прогрева окон {kept:,} строк / {engines} двиг. "
                  f"({100 * kept / n_raw:.1f}%)")

    return out['train'], out['test'], pre


def class_balance(df: pd.DataFrame) -> str:
    """Строка с балансом классов target для отчётов."""

    counts = df['target'].value_counts().sort_index()
    total = len(df)
    return ' | '.join(
        f"{CMAPSS_STAGE_LABELS[int(c)]}: {counts.get(c, 0):,} "
        f"({100 * counts.get(c, 0) / total:.1f}%)"
        for c in (0, 1, 2))


# Самотест (загрузка + сводка + smoke препроцессинга)

if __name__ == '__main__':
    print('=' * 62)
    print('C-MAPSS: загрузка и подготовка данных (этап 1)')
    print('=' * 62)
    download_cmapss()

    print('\nСводка по сабсетам (метки из RUL: '
          f'предупреждение <= {RUL_WARNING}, авария <= {RUL_CRITICAL} циклов):')
    for _subset in CMAPSS_SUBSETS:
        _info = CMAPSS_SUBSET_INFO[_subset]
        for _split in ('train', 'test'):
            _df = add_severity_target(
                compute_rul(load_subset(_subset, _split), _subset, _split))
            print(f"  {_info['label']:<32} {_split:<5}: "
                  f"{len(_df):>6,} строк, {_df['pump_id'].nunique():>3} двиг. | "
                  f"{class_balance(_df)}")

    print('\nSmoke препроцессинга FD001 (rolling-признаки в циклах):')
    _train, _test, _pre = prepare_subset('FD001')
    _fc = _pre.FEATURE_COLS
    print(f"  Признаков: {len(_fc)} = {len(_pre.sensors)} сенсоров x "
          f"({len(_pre.window_sizes)} окна x 3 статистики + diff_{_pre.diff_lag})")
    assert not _train[_fc].isna().any().any(), 'NaN в train-признаках после dropna'
    assert not _test[_fc].isna().any().any(), 'NaN в test-признаках после dropna'
    assert _train['target'].isin([0, 1, 2]).all()
    print(f"  Баланс train: {class_balance(_train)}")
    print(f"  Баланс test:  {class_balance(_test)}")
    print('\n[OK] Этап 1 пройден: данные загружены, размечены, признаки собраны.')
