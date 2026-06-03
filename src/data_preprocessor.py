import pandas as pd
import numpy as np
import os
from typing import Optional


class DataPreprocessor:
    """
    Класс для инженерной подготовки данных.
    Работает как в пакетном режиме (обучение), так и в потоковом (инференс реального времени).
    """
    # Физические пороги (для валидации и документации)
    VIB_WARNING = 3.0
    VIB_CRITICAL = 8.0
    TEMP_WARNING = 82.0
    TEMP_CRITICAL = 93.0

    def __init__(self, window_sizes: Optional[list[int]] = None):
        self.window_sizes = window_sizes or [15, 30, 60]
        self.sensors = ['vibration', 'temperature', 'current', 'pressure']

        # Явный список признаков для ML — сырые значения датчиков исключены намеренно.
        # Сырые колонки остаются в датасете для диагностики и визуализации,
        # но не попадают в обучение, чтобы одиночный выброс-помеха не влиял на модель.
        self.FEATURE_COLS = self._build_feature_cols()

    def _build_feature_cols(self) -> list:
        """Генерирует список имён rolling-признаков по шаблону."""
        cols = []
        for col in self.sensors:
            for w in self.window_sizes:
                cols += [f'{col}_mean_{w}', f'{col}_std_{w}', f'{col}_max_{w}']
            cols.append(f'{col}_diff_30')
        return cols

    # Главный метод

    def process(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Главный метод обработки датафрейма.

        Args:
            df:           DataFrame с сырыми данными (output data_generator.py).
            is_training:  True  → dropna + маппинг target (пакетный режим)
                          False → без dropna, без фильтрации Off/Startup (инференс)

        Returns:
            Обработанный DataFrame. Список признаков для ML — self.FEATURE_COLS.
        """
        # 1. Гарантируется хронологический порядок внутри каждого насоса
        df = df.sort_values(by=['pump_id', 'timestamp']).copy()

        # 2. Rolling features — до фильтрации состояний, чтобы окно было непрерывным
        df = self._calculate_rolling_features(df)

        # 3. Удаляются флаги аппаратных помех: модель не должна их знать,
        # она должна игнорировать помехи через сглаживание скользящим окном.
        # fault_type намеренно сохраняется — нужен fault_recall_analysis.py для валидации;
        # в ML не попадёт, так как отсутствует в FEATURE_COLS.
        df = df.drop(columns=[c for c in ['sensor_anomaly',
                                           'anomaly_vibration', 'anomaly_temperature',
                                           'anomaly_current']
                               if c in df.columns])

        # 4. Маппинг таргета и фильтрация Off/Startup — только при обучении
        if is_training and 'state' in df.columns:
            df = self._prepare_labels(df)

        # 5. При обучении честно удаляем строки, где окно не накопило window_max точек
        if is_training:
            df = df.dropna(subset=self.FEATURE_COLS).reset_index(drop=True)

        return df

    # Приватные методы

    def _calculate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчёт статистик скользящего окна с защитой от Data Leakage.

        Для предсказания состояния в момент T используются только данные [T-W ... T-1].
        Реализация: shift(1) перед rolling гарантирует, что T не попадает в окно.
        """
        def apply_rolling(group: pd.DataFrame) -> pd.DataFrame:
            res = pd.DataFrame(index=group.index)

            for col in self.sensors:
                # shift(1): в строке T — значение T-1. Текущий момент исключён
                shifted = group[col].shift(1)

                for w in self.window_sizes:
                    # min_periods=w: пока окно не заполнено — NaN
                    # dropna() уберёт эти строки при обучении
                    res[f'{col}_mean_{w}'] = shifted.rolling(window=w, min_periods=w).mean()
                    res[f'{col}_max_{w}'] = shifted.rolling(window=w, min_periods=w).max()
                    # std требует минимум 2 точки; меньше — NaN
                    res[f'{col}_std_{w}'] = shifted.rolling(window=w, min_periods=2).std()

                # Градиент за 30 минут: (T-1) − (T-31) = ровно 30 шагов
                # Показывает скорость и направление тренда — ключевой признак для XGBoost
                # при пограничных значениях (например, вибрация 3.0 мм/с, но быстро растёт)
                res[f'{col}_diff_30'] = group[col].shift(1) - group[col].shift(31)

            return res

        # groupby изолирует насосы: окно не перетекает с конца MNHV_001 на начало MNHV_002
        rolling_features = df.groupby('pump_id', group_keys=False).apply(apply_rolling, include_groups=False) # type: ignore
        return pd.concat([df, rolling_features], axis=1)

    def _prepare_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Маппинг состояний в ML-классы и фильтрация нерелевантных состояний.

        Off (0) и Startup (1) перехватываются State-based логикой до ML-слоя,
        поэтому из обучающей выборки они исключаются.

        Маппинг:  2 (Healthy)     → 0 (Норма)
                  3 (Degradation) → 1 (Warning)
                  4 (Critical)    → 2 (Авария)
        """
        df = df[df['state'].isin([2, 3, 4])].copy() # type: ignore
        df['target'] = df['state'].map({2: 0, 3: 1, 4: 2})
        return df


# Точка входа (пакетный режим)

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'enterprise_pump_fleet.csv')
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'processed_features.csv')

    print("Загрузка сырых данных предприятия...")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(
            f"Файл не найден: {raw_data_path}. Сначала запустите data_generator.py"
        )

    df_raw = pd.read_csv(raw_data_path, parse_dates=['timestamp'])

    print("Инициализация DataPreprocessor...")
    preprocessor = DataPreprocessor(window_sizes=[15, 30, 60])

    print("Расчёт rolling-признаков и маппинг таргетов...")
    df_processed = preprocessor.process(df_raw, is_training=True)

    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df_processed.to_csv(processed_data_path, index=False)

    print(f"\nГотово! Датасет сохранён:\n  {processed_data_path}")
    print(f"Форма: {df_processed.shape} (строк x колонок)")
    print(f"\nПризнаки для ML ({len(preprocessor.FEATURE_COLS)} шт.):")
    print(preprocessor.FEATURE_COLS)
    print(f"\nБаланс классов (0=Норма, 1=Предупреждение, 2=Авария):")
    print(df_processed['target'].value_counts().sort_index())