"""
Permutation test (sanity-check) классификатора ТИПА отказа
==========================================================
experiments/validation/permutation_test.py

Цель: доказать, что классификатор типа отказа учит РЕАЛЬНЫЕ физические сигнатуры
датчиков, а не паразитные артефакты выборки (утечки, порядок строк, дисбаланс).

Метод (label permutation): метки `fault_target` в обучающей выборке СЛУЧАЙНО
перемешиваются - связь «признаки → тип» намеренно разрушается. Модель учится на
бессмысленных метках, и её balanced accuracy на тесте ОБЯЗАНА упасть к случайной
≈ 1/3 (три равновероятных типа: overheat / cavitation / electrical).

Интерпретация:
    - ≈0.33 - корректно: в перемешанных метках сигнала нет, модель не «жульничает»
      через паразитные паттерны → высокая точность настоящей модели достоверна.
    - заметно >0.33 - тревога: модель ловит утечку (напр. через pump_id/время),
      и реальные 0.99 были бы недостоверны.

Это контрольный антипод обучения: смысл высокой точности основной модели есть
только тогда, когда на перемешанных метках точность рушится до случайной.

Запуск: python experiments/validation/permutation_test.py
Предусловие: собран data/processed/fault_type_pumps_dataset.csv (data_preprocessor.py).
"""

import pandas as pd
import numpy as np
import os
import xgboost as xgb
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sklearn.metrics import balanced_accuracy_score
from config.settings import TRAIN_PUMPS, TEST_PUMPS, WINDOW_SIZES
from src.data.data_preprocessor import DataPreprocessor

# Тот же контракт признаков (FEATURE_COLS), что и у боевого классификатора типа -
# проба должна видеть ровно те же 40 rolling-признаков, без пересчёта
_preprocessor = DataPreprocessor(window_sizes=WINDOW_SIZES)
FEATURE_COLS = _preprocessor.FEATURE_COLS

project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))))
processed_data_path = os.path.join(project_root, 'data', 'processed', 
                                    'fault_type_pumps_dataset.csv')

print("Загрузка обработанных данных...")
if not os.path.exists(processed_data_path):
    raise FileNotFoundError(
        f"Файл не найден: {processed_data_path}. Сначала запустите data_preprocessor.py"
    )
df = pd.read_csv(processed_data_path)

# Group Split по агрегату (как в обучении): train - MNHV_001..004, test - MNHV_005
# Тест на неизвестном насосе исключает запоминание конкретного оборудования
df_train = df[df['pump_id'].isin(TRAIN_PUMPS)].copy()
df_test = df[df['pump_id'].isin(TEST_PUMPS)].copy()

# X - только признаки FEATURE_COLS; y - целевой тип отказа (0/1/2).
X_train, y_train = df_train[FEATURE_COLS], df_train['fault_target']
X_test, y_test = df_test[FEATURE_COLS], df_test['fault_target']

# Аудит чистоты эксперимента:
#   1) в матрице признаков нет посторонних колонок (только контракт FEATURE_COLS);
#   2) обучающие и тестовый насосы не пересекаются (нет утечки через агрегат).
assert list(X_train.columns) == FEATURE_COLS
assert set(df_train['pump_id']).isdisjoint(df_test['pump_id'])

# КЛЮЧЕВОЙ ШАГ: случайно перемешиваем обучающие метки (фиксированный seed для
# воспроизводимости). Признаки остаются на месте, но их связь с типом отказа
# разрушена - учить теперь нечему, кроме шума.
y_shuf = np.random.default_rng(42).permutation(y_train.values)

# Проба теми же гиперпараметрами, что и боевой XGBoost-классификатор типа -
# чтобы сравнение было честным (отличие только в перемешанных метках).
probe = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=300,
                          max_depth=6, eval_metric='mlogloss', random_state=42, n_jobs=-1)
probe.fit(X_train, y_shuf)

# Ожидаемый результат ≈ 0.33 (случайный выбор из 3 типов). Заметно выше - сигнал
# о паразитном паттерне/утечке, который обесценил бы реальную точность модели.
print("balanced acc на перемешанных метках:", balanced_accuracy_score(y_test, probe.predict(X_test)))