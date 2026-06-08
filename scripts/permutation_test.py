import pandas as pd
import numpy as np
import os
import xgboost as xgb
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from sklearn.metrics import balanced_accuracy_score
from config.settings import TRAIN_PUMPS, TEST_PUMPS, WINDOW_SIZES
from src.data_preprocessor import DataPreprocessor
_preprocessor = DataPreprocessor(window_sizes=WINDOW_SIZES)
FEATURE_COLS = _preprocessor.FEATURE_COLS

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_data_path = os.path.join(project_root, 'data', 'processed', 
                                    'fault_type_pumps_dataset.csv')

print("Загрузка обработанных данных...")
if not os.path.exists(processed_data_path):
    raise FileNotFoundError(
        f"Файл не найден: {processed_data_path}. Сначала запустите data_preprocessor.py"
    )
df = pd.read_csv(processed_data_path)
df_train = df[df['pump_id'].isin(TRAIN_PUMPS)].copy()
df_test = df[df['pump_id'].isin(TEST_PUMPS)].copy()
X_train, y_train = df_train[FEATURE_COLS], df_train['fault_target']
X_test, y_test = df_test[FEATURE_COLS], df_test['fault_target']
# Аудит: в X только признаки
assert list(X_train.columns) == FEATURE_COLS
assert set(df_train['pump_id']).isdisjoint(df_test['pump_id'])
# Перемешиваем метки — точность ОБЯЗАНА рухнуть к ~1/3
y_shuf = np.random.default_rng(42).permutation(y_train.values)
probe = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=300,
                          max_depth=6, eval_metric='mlogloss', random_state=42, n_jobs=-1)
probe.fit(X_train, y_shuf)
print("balanced acc на перемешанных метках:", balanced_accuracy_score(y_test, probe.predict(X_test)))