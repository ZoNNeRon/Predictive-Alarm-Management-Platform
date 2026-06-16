"""
Классификатор ТИПА отказа (Fault-Type Classifier) — вторая модель платформы
===========================================================================
Второй этап иерархической классификации. Первый этап (ml_pipeline.py) определяет
ТЯЖЕСТЬ (0=Норма / 1=Warning / 2=Авария). Этот модуль обучает модель, которая на
аварийных строках называет ФИЗИЧЕСКИЙ ТИП отказа: overheat / cavitation / electrical.

Зачем отдельная модель, а не SHAP-эвристика:
    Прежняя эвристика по SHAP-сигнатуре давала точность ~0.61 (перегрев массово
    утекал в кавитацию) — пороги в шкале SHAP не имеют физического смысла и
    «плывут» при переобучении. Обучаемый классификатор на тех же признаках 
    убирает ручные пороги и даёт измеримую, защищаемую точность.

Дисциплина эксперимента:
    - Group Split по агрегату: обучение на MNHV_001..004, тест на MNHV_005
      (как в ml_pipeline.py) — доказывает обобщение на новый насос.
    - Только предупредительные/аварийные строки (target==1 & ==2) — 
      согласование train/inference.
    - Веса классов 'balanced': перегрев (~52%) не должен задавить электрику (~20%).

Метрики (отличаются от модели тяжести):
    Здесь классы умеренно несбалансированы (нет редкого класса ~0.2%), поэтому
    главные метрики — macro-F1 и balanced accuracy (не дают мажорному классу
    доминировать), плюс per-class recall и confusion matrix. PR-AUC, критичный
    для модели тяжести, здесь вторичен.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score)
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config.settings import TRAIN_PUMPS, TEST_PUMPS, WINDOW_SIZES, FAULT_TYPES, FAULT_LABELS
from src.data.data_preprocessor import DataPreprocessor
from src.visualisation.ml_visualisation import plot_fault_classifier

_preprocessor = DataPreprocessor(window_sizes=WINDOW_SIZES)
FEATURE_COLS = _preprocessor.FEATURE_COLS

# Имена классов в порядке индексов FAULT_TYPES (0,1,2)
TARGET_NAMES = [FAULT_LABELS[ft] for ft in FAULT_TYPES]


# Оценка одной модели

def evaluate_fault_model(name: str, short_name: str, model,
                         X_test: pd.DataFrame, y_test: pd.Series,
                         stage_test: pd.Series) -> dict:
    """
    Полный набор метрик для классификатора типа отказа.

    macro-F1 — среднее F1 по типам без учёта их размера (главная метрика).
    balanced accuracy — среднее recall по классам; устойчиво к дисбалансу типов.
    per-class recall — какую долю каждого типа модель называет верно (важно для
                       редкой электрики).
    confusion matrix — конкретные перепутывания (например, перегрев↔кавитация).
    """

    y_true = y_test.to_numpy()
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(y_true, y_pred, labels=[0, 1, 2],
                                   target_names=TARGET_NAMES, 
                                   output_dict=True, zero_division=0)
    
    # Раздельно по стадии: 1=Warning (раннее), 2=Critical (авария)
    stage = stage_test.to_numpy()
    stage_recall = {}
    for sv, sname in [(1, 'warning'), (2, 'critical')]:
        mask = stage == sv
        rep = classification_report(y_true[mask], y_pred[mask], labels=[0, 1, 2],
                                    target_names=TARGET_NAMES, 
                                    output_dict=True, zero_division=0)
        stage_recall[sname] = {ft: rep[FAULT_LABELS[ft]]['recall'] for ft in FAULT_TYPES} # type: ignore


    print("\nМатрица ошибок (строки — факт, столбцы — предсказание):")
    header = "            " + "  ".join(f"{n[:12]:>12s}" for n in TARGET_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{TARGET_NAMES[i][:12]:>12s}  " + "  ".join(f"{v:>12d}" for v in row))

    print("\nОтчёт по классификации:")
    print(classification_report(y_test, y_pred, labels=[0, 1, 2],
                                target_names=TARGET_NAMES))

    macro_f1 = report['macro avg']['f1-score']                  # type: ignore
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    per_class_recall = {ft: report[FAULT_LABELS[ft]]['recall']  # type: ignore
                        for ft in FAULT_TYPES}

    print(f"macro-F1:          {macro_f1:.4f}")
    print(f"balanced accuracy: {bal_acc:.4f}")
    for ft in FAULT_TYPES:
        print(f"recall [{FAULT_LABELS[ft]:18s}]: {per_class_recall[ft]:.4f}")

    return {
        'name': name,
        'label': short_name,
        'macro_f1': macro_f1,
        'balanced_acc': bal_acc,
        'recall_overheat': per_class_recall['overheat'],
        'recall_cavitation': per_class_recall['cavitation'],
        'recall_electrical': per_class_recall['electrical'],
        'cm': cm,
        'stage_recall': stage_recall,
        'y_pred': y_pred,
    }


# Точка входа

if __name__ == "__main__":
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))))
    fault_data_path = os.path.join(project_root, 'data', 'processed', 'fault_type_pumps_dataset.csv')
    models_dir = os.path.join(project_root, 'models', 'fault_type')
    os.makedirs(models_dir, exist_ok=True)

    # 1. Загрузка производного датасета
    print("Загрузка датасета типа отказа...")
    if not os.path.exists(fault_data_path):
        raise FileNotFoundError(
            f"Файл не найден: {fault_data_path}. Сначала запустите data_preprocessor.py."
        )
    df = pd.read_csv(fault_data_path)

    # 2. Group Split по оборудованию
    df_train = df[df['pump_id'].isin(TRAIN_PUMPS)].copy()
    df_test = df[df['pump_id'].isin(TEST_PUMPS)].copy()

    print(f"Обучающая выборка (насосы 1-4): {len(df_train):,} аварийных строк")
    print(f"Тестовая выборка  (насос 5):    {len(df_test):,} аварийных строк")
    print("\nБаланс типов в обучении:")
    print(df_train['fault_type'].value_counts())

    X_train, y_train = df_train[FEATURE_COLS], df_train['fault_target']
    X_test, y_test = df_test[FEATURE_COLS], df_test['fault_target']

    # 3. Веса для компенсации дисбаланса типов (перегрев >> электрика)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # 4. Обучение трёх моделей (тот же набор, что для модели тяжести — для сопоставимости)
    print("\nОбучение моделей классификации типа отказа...")

    lr_model = LogisticRegression(
        class_weight='balanced', max_iter=1000,
        solver='saga', random_state=42
    )
    lr_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=300, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

    # 5. Оценка
    metrics = [
        evaluate_fault_model("Logistic Regression (Baseline)", "LogReg", 
                             lr_model, X_test, y_test, df_test['severity_stage']),
        evaluate_fault_model("Random Forest (Bagging Ensemble)", "RF", 
                             rf_model, X_test, y_test, df_test['severity_stage']),
        evaluate_fault_model("XGBoost (Boosting Ensemble)", "XGBoost", 
                             xgb_model, X_test, y_test, df_test['severity_stage']),
    ]

    # 6. Сводная таблица для выбора модели
    print("СВОДНАЯ ТАБЛИЦА (тест: MNHV_005)")
    summary = pd.DataFrame([{
        'Модель': m['label'],
        'macro-F1': round(m['macro_f1'], 4),
        'balanced_acc': round(m['balanced_acc'], 4),
        'recall_Перегрев': round(m['recall_overheat'], 4),
        'recall_Кавитация': round(m['recall_cavitation'], 4),
        'recall_Электрика': round(m['recall_electrical'], 4),
    } for m in metrics]).set_index('Модель')
    print(summary.to_string())

    best = max(metrics, key=lambda m: m['macro_f1'])
    print(f"\nЛучшая модель по macro-F1: {best['name']} ({best['macro_f1']:.4f})")

    # 7. Сохранение всех трёх моделей (XGBoost — для интеграции в XAI)
    joblib.dump(lr_model, os.path.join(models_dir, 'fault_lr_model.joblib'))
    joblib.dump(rf_model, os.path.join(models_dir, 'fault_rf_model.joblib'))
    joblib.dump(xgb_model, os.path.join(models_dir, 'fault_xgboost_model.joblib'))

    graphs_dir = os.path.join(project_root, 'artifacts', 'graphs')
    tables_dir = os.path.join(project_root, 'artifacts', 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    summary.to_csv(os.path.join(tables_dir, 'ML_fault_classifier_summary.csv'))
    print(f"\nМодели сохранены в: {models_dir}")
    print(f"Сводная таблица: {os.path.join(tables_dir, 'fault_classifier_summary.csv')}")

    # 8. Построение графиков
    plot_fault_classifier(metrics, graphs_dir)   # лучшую модель выберет по macro-F1
