"""
Аналитическое ядро предиктивной диагностики (ML Pipeline)
=========================================================
Модуль отвечает за обучение, изоляционное тестирование (Group Split) и 
сравнение предиктивных моделей (Logistic Regression, Random Forest, XGBoost) 
для выявления деградации оборудования. Включает программную реализацию 
`AlarmManager` для интеллектуального подавления ложных тревог (Alarm Shelving) 
в пусковых и нерабочих режимах насоса.

В рамках модуля обучается первая и основная модель машинного обучения - модель тяжести,
выявленибщая ненормальный режим работы, т.е. статус "Предупреждение" или "Отказ". 
Обучается на датасете preprocessed_pumps_dataset.csv.
"""

import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (classification_report, confusion_matrix,
                              average_precision_score)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')
import joblib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config.settings import TRAIN_PUMPS, TEST_PUMPS, WINDOW_SIZES
from src.visualisation.ml_visualisation import plot_all
from src.data.data_preprocessor import DataPreprocessor
_preprocessor = DataPreprocessor(window_sizes=WINDOW_SIZES)
FEATURE_COLS = _preprocessor.FEATURE_COLS
from fault_recall_analysis import analyze_fault_recall

# AlarmManager

class AlarmManager:
    """
    Программная реализация защиты от лавин аварийных сигналов (Alarm Flood / Alarm Shelving).

    Принцип: ML-модель работает только тогда, когда насос физически находится
    в штатном режиме работы. Для состояний Off (0) и Startup (1) прогноз
    принудительно гасится — любые аномальные показатели в эти периоды
    являются нормой (пусковые токи, гидроудар) и не должны порождать сигнал.

    Это первый уровень фильтрации в архитектуре платформы (State-based Alarming),
    поверх которого работает ML-слой.
    """
    def __init__(self, model):
        self.model = model

    def predict_with_context(self, features: pd.DataFrame, raw_state: int) -> int:
        """
        Args:
            features: DataFrame с rolling-признаками (одна строка для инференса)
            raw_state: Физическое состояние агрегата из State Machine (0–4)

        Returns:
            0 (Норма) — если насос Off или Startup (принудительно)
            0/1/2 — прогноз ML-модели, если насос в рабочем режиме
        """
        if raw_state in [0, 1]:
            return 0  # Alarm Shelving: гасит любой тревожный сигнал

        return int(self.model.predict(features)[0])


# Оценка модели 

def evaluate_model(name: str, short_name: str, model, 
                   X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Вычисляет и выводит полный набор метрик для одной модели.

    Метрики и их обоснование для задачи предиктивного обслуживания:

    Confusion Matrix — показывает конкретную цену каждого типа ошибки.
        В нефтегазовой отрасли False Negative (пропущенная авария, ошибка 2 рода) 
        несравнимо дороже False Positive (ложная тревога, ошибка 1 рода), 
        поэтому матрица обязательна.

    F1-Macro — среднее F1 по всем классам без учёта их размера.
        Предотвращает "утопание" редких классов (Авария — ~0.2% строк)
        в weighted-среднем, которое доминируется классом Норма.

    Recall класса "Авария" (KPI системы) — доля реальных аварий,
        которую система обнаружила. Это ключевой бизнес-показатель:
        Recall=0.95 означает, что 95 из 100 аварий будут пойманы заблаговременно.

    PR-AUC — площадь под кривой Precision-Recall.
        При сильном дисбалансе классов ROC-AUC завышается (много TN в знаменателе).
        PR-AUC честно оценивает качество работы на редком классе,
        не "разбавляя" результат тривиальными True Negative.
    """

    print(f"\n{'='*50}")
    print(f"РЕЗУЛЬТАТЫ МОДЕЛИ: {name}")
    print(f"{'='*50}")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Confusion Matrix
    print("\nМатрица ошибок (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))

    # Classification Report
    target_names = ['0: Норма', '1: Warning', '2: Авария']
    report = classification_report(y_test, y_pred,
                                   target_names=target_names, output_dict=True)
    print("\nОтчет по классификации (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # PR-AUC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    pr_auc_macro = average_precision_score(y_test_bin, y_proba, average="macro")
    pr_auc_critical = average_precision_score(y_test_bin[:, 2], y_proba[:, 2]) # type: ignore

    # Recall класса "Авария" — явный KPI
    recall_critical = report['2: Авария']['recall'] # type: ignore

    print(f"PR-AUC (Macro Average):    {pr_auc_macro:.4f}")
    print(f"PR-AUC (Класс 'Авария'):   {pr_auc_critical:.4f}")
    print(f"Recall (Класс 'Авария'):   {recall_critical:.4f}")

    return {
        'name': name,
        'label': short_name,                            # имя для графиков
        'f1_macro': report['macro avg']['f1-score'],    # type: ignore
        'f1_warning': report['1: Warning']['f1-score'], # type: ignore
        'f1_critical': report['2: Авария']['f1-score'], # type: ignore
        'recall_critical': recall_critical,
        'pr_auc_macro': pr_auc_macro,
        'pr_auc_critical': pr_auc_critical,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'cm': confusion_matrix(y_test, y_pred),
    }


# Точка входа 

if __name__ == "__main__":
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))))
    processed_data_path = os.path.join(project_root, 'data', 'processed', 
                                       'preprocessed_pumps_dataset.csv')
    save_graps_dir = os.path.join(project_root, 'artifacts', 'graphs')
    save_tables_dir = os.path.join(project_root, 'artifacts', 'tables')

    # 1. Загрузка данных
    print("Загрузка обработанных данных...")
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(
            f"Файл не найден: {processed_data_path}. Сначала запустите data_preprocessor.py"
        )
    df = pd.read_csv(processed_data_path)

    # 2. Group Split по оборудованию (доказывает обобщение на новый агрегат)
    df_train = df[df['pump_id'].isin(TRAIN_PUMPS)].copy()
    df_test = df[df['pump_id'].isin(TEST_PUMPS)].copy()

    print(f"Обучающая выборка (насосы 1-4): {len(df_train):,} строк")
    print(f"Тестовая выборка  (насос 5):    {len(df_test):,} строк")

    # 3. Формирование матриц X и Y через явный контракт FEATURE_COLS
    # Это гарантирует идентичный набор признаков при обучении и инференсе
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"В датасете отсутствуют признаки: {missing[:5]}...")

    X_train = df_train[FEATURE_COLS]
    y_train = df_train['target']
    X_test = df_test[FEATURE_COLS]
    y_test = df_test['target']

    print(f"Признаков для ML: {len(FEATURE_COLS)}")

    # 4. Веса классов для компенсации дисбаланса (Critical — редкий класс)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # 5. Обучение моделей
    print("\nОбучение моделей запущено...")

    # Logistic Regression — линейный baseline
    # solver='saga': поддерживает многоклассовую задачу и большие датасеты
    lr_model = LogisticRegression(
        class_weight='balanced', max_iter=500,
        solver='saga', random_state=42
    )
    lr_model.fit(X_train, y_train)

    # Random Forest — ансамбль бэггинга
    # n_estimators=300: достаточно для стабилизации дисперсии на 320k строк
    rf_model = RandomForestClassifier(
        n_estimators=300, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # XGBoost — ансамбль бустинга с взвешиванием через sample_weight
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='aucpr',
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

    # 6. Оценка
    metrics = [
        evaluate_model("Logistic Regression (Baseline)",   "LogReg",  lr_model,  X_test, y_test),
        evaluate_model("Random Forest (Bagging Ensemble)", "RF",      rf_model,  X_test, y_test),
        evaluate_model("XGBoost (Boosting Ensemble)",      "XGBoost", xgb_model, X_test, y_test),
    ]

    # 7. Три технических графика
    plot_all(metrics, y_test, save_graps_dir)

    # 8. Демонстрация AlarmManager (Alarm Shelving)
    print(f"\n{'='*50}")
    print("ДЕМОНСТРАЦИЯ ЛОГИКИ ФИЛЬТРАЦИИ (Alarm Shelving)")
    print(f"{'='*50}")

    manager = AlarmManager(xgb_model)
    critical_indices = y_test[y_test == 2].index

    if len(critical_indices) > 0:
        idx = critical_indices[-1]
        critical_features = pd.DataFrame([X_test.loc[idx].values], columns=FEATURE_COLS)

        pred_a = manager.predict_with_context(critical_features, raw_state=2)
        print(f"\nСценарий А: State=2 (штатная работа), признаки аварийные.")
        print(f"  → AlarmManager: Класс {pred_a} — Авария. Сигнал передан на дашборд.")

        pred_b = manager.predict_with_context(critical_features, raw_state=1)
        print(f"\nСценарий Б: State=1 (запуск), те же аварийные признаки.")
        print(f"  → AlarmManager: Класс {pred_b} — Норма. "
              f"Сигнал подавлен (Alarm Shelving: пусковые токи в норме).")
    else:
        print("В тестовой выборке аварий не найдено.")

    # 9. Сохранение моделей для инференса и XAI
    models_dir = os.path.join(project_root, 'models', 'severity')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path_lr = os.path.join(models_dir, 'severity_lr_model.joblib')
    joblib.dump(lr_model, model_path_lr)
    model_path_rf = os.path.join(models_dir, 'severity_rf_model.joblib')
    joblib.dump(rf_model, model_path_rf)
    model_path_xgb = os.path.join(models_dir, 'severity_xgboost_model.joblib')
    joblib.dump(xgb_model, model_path_xgb)
    print(f"\n{'='*50}")
    print(f"Модели успешно сохранены в:\n{models_dir}")

    # 10. Анализ recall по типам отказа
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'industrial_pumps_dataset.csv')
    analyze_fault_recall(xgb_model, df_test, FEATURE_COLS, save_graps_dir,  # type: ignore
                         save_tables_dir, raw_data_path=raw_data_path)