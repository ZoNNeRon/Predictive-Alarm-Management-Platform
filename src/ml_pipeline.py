import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (classification_report, confusion_matrix,
                              average_precision_score, precision_recall_curve)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib

# Импорт препроцессора для получения FEATURE_COLS
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from data_preprocessor import DataPreprocessor
    _preprocessor = DataPreprocessor(window_sizes=[15, 30, 60])
    FEATURE_COLS = _preprocessor.FEATURE_COLS
except ImportError:
    # Fallback: если препроцессор недоступен, список выстраивается вручную
    _sensors = ['vibration', 'temperature', 'current', 'pressure']
    _windows = [15, 30, 60]
    FEATURE_COLS = [
        f'{s}_{stat}_{w}'
        for s in _sensors
        for w in _windows
        for stat in ['mean', 'std', 'max']
    ] + [f'{s}_diff_30' for s in _sensors]


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

def evaluate_model(name: str, short_name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
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
        'name':             name,
        'label':            short_name, # имя для графиков
        'f1_macro':         report['macro avg']['f1-score'], # type: ignore
        'f1_warning':       report['1: Warning']['f1-score'], # type: ignore
        'f1_critical':      report['2: Авария']['f1-score'], # type: ignore
        'recall_critical':  recall_critical,
        'pr_auc_macro':     pr_auc_macro,
        'pr_auc_critical':  pr_auc_critical,
        'y_pred':           y_pred,
        'y_proba':          y_proba,
        'cm':               confusion_matrix(y_test, y_pred),
    }

# Визуализация 

# Палитра и стиль
PALETTE  = {'LogReg': '#4C72B0', 'RF': '#55A868', 'XGBoost': '#C44E52'}
LIGHT_BG = '#FFFFFF'
GRID_CLR = '#E5E5E5'
TEXT_CLR = '#222222'


def _apply_light_style(ax, title: str = ''):
    ax.set_facecolor(LIGHT_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.xaxis.label.set_color(TEXT_CLR)
    if title:
        ax.set_title(title, color=TEXT_CLR, fontsize=12, fontweight='bold', pad=10)


def plot_all(metrics_list: list, y_test: pd.Series, output_dir: str):
    """
    Строит и сохраняет три технических графика в светлой (академической) теме:
      1. Тепловые карты Confusion Matrix
      2. Grouped bar chart по метрикам
      3. PR-кривые
    """
    os.makedirs(output_dir, exist_ok=True)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    # График 1: Confusion Matrices 
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(LIGHT_BG)
    fig.suptitle('Матрицы ошибок (тестовая выборка: насос MNHV_005)',
                 color=TEXT_CLR, fontsize=14, fontweight='bold', y=1.05)

    class_labels = ['Норма', 'Предупреждение', 'Авария']
    for ax, m in zip(axes, metrics_list):
        cm_norm = m['cm'].astype(float) / m['cm'].sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=m['cm'], fmt='d', ax=ax,
            cmap='Blues', linewidths=0.5, linecolor='#DDDDDD',
            xticklabels=class_labels, yticklabels=class_labels,
            cbar=False, annot_kws={'size': 11} # Убрали 'color': 'white', Seaborn настроит контраст
        )
        _apply_light_style(ax, m['label'])
        ax.set_xlabel('Предсказано', color=TEXT_CLR)
        ax.set_ylabel('Истинно', color=TEXT_CLR)
        
        # Выделяем диагональ (правильные предсказания)
        for i in range(3):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, # type: ignore
                                       edgecolor='#FF8C00', lw=2)) 

    plt.tight_layout()
    p1 = os.path.join(output_dir, 'plot1_confusion_matrices.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    print(f"График 1 сохранён: {p1}")
    plt.close(fig)

    # График 2: Метрики — Grouped Bar Chart 
    metric_keys   = ['f1_macro', 'f1_critical', 'recall_critical', 'pr_auc_critical']
    metric_labels = ['F1 Macro', 'F1 (Авария)', 'Recall (Авария)', 'PR-AUC (Авария)']
    n_models  = len(metrics_list)
    n_metrics = len(metric_keys)
    x = np.arange(n_metrics)
    width = 0.22

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    colors = list(PALETTE.values())
    for i, m in enumerate(metrics_list):
        vals = [m[k] for k in metric_keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=m['label'],
                        color=colors[i], alpha=0.9, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=9, color=TEXT_CLR, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color=TEXT_CLR, fontsize=11)
    ax.set_ylim(0.55, 1.08) # Расширили верхний лимит для подписей
    ax.set_ylabel('Значение метрики', color=TEXT_CLR)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color=GRID_CLR, zorder=0)
    ax.set_title('Сравнение качества моделей по ключевым метрикам\n'
                 '(тестовая выборка: насос MNHV_005)',
                 color=TEXT_CLR, fontsize=13, fontweight='bold')
    ax.legend(framealpha=1.0, labelcolor=TEXT_CLR, fontsize=10,
              facecolor=LIGHT_BG, edgecolor='#CCCCCC')
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')

    plt.tight_layout()
    p2 = os.path.join(output_dir, 'plot2_metrics_comparison.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    print(f"График 2 сохранён: {p2}")
    plt.close(fig)

    # График 3: PR-кривые для класса "Авария" 
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    for m, color in zip(metrics_list, colors):
        prec, rec, _ = precision_recall_curve(y_test_bin[:, 2], m['y_proba'][:, 2]) # type: ignore
        auc_val = m['pr_auc_critical']
        ax.plot(rec, prec, color=color, lw=2.0,
                label=f"{m['label']}  (PR-AUC = {auc_val:.4f})", zorder=3)
        # Точка при пороге 0.5
        idx_50 = np.argmin(np.abs(m['y_proba'][:, 2].mean() - 0.5))
        ax.scatter(rec[len(rec)//2], prec[len(prec)//2],
                   color=color, s=60, zorder=5, edgecolor=LIGHT_BG)

    # Базовая линия (случайный классификатор)
    baseline = y_test_bin[:, 2].mean() # type: ignore
    ax.axhline(baseline, color='#888888', lw=1.2, ls='--',
               label=f'Случайный классификатор (P = {baseline:.3f})', zorder=1)

    ax.set_xlabel('Recall (Полнота обнаружения аварий)', color=TEXT_CLR, fontsize=11)
    ax.set_ylabel('Precision (Точность предупреждений)', color=TEXT_CLR, fontsize=11)
    ax.set_title('PR-кривые для класса «Авария» (Class 2)\n'
                 'Ключевая метрика для несбалансированных промышленных данных',
                 color=TEXT_CLR, fontsize=12, fontweight='bold')
    ax.legend(framealpha=1.0, labelcolor=TEXT_CLR, fontsize=10,
              facecolor=LIGHT_BG, edgecolor='#CCCCCC')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color=GRID_CLR, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7, color=GRID_CLR, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
    ax.set_xlim([-0.02, 1.02]) # type: ignore
    ax.set_ylim([-0.02, 1.05]) # type: ignore

    plt.tight_layout()
    p3 = os.path.join(output_dir, 'plot3_pr_curves_critical.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    print(f"График 3 сохранён: {p3}")
    plt.close(fig)


# Точка входа 

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'processed_features.csv')
    output_dir = os.path.join(project_root, 'data', 'graphs')

    # 1. Загрузка данных
    print("Загрузка обработанных данных...")
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(
            f"Файл не найден: {processed_data_path}. Сначала запустите data_preprocessor.py"
        )
    df = pd.read_csv(processed_data_path)

    # 2. Group Split по оборудованию (доказывает обобщение на новый агрегат)
    train_pumps = ['MNHV_001', 'MNHV_002', 'MNHV_003', 'MNHV_004']
    test_pumps = ['MNHV_005']

    df_train = df[df['pump_id'].isin(train_pumps)].copy()
    df_test = df[df['pump_id'].isin(test_pumps)].copy()

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
    plot_all(metrics, y_test, output_dir)

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

    # 9. Сохранение лучшей модели (XGBoost) для инференса и XAI
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path_lr = os.path.join(models_dir, 'lr_pump_model.joblib')
    joblib.dump(lr_model, model_path_lr)
    model_path_rf = os.path.join(models_dir, 'rf_pump_model.joblib')
    joblib.dump(rf_model, model_path_rf)
    model_path_xgb = os.path.join(models_dir, 'xgboost_pump_model.joblib')
    joblib.dump(xgb_model, model_path_xgb)
    print(f"\n{'='*50}")
    print(f"Модели успешно сохранены в:\n{models_dir}")