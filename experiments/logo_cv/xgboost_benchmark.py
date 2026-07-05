"""
Раздел оценки робастности модели
================================
experiments/logo_cv/xgboost_benchmark.py

В данном разделе реализуется подход "Leave-One-Group-Out Cross-Validation
(LOGO CV), в рамках которого модель XGBoost обучается на выборке из 4-х
насосов и проверяется на 5-ом, каждый раз оставляя отличный насос на тест, 
таким образом, доказываются высокие обобщающие свойства модели и подтверждается
высокий результат.
"""

import os
import sys
from typing import Optional, cast

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from sklearn.metrics import (classification_report, confusion_matrix)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.use('Agg')

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config.settings import PUMPS, WINDOW_SIZES
from src.data.data_preprocessor import DataPreprocessor
_preprocessor = DataPreprocessor(window_sizes=WINDOW_SIZES)
FEATURE_COLS = _preprocessor.FEATURE_COLS


def xgboost_logo_cv(df: pd.DataFrame, feature_cols: list, pumps: list) -> list:
    """
    Leave-One-Group-Out CV по агрегатам: каждый насос по очереди - тест,
    остальные четыре - обучение. Обучает XGBoost на каждом фолде и возвращает
    список метрик (по одному словарю evaluate_model на фолд). Доказывает, что
    результат не привязан к конкретному тестовому насосу.
    """

    all_folds_metrics = []
    # Главный цикл кросс-валидации
    for fold_idx, test_pump in enumerate(pumps, 1):
        print(f"\n{'='*60}")
        print(f"Фолд {fold_idx}/5 | Тестовый насос: {test_pump}")
        print(f"{'='*60}")
        
        # Списки для сплита: 1 насос в тест, остальные 4 в трэйн
        train_pumps = [p for p in pumps if p != test_pump]
        print(f"Обучение на: {train_pumps}")
        
        df_train = df[df['pump_id'].isin(train_pumps)].copy()
        df_test = df[df['pump_id'].isin([test_pump])].copy()
        
        # Формирование матриц X и Y через явный контракт FEATURE_COLS
        # Это гарантирует идентичный набор признаков при обучении и инференсе
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"В датасете отсутствуют признаки: {missing[:5]}...")
        
        X_train = df_train[feature_cols]
        y_train = df_train['target']
        X_test = df_test[feature_cols]
        y_test = df_test['target']
        
        # Балансировка весов для текущего фолда
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        # Инициализация и обучение XGBoost
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
        
        # Оценка на текущем тестовом насосе
        metrics = evaluate_model("XGBoost (Boosting Ensemble)", 
                                 f"XGBoost_{test_pump}", xgb_model, 
                                 X_test, y_test)
        
        # Сохранение результатов итерации
        all_folds_metrics.append(metrics)

    return all_folds_metrics

# Данная функция дублирует evaluate_model() из
# src/ml/severity_classifier_pipeline.py - осознанный дубль, чтобы эксперимент
# не зависел от рефакторинга боевого пайплайна (не нужен общий модуль).
def evaluate_model(name: str, short_name: str, model, 
                   X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Вычисляет и выводит полный набор метрик для одной модели.

    Метрики и их обоснование для задачи предиктивного обслуживания:

    Confusion Matrix - показывает конкретную цену каждого типа ошибки.
        В нефтегазовой отрасли False Negative (пропущенная авария, ошибка 2 рода) 
        несравнимо дороже False Positive (ложная тревога, ошибка 1 рода), 
        поэтому матрица обязательна.

    F1-Macro - среднее F1 по всем классам без учёта их размера.
        Предотвращает "утопание" редких классов (Авария - ~0.2% строк)
        в weighted-среднем, которое доминируется классом Норма.

    Recall класса "Авария" (KPI системы) - доля реальных аварий,
        которую система обнаружила. Это ключевой бизнес-показатель:
        Recall=0.95 означает, что 95 из 100 аварий будут пойманы заблаговременно.

    PR-AUC - площадь под кривой Precision-Recall.
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
    report = cast(dict, classification_report(y_test, y_pred,
                                              target_names=target_names, output_dict=True))
    print("\nОтчет по классификации (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # PR-AUC
    y_test_bin = cast(np.ndarray, label_binarize(y_test, classes=[0, 1, 2]))
    pr_auc_macro = average_precision_score(y_test_bin, y_proba, average="macro")
    pr_auc_critical = average_precision_score(y_test_bin[:, 2], y_proba[:, 2])

    # Recall класса "Авария" - явный KPI
    recall_critical = report['2: Авария']['recall']

    print(f"PR-AUC (Macro Average):    {pr_auc_macro:.4f}")
    print(f"PR-AUC (Класс 'Авария'):   {pr_auc_critical:.4f}")
    print(f"Recall (Класс 'Авария'):   {recall_critical:.4f}")

    return {
        'name': name,
        'label': short_name,                            # имя для графиков
        'f1_macro': report['macro avg']['f1-score'],
        'f1_warning': report['1: Warning']['f1-score'],
        'f1_critical': report['2: Авария']['f1-score'],
        'recall_critical': recall_critical,
        'pr_auc_macro': pr_auc_macro,
        'pr_auc_critical': pr_auc_critical,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'cm': confusion_matrix(y_test, y_pred),
    }

# Академическая серо-голубая палитра 
LIGHT_BG = '#FFFFFF'
GRID_CLR = '#E5E5E5'
TEXT_CLR = '#2A2A2A'
SPINE_CLR = '#BFBFBF'
 
# Три оттенка одной сине-серой гаммы (тёмный → светлый), приятные для глаза
C_F1 = '#33506E'    # тёмно-сине-серый
C_PR = '#6B8CAE'    # средний голубовато-серый
C_REC = '#A9C0D6'   # светлый голубой
CM_CMAP = 'Blues'     # монохромная синяя для heatmap
 
CLASS_LABELS = ['Норма', 'Предупр.', 'Авария']
 
def comparison_plot(cv_metrics: list, save_dir: Optional[str] = None):
    """
    Дашборд LOGO-CV: верхний ряд (метрики + качество), нижний ряд (матрицы по фолдам).
 
    Args:
        cv_metrics: список словарей от evaluate_model, по одному на фолд.
        save_dir:   директория сохранения; по умолчанию artifacts/graphs.
    
    В графиках исключены показания MNHV_005 для отсутствия дублирования в 
    визуализации для диссертации. Однако, среднее значение считается по всем
    насосам, включая MNHV_005.
    """

    # ЧЕСТНЫЕ средние по ВСЕМ 5 насосам для Панели Б
    f1_all = [m['f1_macro'] for m in cv_metrics]
    pr_auc_all = [m['pr_auc_critical'] for m in cv_metrics]
    recall_all = [m['recall_critical'] for m in cv_metrics]
    
    # float(): np.mean/np.std возвращают floating[Any]; приводим к python float,
    # иначе ax.text(y=...) не принимает numpy-скаляр по типам.
    means = [float(np.mean(f1_all)), float(np.mean(pr_auc_all)), float(np.mean(recall_all))]
    stds = [float(np.std(f1_all)), float(np.std(pr_auc_all)), float(np.std(recall_all))]
    
    # Фильтр данных для Панели А и нижнего ряда матриц (без MNHV_005)
    cv_plot = [m for m in cv_metrics if 'MNHV_005' not in m['label']]
    
    # Формирование списков только для отрисовки детализации (длина = 4)
    pumps = [m['label'].replace('XGBoost_', '') for m in cv_plot]
    f1_plot = [m['f1_macro'] for m in cv_plot]
    pr_auc_plot = [m['pr_auc_critical'] for m in cv_plot]
    recall_plot = [m['recall_critical'] for m in cv_plot]
    
    n = len(cv_plot) # Теперь n четко равно 4
 
    # Сетка: 2 ряда
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor(LIGHT_BG)
    gs = GridSpec(2, n, figure=fig, height_ratios=[1, 0.85],
                  hspace=0.42, wspace=0.30)
 
    # Верхний ряд: метрики занимают первые (n-1) колонок, качество - последнюю
    ax_metrics = fig.add_subplot(gs[0, :n-1])
    ax_quality = fig.add_subplot(gs[0, n-1])
 
    fig.suptitle('Оценка устойчивости XGBoost на новом оборудовании (Leave-One-Group-Out CV)',
                 color=TEXT_CLR, fontsize=16, fontweight='bold', y=0.98)
 
    # Панель A: метрики по фолдам 
    ax_metrics.set_facecolor(LIGHT_BG)
    x = np.arange(n)
    w = 0.25
    r1 = ax_metrics.bar(x - w, f1_plot, w, label='F1 Macro', 
                        color=C_F1, zorder=3)
    r2 = ax_metrics.bar(x, pr_auc_plot, w, label='PR-AUC (Авария)', 
                        color=C_PR, zorder=3)
    r3 = ax_metrics.bar(x + w, recall_plot, w, label='Recall (Авария)', 
                        color=C_REC, zorder=3)
 
    ax_metrics.set_title('Разброс метрик по фолдам (каждый насос тестовый по очереди)',
                         color=TEXT_CLR, fontsize=14, fontweight='bold', pad=10)
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(pumps, color=TEXT_CLR, fontsize=12)
    ax_metrics.set_ylim(0.6, 1.1)
    ax_metrics.set_ylabel('Значение метрики', color=TEXT_CLR, fontsize=12)
    ax_metrics.yaxis.grid(True, linestyle='--', alpha=0.7, color=GRID_CLR, zorder=0)
    ax_metrics.legend(loc='lower left', framealpha=1.0, facecolor=LIGHT_BG,
                      edgecolor=SPINE_CLR, fontsize=9)
    for rects in (r1, r2, r3):
        for rect in rects:
            h = rect.get_height()
            ax_metrics.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                                xytext=(0, 4), textcoords="offset points",
                                ha='center', va='bottom', fontsize=10,
                                color=TEXT_CLR, fontweight='bold', rotation=90)
 
    # Панель B: итоговое качество (среднее ± std) 
    ax_quality.set_facecolor(LIGHT_BG)
    qlabels = ['F1\nMacro', 'PR-AUC\n(Авар.)', 'Recall\n(Авар.)']
    bars = ax_quality.bar(range(3), means, width=0.62, yerr=stds, capsize=6,
                          color=[C_F1, C_PR, C_REC], alpha=0.95,
                          ecolor='#444444', zorder=3)
    ax_quality.set_title('Итоговое качество\n(mean ± std по фолдам)',
                         color=TEXT_CLR, fontsize=14, fontweight='bold', pad=10)
    ax_quality.set_xticks(range(3))
    ax_quality.set_xticklabels(qlabels, fontsize=12)
    ax_quality.set_ylim(0.6, 1.1)
    ax_quality.yaxis.grid(True, linestyle='--', alpha=0.7, color=GRID_CLR, zorder=0)
    for i, bar in enumerate(bars):
        ax_quality.text(bar.get_x() + bar.get_width()/2, means[i] + stds[i] + 0.015,
                        f'{means[i]:.3f}\n± {stds[i]:.3f}', ha='center', va='bottom',
                        color=TEXT_CLR, fontweight='bold', fontsize=10)
 
    # Нижний ряд: матрица ошибок для каждого фолда отдельно 
    for fold_idx, m in enumerate(cv_plot):
        ax = fig.add_subplot(gs[1, fold_idx])
        cm = np.array(m['cm'], dtype=float)
 
        # Приведение к 3×3 на случай отсутствующих классов в фолде
        if cm.shape != (3, 3):
            full = np.zeros((3, 3))
            full[:cm.shape[0], :cm.shape[1]] = cm
            cm = full
 
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = np.divide(cm, row_sums, where=row_sums != 0)
 
        ax.imshow(cm_norm, cmap=CM_CMAP, vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'{pumps[fold_idx]}', color=TEXT_CLR, fontsize=12,
                     fontweight='bold', pad=6)
        ax.set_xticks(range(3))
        ax.set_xticklabels(CLASS_LABELS, fontsize=10, rotation=30)
        ax.set_yticks(range(3))
        # Подписи оси Y только у крайней левой матрицы - экономия места
        if fold_idx == 0:
            ax.set_yticklabels(CLASS_LABELS, fontsize=10)
            ax.set_ylabel('Истинно', color=TEXT_CLR, fontsize=12)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('Предсказано', color=TEXT_CLR, fontsize=12)
 
        for i in range(3):
            for j in range(3):
                frac = cm_norm[i, j]
                cnt = int(cm[i, j])
                clr = 'white' if frac > 0.5 else TEXT_CLR
                ax.text(j, i, f'{frac:.0%}\n{cnt}', ha='center', va='center',
                        fontsize=8, fontweight='bold', color=clr)
        # Жёлтая рамка на диагонали - верные предсказания
        for k in range(3):
            ax.add_patch(Rectangle((k-0.5, k-0.5), 1, 1, fill=False,
                                   edgecolor='#E8A03D', lw=2.0))
 
    # Подпись нижнего ряда
    fig.text(0.5, 0.46, 'Матрицы ошибок по фолдам (нормировка по строкам, в ячейке - доля и значение)',
             ha='center', color=TEXT_CLR, fontsize=14, fontweight='bold')
 
    # Единая стилизация рамок верхних панелей
    for ax in (ax_metrics, ax_quality):
        ax.tick_params(colors=TEXT_CLR)
        for sp in ax.spines.values():
            sp.set_edgecolor(SPINE_CLR)
 
    if save_dir is None:
        save_dir = os.path.join(_PROJECT_ROOT, 'artifacts', 'graphs')
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'ML_plot5_logo_cv_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"\nГрафик LOGO-CV сохранён: {path}")
 
# Точка входа

if __name__ == "__main__":
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))))
    processed_data_path = os.path.join(project_root, 'data', 'processed', 
                                       'preprocessed_pumps_dataset.csv')
    output_dir = os.path.join(project_root, 'artifacts', 'graphs')

    # Загрузка данных
    print("Загрузка обработанных данных...")
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(
            f"Файл не найден: {processed_data_path}. Сначала запустите data_preprocessor.py"
        )
    df = pd.read_csv(processed_data_path)

    print(f"\nЗапуск Leave-One-Group-Out кросс-валидации для {len(PUMPS)} агрегатов...")

    # Моделирование кросс-валидации с изоляцией по группе 
    # Leave-One-Group-Out Cross-Validation (LOGO-CV)
    cv_metrics = xgboost_logo_cv(df, FEATURE_COLS, PUMPS)

    # Формирование графика
    comparison_plot(cv_metrics, output_dir)

    print(f"\n{'*'*51}")
    print("ИТОГИ LEAVE-ONE-GROUP-OUT КРОСС-ВАЛИДАЦИИ (XGBoost)")
    print(f"{'*'*51}")
    
    f1_scores = [m['f1_macro'] for m in cv_metrics]
    pr_aucs = [m['pr_auc_critical'] for m in cv_metrics]
    recalls = [m['recall_critical'] for m in cv_metrics]
    
    # Вывод среднего значения и стандартного отклонения (стабильность модели)
    print(f"Средний F1 (Macro):      {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Средний PR-AUC (Авария): {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")
    print(f"Средний Recall (Авария): {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")