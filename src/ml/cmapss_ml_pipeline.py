"""
ML-пайплайн валидации на реальных данных NASA C-MAPSS
=====================================================
src/ml/cmapss_ml_pipeline.py

Обучение и оценка модели ТЯЖЕСТИ (Норма / Предупреждение / Авария) на
реальных run-to-failure данных турбовентиляторных двигателей - доказательство
переносимости аналитического ядра платформы на другой класс оборудования.

Переиспользуется без изменений:
  - подготовка данных и rolling-признаки: src.data.cmapss_dataset
    (наследник насосного DataPreprocessor);
  - весь блок метрик: evaluate_model() из severity_classifier_pipeline
    (Confusion Matrix, F1-Macro, Recall(Авария) как KPI, PR-AUC);
  - тройка моделей и гиперпараметры насосного пайплайна:
    LogisticRegression (baseline) / RandomForest / XGBoost,
    балансировка class_weight='balanced' + sample_weight.

Отличие от насосного пайплайна (задокументированное): LR обёрнута в
StandardScaler - сырые величины сенсоров C-MAPSS различаются на 3 порядка
(s9 ~9000, s21 ~23); RF/XGBoost масштабо-инвариантны и идут как есть.

ДВА РЕЖИМА ОЦЕНКИ (--split):
  official - обучение на официальном train, оценка на официальном test
             с истинным RUL из RUL_FD00X.txt. Test-траектории обрезаны
             до отказа по замыслу бенчмарка, поэтому классы Предупреждение/
             Авария в test редки (~6%/~1%) - реалистичный продакшн-профиль,
             под который и выбран PR-AUC. 
  holdout  - контрольный группированный сплит по ДВИГАТЕЛЯМ внутри train
             (как Group Split насосного пайплайна): выделенные двигатели
             дожиты до отказа, баланс классов test повторяет train.
             Подтверждает, что метрики official не артефакт дисбаланса.

Выходы (философия «сводной подтверждённой картины» - по умолчанию строится
ТОЛЬКО сводный слой, пер-сабсетная детализация по флагу --detail-plots):
  - models/cmapss/cmapss_{subset}_{lr,rf,xgboost}_model.joblib
    (xgboost official - опорная модель для XAI);
  - artifacts/tables/cmapss_ml_summary.csv - сводка сабсет x сплит x модель
    (обновляется слиянием: прогон сабсета заменяет только свои строки);
  - artifacts/tables/cmapss_pr_curves.csv - точки PR-кривых класса «Авария»
    на единой сетке recall (тот же принцип слияния);
  - artifacts/tables/cmapss_confusion.csv - счётчики ячеек матриц ошибок
    по разрезу сабсет x сплит x модель (тот же принцип слияния);
  - artifacts/tables/cmapss_{subset}_lead_time.csv - упреждение обнаружения
    по двигателям holdout-теста (XGBoost), считается всегда;
  - artifacts/graphs/cmapss_summary_plot1-3_*.png + plot6 - сравнение моделей
    (mean ± std по сабсетам), усреднённые PR-кривые, сводный lead time и
    матрицы ошибок (сумма счётчиков по сабсетам, сплит x модель);
  - [--detail-plots] artifacts/graphs/cmapss_{subset}_plot1-4_*.png -
    пер-сабсетные матрицы ошибок, метрики по сплитам, PR-кривые, lead time
    (материал приложения; включается автоматически при прогоне 1 сабсета).

Запуск (из корня репозитория):
  python -m src.ml.cmapss_ml_pipeline                    # итоговый прогон
  python -m src.ml.cmapss_ml_pipeline FD002 --split official
  python -m src.ml.cmapss_ml_pipeline all --detail-plots
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple, cast

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import warnings
warnings.filterwarnings('ignore')

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings.settings_cmapss import (
    CMAPSS_SUBSETS, CMAPSS_SUBSET_INFO, CMAPSS_MODELS_SUBDIR,
    CMAPSS_TABLES_PREFIX, CMAPSS_RANDOM_STATE)
from sklearn.metrics import precision_recall_curve

from src.data.cmapss_dataset import prepare_subset, class_balance
from src.ml.severity_classifier_pipeline import evaluate_model
from src.visualisation.cmapss_visualisation import (
    plot_confusion_matrices, plot_metrics_by_split, plot_pr_curves,
    plot_lead_time, plot_summary_generalisation, plot_summary_pr_curves,
    plot_summary_confusion, plot_summary_lead_time)

MODELS_DIR = os.path.join(_PROJECT_ROOT, *CMAPSS_MODELS_SUBDIR)
TABLES_DIR = os.path.join(_PROJECT_ROOT, 'artifacts', 'tables')
GRAPHS_DIR = os.path.join(_PROJECT_ROOT, 'artifacts', 'graphs')

HOLDOUT_TEST_FRAC = 0.2  # доля двигателей train, уходящая в контрольный тест

# Единая сетка recall для усреднения PR-кривых между сабсетами: кривые
# разных тестов имеют разное число точек, усреднять можно только после
# интерполяции на общий грид (вход сводного графика 2)
PR_RECALL_GRID = np.linspace(0.0, 1.0, 201)

MODEL_TITLES = {
    'lr': ('Logistic Regression (Baseline)', 'LogReg'),
    'rf': ('Random Forest (Bagging Ensemble)', 'RF'),
    'xgboost': ('XGBoost (Boosting Ensemble)', 'XGBoost'),
}


# Сплиты

def holdout_split(train_df: pd.DataFrame, test_frac: float = HOLDOUT_TEST_FRAC,
                  seed: int = CMAPSS_RANDOM_STATE
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Группированный сплит по двигателям: часть агрегатов - целиком в тест.

    Строки одного двигателя не разрываются между train и test (тот же
    принцип, что Group Split насосного пайплайна) - оценивается обобщение
    на unseen-агрегат, а не запоминание траектории.
    """

    engines = np.array(sorted(train_df['pump_id'].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(engines)
    n_test = max(1, round(len(engines) * test_frac))
    test_engines = set(engines[:n_test])
    tr = cast(pd.DataFrame, train_df[~train_df['pump_id'].isin(test_engines)])
    te = cast(pd.DataFrame, train_df[train_df['pump_id'].isin(test_engines)])
    return tr.copy(), te.copy()


# Обучение

def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    """
    Тройка моделей с гиперпараметрами насосного пайплайна.

    LR - в связке со StandardScaler (масштабы сенсоров C-MAPSS растянуты
    на 3 порядка); RF/XGBoost масштабо-инвариантны.
    """

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    lr_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced', max_iter=500,
                           solver='saga', random_state=CMAPSS_RANDOM_STATE))
    lr_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=300, class_weight='balanced',
        random_state=CMAPSS_RANDOM_STATE, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, eval_metric='aucpr',
        n_estimators=300, learning_rate=0.1, max_depth=6,
        random_state=CMAPSS_RANDOM_STATE, n_jobs=-1)
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

    return {'lr': lr_model, 'rf': rf_model, 'xgboost': xgb_model}


# Упреждение обнаружения (lead time)

def compute_lead_times(model, test_df: pd.DataFrame,
                       feature_cols: list) -> pd.DataFrame:
    """За сколько циклов до отказа модель впервые подняла каждую стадию.

    Считается по двигателям, дожитым до отказа (holdout-тест): для каждого
    агрегата берётся первый цикл с прогнозом >=1 (Предупреждение) и первый
    с прогнозом ==2 (Авария); lead = истинный RUL в этот момент. NaN - стадия
    не поднята ни разу (пропуск двигателя).
    """

    rows = []
    for pid, g in test_df.sort_values(['pump_id', 'timestamp']).groupby('pump_id'):
        X = cast(pd.DataFrame, g[feature_cols])
        preds = np.asarray(model.predict(X))
        rul = g['RUL'].to_numpy()
        warn_idx = np.flatnonzero(preds >= 1)
        crit_idx = np.flatnonzero(preds == 2)
        rows.append({
            'pump_id': pid,
            'lead_warning': float(rul[warn_idx[0]]) if len(warn_idx) else np.nan,
            'lead_critical': float(rul[crit_idx[0]]) if len(crit_idx) else np.nan,
            'trajectory_cycles': int(len(g)),
        })
    return pd.DataFrame(rows)


# Прогон одного сабсета

def run_subset(subset: str, split_mode: str, save_models: bool = True,
               detail_plots: bool = False
               ) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Полный прогон сабсета: подготовка -> обучение -> оценка evaluate_model.

    split_mode: 'official' | 'holdout' | 'both'.
    Возвращает (строки сводной таблицы, точки PR-кривых, счётчики матриц
    ошибок) - все по разрезу сабсет x сплит x модель; PR-кривые
    интерполированы на PR_RECALL_GRID (вход сводного графика 2), счётчики
    матриц - вход сводного графика 6 (суммируются по сабсетам). При
    save_models сохраняет модели official-сплита в models/cmapss/
    (xgboost - опорная для XAI-этапа). Lead-time CSV на holdout считается
    ВСЕГДА (вход сводного графика 3); пер-сабсетные PNG (матрицы ошибок,
    PR-кривые, lead time) - только при detail_plots.
    """

    info = CMAPSS_SUBSET_INFO[subset]
    print(f"\n{'='*62}\nСабсет {info['label']}\n{'='*62}")
    train_df, test_df, pre = prepare_subset(subset)
    fc = pre.FEATURE_COLS

    splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    if split_mode in ('official', 'both'):
        splits['official'] = (train_df, test_df)
    if split_mode in ('holdout', 'both'):
        splits['holdout'] = holdout_split(train_df)

    rows: List[dict] = []
    curve_rows: List[dict] = []
    cm_rows: List[dict] = []
    for split_name, (tr, te) in splits.items():
        print(f"\n--- Сплит: {split_name} | обучение {len(tr):,} строк "
              f"({tr['pump_id'].nunique()} двиг.), тест {len(te):,} строк "
              f"({te['pump_id'].nunique()} двиг.)")
        print(f"    Баланс train: {class_balance(tr)}")
        print(f"    Баланс test:  {class_balance(te)}")

        # cast: pandas-stubs выводят df[list] как Series - здесь это DataFrame
        X_train = cast(pd.DataFrame, tr[fc])
        X_test = cast(pd.DataFrame, te[fc])
        y_train, y_test = tr['target'], te['target']
        models = train_models(X_train, y_train)

        metrics_list: List[dict] = []
        for key, model in models.items():
            title, short = MODEL_TITLES[key]
            m = evaluate_model(f"{title} [{subset}/{split_name}]",
                               short, model, X_test, y_test)
            metrics_list.append(m)
            rows.append({
                'subset': subset,
                'split': split_name,
                'model': short,
                'f1_macro': round(m['f1_macro'], 4),
                'f1_warning': round(m['f1_warning'], 4),
                'f1_critical': round(m['f1_critical'], 4),
                'recall_critical': round(m['recall_critical'], 4),
                'pr_auc_critical': round(m['pr_auc_critical'], 4),
                'pr_auc_macro': round(m['pr_auc_macro'], 4),
            })

        # Счётчики ячеек матриц ошибок - материал сводного графика 6:
        # матрицы сабсетов агрегируются СУММОЙ абсолютных счётчиков
        # (каждая тестовая строка учтена один раз, «средних матриц» нет)
        for m in metrics_list:
            cm = np.asarray(m['cm'])
            cm_rows.extend({
                'subset': subset,
                'split': split_name,
                'model': m['label'],
                'true_class': i,
                'pred_class': j,
                'count': int(cm[i, j]),
            } for i in range(cm.shape[0]) for j in range(cm.shape[1]))

        # Точки PR-кривых класса «Авария» на единой сетке recall - материал
        # усреднения между сабсетами (сводный график 2); кривые разных тестов
        # имеют разное число порогов, поэтому интерполяция обязательна
        y_bin = (np.asarray(y_test) == 2).astype(int)
        baseline = float(y_bin.mean())
        for m in metrics_list:
            prec, rec, _ = precision_recall_curve(y_bin, m['y_proba'][:, 2])
            # precision_recall_curve отдаёт recall по убыванию,
            # np.interp требует возрастающий xp
            prec_grid = np.interp(PR_RECALL_GRID, rec[::-1], prec[::-1])
            curve_rows.extend({
                'subset': subset,
                'split': split_name,
                'model': m['label'],
                'recall': round(float(r), 4),
                'precision': round(float(p), 6),
                'baseline': round(baseline, 6),
            } for r, p in zip(PR_RECALL_GRID, prec_grid))

        if detail_plots:
            os.makedirs(GRAPHS_DIR, exist_ok=True)
            plot_confusion_matrices(metrics_list, subset, split_name, GRAPHS_DIR)
            plot_pr_curves(metrics_list, y_test, subset, split_name, GRAPHS_DIR)

        if split_name == 'holdout':
            # lead time честен только на дожитых до отказа траекториях;
            # CSV считается ВСЕГДА - он питает сводный график 3
            lead_df = compute_lead_times(models['xgboost'], te, fc)
            lead_csv = os.path.join(
                TABLES_DIR,
                f'{CMAPSS_TABLES_PREFIX}_{subset.lower()}_lead_time.csv')
            os.makedirs(TABLES_DIR, exist_ok=True)
            lead_df.to_csv(lead_csv, index=False)
            med_w = lead_df['lead_warning'].median()
            med_c = lead_df['lead_critical'].median()
            print(f"\nLead time (XGBoost, медиана по {len(lead_df)} двиг.): "
                  f"Предупреждение за {med_w:.0f} циклов до отказа, "
                  f"Авария за {med_c:.0f}. Таблица: {lead_csv}")
            if detail_plots:
                plot_lead_time(lead_df, subset, GRAPHS_DIR,
                               model_label='XGBoost')

        if save_models and split_name == 'official':
            os.makedirs(MODELS_DIR, exist_ok=True)
            for key, model in models.items():
                path = os.path.join(
                    MODELS_DIR, f'cmapss_{subset.lower()}_{key}_model.joblib')
                joblib.dump(model, path)
            print(f"\nМодели {subset} (official) сохранены в: {MODELS_DIR}")

    return rows, curve_rows, cm_rows


def merge_update_csv(out_csv: str, fresh: pd.DataFrame,
                     keys: Tuple[str, ...] = ('subset', 'split', 'model')
                     ) -> pd.DataFrame:
    """
    Слияние результатов с существующим CSV, а не перезапись.

    Свежие строки заменяют только собственные комбинации keys; результаты
    остальных сабсетов/сплитов сохраняются - иначе частный перезапуск
    затирал бы общую картину. Возвращает слитую таблицу (не сохраняет).
    """

    if os.path.isfile(out_csv):
        old = pd.read_csv(out_csv)
        fresh_keys = set(cast(pd.DataFrame, fresh[list(keys)])
                         .itertuples(index=False, name=None))
        mask = [t not in fresh_keys
                for t in cast(pd.DataFrame, old[list(keys)])
                .itertuples(index=False, name=None)]
        fresh = pd.concat([old[mask], fresh], ignore_index=True)
    return fresh


# Точка входа

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Валидация модели тяжести на NASA C-MAPSS')
    parser.add_argument('subset', nargs='?', default='all',
                        choices=list(CMAPSS_SUBSETS) + ['all'],
                        help='сабсет C-MAPSS или all (по умолчанию all)')
    parser.add_argument('--split', default='both',
                        choices=['official', 'holdout', 'both'],
                        help='режим оценки (по умолчанию both)')
    parser.add_argument('--detail-plots', action='store_true',
                        help='пер-сабсетные графики (материал приложения); '
                             'по умолчанию строится только сводный слой')
    args = parser.parse_args()

    subsets = list(CMAPSS_SUBSETS) if args.subset == 'all' else [args.subset]
    # одиночный сабсет: сводный слой малоинформативен - включается детализация
    detail = args.detail_plots or len(subsets) == 1
    all_rows: List[dict] = []
    all_curves: List[dict] = []
    all_cms: List[dict] = []
    for s in subsets:
        rows, curve_rows, cm_rows = run_subset(s, args.split,
                                               detail_plots=detail)
        all_rows.extend(rows)
        all_curves.extend(curve_rows)
        all_cms.extend(cm_rows)

    # Все три таблицы ОБНОВЛЯЮТСЯ слиянием: прогон одного сабсета заменяет
    # только собственные строки (см. merge_update_csv)
    os.makedirs(TABLES_DIR, exist_ok=True)
    out_csv = os.path.join(TABLES_DIR, f'{CMAPSS_TABLES_PREFIX}_ml_summary.csv')
    summary = merge_update_csv(out_csv, pd.DataFrame(all_rows))
    summary = summary.sort_values(['subset', 'split', 'model'],
                                  ignore_index=True)
    summary.to_csv(out_csv, index=False)

    curves_csv = os.path.join(TABLES_DIR, f'{CMAPSS_TABLES_PREFIX}_pr_curves.csv')
    curves = merge_update_csv(curves_csv, pd.DataFrame(all_curves))
    curves = curves.sort_values(['subset', 'split', 'model', 'recall'],
                                ignore_index=True)
    curves.to_csv(curves_csv, index=False)

    confusion_csv = os.path.join(TABLES_DIR,
                                 f'{CMAPSS_TABLES_PREFIX}_confusion.csv')
    confusion = merge_update_csv(confusion_csv, pd.DataFrame(all_cms))
    confusion = confusion.sort_values(
        ['subset', 'split', 'model', 'true_class', 'pred_class'],
        ignore_index=True)
    confusion.to_csv(confusion_csv, index=False)

    print(f"\n{'='*62}\nСВОДНАЯ ТАБЛИЦА (C-MAPSS, модель тяжести)\n{'='*62}")
    print(summary.to_string(index=False))
    print(f"\nСводка сохранена: {out_csv}")
    print(f"PR-кривые сохранены: {curves_csv}")
    print(f"Матрицы ошибок сохранены: {confusion_csv}")

    if detail:
        for s in subsets:
            plot_metrics_by_split(summary, s, GRAPHS_DIR)

    # Сводный слой (главные фигуры диплома) строится ВСЕГДА - из ПОЛНЫХ
    # слитых таблиц и всех накопленных lead-time CSV: обновляется даже при
    # прогоне одного сабсета, если остальные уже посчитаны.
    if summary['subset'].nunique() > 1:
        plot_summary_generalisation(summary, GRAPHS_DIR)
        if curves['subset'].nunique() > 1:
            plot_summary_pr_curves(curves, summary, GRAPHS_DIR)
        if confusion['subset'].nunique() > 1:
            plot_summary_confusion(confusion, GRAPHS_DIR)
        lead_frames = {}
        for s in sorted(summary['subset'].unique()):
            p = os.path.join(
                TABLES_DIR,
                f'{CMAPSS_TABLES_PREFIX}_{s.lower()}_lead_time.csv')
            if os.path.isfile(p):
                lead_frames[s] = pd.read_csv(p)
        if len(lead_frames) > 1:
            plot_summary_lead_time(lead_frames, GRAPHS_DIR)
