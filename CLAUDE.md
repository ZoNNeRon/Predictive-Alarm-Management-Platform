# CLAUDE.md — Контекст проекта: Predictive Alarm Platform

> Этот файл читается автоматически при запуске Claude Code в данном репозитории.
> Он описывает цель проекта, принятые архитектурные решения и текущее состояние разработки.
> Не удаляй и не изменяй этот файл без явного указания.

---

## Что это за проект

**Магистерская дипломная работа** (Иннополис, 2026):
«Разработка платформы предиктивного управления аварийными сигналами на основе промышленных данных предприятия нефтегазовой отрасли».

Цель — создать программный прототип платформы, которая:
1. Принимает телеметрию с насосного оборудования (вибрация, температура, ток, давление)
2. Классифицирует состояние агрегата: Норма / Warning / Авария
3. Подавляет ложные сигналы (alarm shelving) на основе физического контекста
4. Объясняет предсказание через SHAP (XAI)
5. Выдаёт текстовое предписание через LLM-агент (RAG) на основе нормативной документации
6. Визуализирует всё это в двухуровневом Streamlit-дашборде (оператор + инженер)

**Объект мониторинга:** центробежный вертикальный насос МНХВ, 2900 об/мин.
**Пороги по ГОСТ 32601-2013:** вибрация Warning ≥ 3.0 мм/с, Critical ≥ 8.0 мм/с; температура Warning ≥ 82°C, Critical ≥ 93°C.

---

## Структура репозитория

> **Важно:** репозиторий реструктурирован в пакетную раскладку. Код вынесен в
> доменные подпакеты `src/<domain>/`, эксперименты — в `experiments/`, все
> артефакты прогона (графики, таблицы, векторная БД) — в `artifacts/`.

```
predictive_alarm_platform/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── data_generator.py        # ГОТОВО — AR(1) + State Machine + 3 типа отказа (A/Б/В)
│   │   ├── data_preprocessor.py     # ГОТОВО — rolling features, build_fault_dataset()
│   │   └── cmapss_dataset.py        # ГОТОВО — загрузка NASA C-MAPSS из сети, RUL-метки, CmapssPreprocessor
│   ├── ml/
│   │   ├── severity_classifier_pipeline.py # ГОТОВО — модель ТЯЖЕСТИ: LR/RF/XGBoost, AlarmManager (бывш. ml_pipeline.py)
│   │   ├── fault_classifier_pipeline.py    # ГОТОВО — модель ТИПА отказа: overheat/cavitation/electrical
│   │   ├── fault_recall_analysis.py        # ГОТОВО — recall по типам отказа, доказательство 3 сигнатур
│   │   └── cmapss_ml_pipeline.py           # ГОТОВО — валидация ядра на C-MAPSS: 4 сабсета × 2 сплита × 3 модели
│   ├── xai/
│   │   ├── xai_module.py            # ГОТОВО — SHAP TreeExplainer, SymptomVector, SHAP-эвристика
│   │   └── cmapss_xai_module.py     # ГОТОВО — XAI-прогон C-MAPSS (XAIExplainer без fault-модели)
│   ├── rag/
│   │   └── rag_database.py          # ГОТОВО — ChromaDB + multilingual-e5-large, структурный чанкинг SOP
│   ├── agent/
│   │   └── ai_agent.py              # ГОТОВО — DiagnosticAgent (Ollama), 4-канальный промпт, стриминг
│   ├── runtime/                     # ГОТОВО — слой реального времени между UI и ядром
│   │   ├── platform_backend.py      # Адаптер UI↔ядро: PlatformBackend (боевой) + ProtoBackend (отладка)
│   │   ├── alarm_runtime.py         # FSM тревог с дебаунсом, Incident, AlarmJournal (гейтинг LLM)
│   │   └── online_preprocessor.py   # Потоковый stateful-препроцессор (паритет с offline FEATURE_COLS)
│   ├── app/
│   │   └── app.py                   # ГОТОВО — Streamlit двухуровневый UI (оператор + инженер)
│   └── visualisation/               # (бывш. visualisation_instruments/)
│       ├── __init__.py              # Реэкспорт всех функций визуализации
│       ├── simulation_visualisation.py  # Графики генератора (plot_smart_episode)
│       ├── ml_visualisation.py          # Графики ML (confusion, metrics, PR, recall, fault classifier)
│       ├── rag_visualisation.py         # Графики RAG (состав базы, чанки, качество поиска)
│       ├── xai_visualisation.py         # Графики XAI (waterfall, beeswarm — severity + fault)
│       ├── ai_visualisation.py          # Графики бенчмарка LLM-агента
│       ├── realtime_val_visualisation.py # ГОТОВО — ValidationCollector + графики real-time валидации (лавина тревог)
│       └── cmapss_visualisation.py      # ГОТОВО — сводный слой C-MAPSS (mean±std, PR-кривые, наложенный beeswarm)
│
├── experiments/                     # (бывш. scripts/) — исследования и регрессионные тесты
│   ├── data_stream/
│   │   └── demo_stream.py           # ГОТОВО — детерминированный демо-сценарий из датасета + ScenarioPlayer
│   ├── realtime_validation/         # ГОТОВО — режим реального времени (живая генерация парка)
│   │   ├── live_generator.py        # LiveMultiPumpGenerator + RealtimeConfig (AR(1) один-в-один с data_generator)
│   │   ├── realtime_player.py       # RealtimePlayer — драйвер тика (интерфейс ScenarioPlayer)
│   │   └── realtime_preprocessor.py # RealtimeProgressivePreprocessor — прогрессивный прогрев (признаки с 1-й строки)
│   ├── logo_cv/
│   │   └── xgboost_benchmark.py     # ГОТОВО — LOGO CV (5 фолдов), доказательство обобщения
│   ├── llm_benchmark/
│   │   └── ai_agent_benchmark.py    # ГОТОВО — бенчмарк 3 LLM: 6 сценариев (fault×stage), 5 метрик
│   └── validation/
│       ├── rag_regression_guard_test.py # ГОТОВО — guard: fault_type+stage привязка SOP
│       ├── permutation_test.py          # ГОТОВО — sanity check: метки перемешаны → accuracy ~1/3
│       ├── online_parity_test.py        # ГОТОВО — регрессия паритета препроцессоров (online == offline)
│       └── protobackend_smoke_test.py   # ГОТОВО — smoke: контракт UI ↔ ProtoBackend заморожен
│
├── config/
│   ├── __init__.py
│   ├── prompts/
│   │   └── diagnostic_agent.md  # Системный промпт LLM-агента (anti-hallucination)
│   └── settings/
│       ├── __init__.py
│       ├── settings.py          # Централизованная конфигурация (пороги, seeds, RAG, LLM)
│       └── settings_cmapss.py   # Константы ветки C-MAPSS (URL, сенсоры, RUL-границы, окна)
│
├── data/
│   ├── raw/
│   │   └── industrial_pumps_dataset.csv        # 648 000 строк, 5 насосов, 90 дней, шаг 1 мин
│   ├── processed/
│   │   ├── preprocessed_pumps_dataset.csv      # 40 rolling-признаков, target 0/1/2, fault_type
│   │   └── fault_type_pumps_dataset.csv        # выборка для классификатора типа (target=1&2)
│   └── cmapss_dataset/                         # NASA C-MAPSS (12 txt) — НЕ в git, автозагрузка из сети
│
├── artifacts/                                   # (бывш. data/graphs + data/tables + chroma_db) — всё, что генерится прогоном
│   ├── chroma_db/                               # Локальная векторная БД ChromaDB
│   ├── graphs/                                  # Графики (PNG)
│   │   ├── DG_plot_alarm_detection.png          # Симуляция: окно вокруг отказа (5 панелей)
│   │   ├── ML_plot1_confusion_matrices.png      # Модель тяжести: confusion matrix (3 модели)
│   │   ├── ML_plot2_metrics_comparison.png
│   │   ├── ML_plot3_pr_curves_critical.png
│   │   ├── ML_plot4_fault_recall_analysis.png
│   │   ├── ML_plot5_logo_cv_comparison.png      # LOGO CV: метрики по фолдам
│   │   ├── ML_fault_plot1_confusion_matrix.png  # Классификатор типа: confusion matrix
│   │   ├── ML_fault_plot2_model_comparison.png
│   │   ├── ML_fault_plot3_stage_split.png       # Recall по стадии (Warning vs Critical)
│   │   ├── shap_severity_plot1_waterfall_MNHV_005.png  # Severity-модель: waterfall
│   │   ├── shap_severity_plot2_beeswarm_overheat.png   # Severity-модель: beeswarm
│   │   ├── shap_severity_plot3_beeswarm_cavitation.png
│   │   ├── shap_severity_plot4_beeswarm_electrical.png
│   │   ├── shap_fault_plot1_waterfall_MNHV_005.png # Fault-классификатор: waterfall
│   │   ├── shap_fault_plot2_beeswarm_overheat.png  # Fault-классификатор: beeswarm
│   │   ├── shap_fault_plot3_beeswarm_cavitation.png
│   │   ├── shap_fault_plot4_beeswarm_electrical.png
│   │   ├── rag_plot1_kb_composition.png         # Состав БД (pie chart)
│   │   ├── rag_plot2_chunk_distribution.png     # Распределение длин чанков
│   │   ├── rag_plot3_retrieval_quality.png      # Качество поиска (L2 + % релевантных)
│   │   ├── rag_plot4_fault_coverage_heatmap.png # 4-канальное покрытие: Fault × Doc Type
│   │   ├── rag_plot5_section_sourcing.png       # Источники по разделам ответа агента
│   │   ├── agent_plot1_performance.png          # Производительность (время + токены/с)
│   │   ├── agent_plot2_quality_auto.png         # Профиль качества (дискриминирующие метрики)
│   │   ├── agent_plot3_summary_heatmap.png      # Сводный хитмап (relative quality)
│   │   ├── agent_plot4_stage_breakdown.png      # Стадийный разрез (Warning vs Авария)
│   │   ├── agent_plot5_expert_radar.png         # Радар экспертных оценок
│   │   ├── validation_avalanche.png             # Real-time валидация: лавина тревог (наивный vs система)
│   │   ├── validation_alarm_rate.png            # Темп тревог во времени (норматив ISA 18.2)
│   │   ├── validation_detection_latency.png     # Латентность обнаружения отказа
│   │   ├── validation_confusion.png             # Матрица ошибок детектирования (истина генератора)
│   │   ├── validation_timeline.png              # Таймлайн состояний парка
│   │   ├── cmapss_summary_plot1_generalisation.png # C-MAPSS: сравнение моделей mean±std (holdout/official)
│   │   ├── cmapss_summary_plot2_pr_curves.png   # C-MAPSS: усреднённые PR-кривые класса «Авария»
│   │   ├── cmapss_summary_plot3_lead_time.png   # C-MAPSS: упреждение обнаружения (боксплоты по сабсетам)
│   │   ├── cmapss_summary_plot4_shap_importance.png # C-MAPSS: SHAP-важность mean±std по сабсетам
│   │   ├── cmapss_summary_plot5_shap_beeswarm.png   # C-MAPSS: наложенный beeswarm всех сабсетов
│   │   └── cmapss_summary_plot6_confusion.png       # C-MAPSS: матрицы ошибок сплит×модель (сумма по сабсетам)
│   └── tables/                                  # CSV-таблицы результатов
│       ├── ML_fault_recall_table.csv
│       ├── ML_fault_classifier_summary.csv
│       ├── agent_benchmark_multi.csv            # Все прогоны бенчмарка (сырые данные)
│       ├── agent_summary_table.csv              # Сводка по моделям (index=model)
│       ├── agent_summary_by_stage.csv           # Сводка model × stage
│       ├── cmapss_ml_summary.csv                # C-MAPSS: сводка сабсет × сплит × модель (merge-update)
│       ├── cmapss_pr_curves.csv                 # C-MAPSS: точки PR-кривых на единой сетке recall
│       ├── cmapss_confusion.csv                 # C-MAPSS: счётчики матриц ошибок (merge-update)
│       └── cmapss_{fd001..fd004}_lead_time.csv  # C-MAPSS: упреждение по двигателям holdout-теста
│
├── models/
│   ├── severity/                                # Модель ТЯЖЕСТИ
│   │   ├── severity_lr_model.joblib             # LR
│   │   ├── severity_rf_model.joblib             # RF
│   │   └── severity_xgboost_model.joblib        # XGBoost (основная)
│   ├── fault_type/                              # Классификатор ТИПА отказа
│   │   ├── fault_lr_model.joblib                # LR
│   │   ├── fault_rf_model.joblib                # RF
│   │   └── fault_xgboost_model.joblib           # XGBoost
│   └── cmapss/                                  # C-MAPSS: 4 сабсета × {lr,rf,xgboost} (official-сплит)
│       └── cmapss_{fd00X}_{lr,rf,xgboost}_model.joblib
│
├── knowledge_base/
│   ├── gosts/
│   │   ├── gost_32601_2013.pdf        # ГОСТ 32601 (307 стр., НЕ загружается в БД)
│   │   └── gost_extract.md            # Выжимка: раздел 6.9, Таблицы 8-9, пороги
│   ├── manuals/
│   │   ├── mnhv_manual.pdf            # Мануал МНХВ (НЕ загружается — ЕСКД-штампы)
│   │   └── mnhv_extract.md            # Очищенный мануал: параметры, Таблица 4, ТО
│   ├── regulations/
│   │   ├── tm_regulation.pdf          # Исходный PDF (не загружается напрямую)
│   │   └── tm_regulation.md           # Очищенный MD-вариант (загружается в БД, sop)
│   ├── schedules/
│   │   ├── tm_schedule.pdf            # Исходный PDF (не загружается напрямую)
│   │   └── tm_schedule.md             # Очищенный MD-вариант (загружается в БД, schedule)
│   └── diagnostics/
│       └── diagnostics_extended.md    # Расширенная вибродиагностика (diagnostics)
│
├── requirements.txt
├── README.md                          # Витрина проекта для GitHub (RU)
└── CLAUDE.md                          # ← этот файл
```

> **Пути в коде.** Запуск модулей как `python -m src.<domain>.<module>` из корня
> репозитория; `__main__`-блоки сами добавляют корень и `src/` в `sys.path`.
> Все доменные подпакеты `src/<domain>/` — регулярные пакеты с пустым
> `__init__.py`; поддиректории `experiments/` работают как namespace-пакеты
> (это скрипты/тесты, `__init__.py` им не нужен).
> Модели грузятся из `models/severity/severity_xgboost_model.joblib` и
> `models/fault_type/fault_xgboost_model.joblib`; ChromaDB — из `artifacts/chroma_db`;
> графики/таблицы пишутся в `artifacts/graphs` и `artifacts/tables`. Эти же пути
> использует боевой `PlatformBackend` (`src/runtime/platform_backend.py`).
> Ветка C-MAPSS: модели — `models/cmapss/`, данные — `data/cmapss_dataset/`
> (автозагрузка из сети, в git не публикуются).

---

## Что уже реализовано (детально)

### 1. `src/data/data_generator.py` — Генератор данных
- **Конечный автомат (State Machine):** 5 состояний — Off (0), Startup (1), Healthy (2), Degradation (3), Critical (4)
- **AR(1) процесс** для каждого сенсора: `x[t] = μ + φ*(x[t-1] - μ) + ε[t]`. φ = 0.88–0.92 для Healthy, 0.80 для Critical
- **Три типа отказа** с разными физическими сигнатурами (распределение среди аварийных циклов):
  - `overheat` (Тип А, 55%) — температура↑ к 93+°C, ток↑ и волатильный, вибрация умеренно↑, давление норма
  - `cavitation` (Тип Б, 30%) — вибрация↑↑ к 8+ мм/с, давление↓ и пульсирует, температура и ток норма
  - `electrical` (Тип В, 15%) — ток скачет↑↑ (спайки 10%), вибрация и температура в зелёной зоне
- Деградация: AR(1) **поверх нарастающего linspace-тренда** по сигнатуре типа — XAI распознаёт как cumulative signal
- Помехи датчиков: вероятность 0.001, три отдельных флага `anomaly_vibration` / `anomaly_temperature` / `anomaly_current`
- **5 насосов** (`MNHV_001`–`MNHV_005`), `PUMP_SEEDS` из `config/settings/settings.py` (42, 137, 2026, 54321, 99999)
- **648 000 строк**, 90 дней, шаг 1 минута; схема: `[timestamp, pump_id, state, state_name, fault_type, vibration, temperature, current, pressure, anomaly_vibration, anomaly_temperature, anomaly_current]`
- Визуализация вынесена в `src/visualisation/simulation_visualisation.py`; вызов: `plot_smart_episode(df, ...)` через импорт из `src.visualisation`
- Выход: `data/raw/industrial_pumps_dataset.csv`

### 2. `src/data/data_preprocessor.py` — Подготовка признаков
- **Rolling features** (15, 30, 60 мин): mean, std, max для каждого из 4 сенсоров
- **Gradient:** `shift(1) - shift(31)` — ровно 30 шагов, консистентно с rolling(30)
- **Data leakage защита:** `shift(1)` перед каждым `.rolling()` — в строке T используется только [T-W … T-1]
- **`groupby('pump_id')`:** окна не перетекают между насосами; `min_periods=w` для mean/max
- Флаги `anomaly_*` удаляются ДО подачи в ML; `fault_type` намеренно сохраняется
- **`FEATURE_COLS`** — явный контракт из 40 признаков для train и inference
- Пороги `VIB_*/TEMP_*`, `WINDOW_SIZES`, `FAULT_TYPES` импортируются из `config/settings/settings.py`
- **`build_fault_dataset(df_processed)`** — новый метод: формирует выборку для классификатора типа из строк `target==1` и `target==2`; добавляет `fault_target` (0=overheat, 1=cavitation, 2=electrical) и `severity_stage`; те же `FEATURE_COLS` без пересчёта
- Выходы: `preprocessed_pumps_dataset.csv` + `fault_type_pumps_dataset.csv`

### 3. `src/ml/severity_classifier_pipeline.py` — Модель тяжести (бывш. `ml_pipeline.py`)
- **Group Split:** train = MNHV_001–004, test = MNHV_005 (из `config/settings`: `TRAIN_PUMPS`, `TEST_PUMPS`)
- Три модели: Logistic Regression (baseline), Random Forest (n=300), XGBoost (n=300, lr=0.1, depth=6)
- Балансировка: `class_weight='balanced'` + `sample_weight` через `compute_sample_weight`
- **Метрики:** Confusion Matrix, F1-Macro, Recall(Авария) как KPI, PR-AUC
- **Результаты XGBoost:** F1-Macro=0.97, Recall(Critical)=0.949, PR-AUC(Critical)=0.990
- **`AlarmManager`:** если `raw_state in [0, 1]` → принудительно возвращает 0 (Alarm Shelving)
- Визуализация вынесена в `src/visualisation/ml_visualisation.py`; вызов `plot_all(metrics, y_test, output_dir)` — графики сохраняются с префиксом `ML_`
- **Сохранение моделей:** `models/severity/severity_{lr,rf,xgboost}_model.joblib`
- **Шаг 10** в `__main__`: вызов `fault_recall_analysis.analyze_fault_recall(xgb_model, df_test, ...)`

### 4. `src/ml/fault_recall_analysis.py` — Валидация по типам отказа
- **Цель:** доказать, что модель ТЯЖЕСТИ различает три физических сценария, а не работает по «всё выросло → авария»
- **`analyze_fault_recall(model, df_test, feature_cols, save_graphs_dir, save_tables_dir)`** — вызывается из `severity_classifier_pipeline.py` (шаг 10) и автономно
- **Recall(Critical):** доля строк `state=4` данного `fault_type`, предсказанных как класс 2
- **Recall(Warning):** доля строк `state=3` данного `fault_type`, предсказанных как ≥ 1 (угроза не пропущена)
- **Результаты (тест MNHV_005, из `ML_fault_recall_table.csv`):** Recall(Авария) 0.945 / 0.944 / 0.969, Recall(Предупреждение) 0.970 / 0.995 / 0.983 (Перегрев/Кавитация/Электрика) — все три сигнатуры распознаются, ни один тип не пропущен
- **Fallback:** если `fault_type` отсутствует в processed CSV — присоединяется из raw по `[timestamp, pump_id]`
- Визуализация вынесена в `src/visualisation/ml_visualisation.py` (функция `recall_plot`)
- **Выходы:** `artifacts/graphs/ML_plot4_fault_recall_analysis.png`, `artifacts/tables/ML_fault_recall_table.csv`

### 5. `src/ml/fault_classifier_pipeline.py` — Классификатор типа отказа (вторая модель)
- **Второй этап иерархической классификации:** `severity_classifier_pipeline.py` определяет ТЯЖЕСТЬ (0/1/2); этот модуль определяет ФИЗИЧЕСКИЙ ТИП (overheat / cavitation / electrical)
- **Зачем отдельная модель:** SHAP-эвристика в `xai_module.py` давала точность ~0.61 из-за отсутствия физического смысла у абсолютных порогов SHAP; обучаемый классификатор даёт измеримую, защищаемую точность
- Вход: `fault_type_pumps_dataset.csv` (строки с `target==1 & 2`, `fault_target` 0/1/2)
- **Group Split:** train = MNHV_001–004, test = MNHV_005; `class_weight='balanced'`
- Три модели: LR, RF, XGBoost (n=300, lr=0.1, depth=6, `mlogloss`)
- **Метрики:** macro-F1, balanced accuracy, per-class recall, раздельно по стадии (Warning / Critical) через `severity_stage`
- **Результаты (тест MNHV_005, из `ML_fault_classifier_summary.csv`):** XGBoost (основная) macro-F1=**0.995**, balanced acc=**0.995**, recall по типам 0.996 / 0.997 / 0.992 (Перегрев/Кавитация/Электрика); RF 0.993 / 0.992; LR 0.974 / 0.976
- Визуализация через `src/visualisation/ml_visualisation.py` (функция `plot_fault_classifier`)
- **Выходы:**
  - Модели: `models/fault_type/fault_{lr,rf,xgboost}_model.joblib`
  - Графики: `ML_fault_plot1_confusion_matrix.png`, `ML_fault_plot2_model_comparison.png`, `ML_fault_plot3_stage_split.png`
  - Таблица: `artifacts/tables/ML_fault_classifier_summary.csv`

### 6. `src/xai/xai_module.py` — Объяснимый ИИ (SHAP)
- **`XAIExplainer(model_path, fault_model_path)`:** грузит ОБЕ модели — `shap.TreeExplainer` для XGBoost тяжести (`target_class_idx=2`, Авария) и для классификатора типа
- Возвращает **`SymptomVector`** (dataclass): `predicted_class`, `probabilities`, `critical_probability`, `top_symptoms`, `shap_base_value`, `inferred_fault`, `true_fault`, а также поля fault-классификатора (`fault_top_symptoms`, `fault_confidence`, `fault_probabilities`)
- Сортировка по `abs(shap_weight)` — признак -0.8 важнее признака +0.1
- **`_explain_fault_type(feature_row)`** — тип отказа определяется обученным классификатором (`fault_model.predict_proba`), SHAP объясняет выбор предсказанного класса; прежняя SHAP-эвристика (`_infer_fault_type`) удалена в пользу обучаемой модели (см. fault_classifier_pipeline.py)
- Пороги ГОСТ из `config/settings/settings.py`
- Визуализация вынесена в `src/visualisation/xai_visualisation.py`:
  - `plot_severity_waterfall()`, `plot_severity_summary_by_fault_type()` → **severity-модель** (`shap_severity_plot1-4_*`)
  - `plot_fault_waterfall()`, `plot_fault_summary_by_type()` → **fault-классификатор** (`shap_fault_plot1-4_*`)
- **Выходы:** `shap_severity_plot1-4_*` (severity SHAP) + `shap_fault_plot1-4_*` (fault-classifier SHAP)

### 7. `src/rag/rag_database.py` — База знаний (RAG)
- **Стек:** LangChain + ChromaDB (локально) + `intfloat/multilingual-e5-large` (MPS, M2)
- **`StructuredPDFLoader`:** pymupdf4llm → Markdown (сохраняет таблицы), fallback → pdfplumber; оставлен как расширение — в текущей версии не используется
- **`TextKnowledgeLoader`:** загрузка `.md`/`.txt` файлов
- **`doc_type_map` — единственный allowlist:** загружаются только файлы, явно перечисленные в нём (из `config/settings`)
- Активный `doc_type_map` (все файлы — `.md`): `tm_regulation.md` (sop), `tm_schedule.md` (schedule), `gost_extract.md` (gost), `mnhv_extract.md` (manual), `diagnostics_extended.md` (diagnostics)
- **`doc_type`:** manual / gost / sop / schedule / diagnostics; метаданные чанков: `source`, `section`, `chunk_id`, `doc_type`
- **Разные chunk_size по типу:** gost=700, manual=600, sop=550, schedule=350, diagnostics=500
- **e5 prefix:** `passage:` при загрузке, `query:` при поиске — обязательно для e5-large
- **Порог релевантности:** `RELEVANCE_THRESHOLD=1.2` (L2) из `config/settings`; для графика ТО — `1.5`
- Батчевая запись по 500 чанков

**Структурированный чанкинг регламента SOP (`_chunk_sop_document`):**
- Регламент `tm_regulation.md` содержит сценарии по типам отказа с двумя стадийными блоками оператора
- Заголовки `### Сценарий ... [fault_type: overheat/cavitation/electrical]` — тег типа
- Каждый сценарий режется на подразделы: `operator` (Действия оператора), `repair` (Работы ТОиР), `reference` (Симптоматика/Причины)
- Operator-блоки несут тег `stage`: `warning` или `critical` (из заголовка подраздела)
- `fault_type` carry-forward: если чанк потерял заголовок при сплите — наследует тип предыдущего

**Функция `resolve_stage(symptom_vector)`** (единый резолвер для агента и бенчмарка):
- Возвращает `'warning'` / `'critical'` / `'unknown'`
- `'unknown'` — если класс ∉ {1,2} или `fault_confidence < FAULT_CONFIDENCE_THRESHOLD (0.5)`

**Методы поиска:**
- `search(query, k, doc_type_filter, metadata_filter, apply_threshold)` — базовый с `$and`-фильтром; `apply_threshold=False` для фолбэков
- `search_by_symptoms(symptom_dict, k)` — multi-query: два запроса (описательный + прескриптивный), дедупликация по содержимому; из результатов фильтруются `sop_part in ('operator', 'repair')` — они идут в отдельные каналы
- `search_operator_actions(fault_type, stage, k)` — строго по `fault_type + sop_part='operator' + stage`; лестница из 4 фолбэков (ослабляет фильтр по шагам)
- `search_repair_works(fault_type, k)` — строго по `fault_type + sop_part='repair'`; лестница из 3 фолбэков
- `search_maintenance_schedule(pump_id, k)` — по `doc_type='schedule'`, threshold=1.5

- Визуализация: `plot_all_rag(kb, test_queries, plots_dir)` в `__main__` (принимает объект `kb`, не отдельные параметры)
- **Выходы:** `rag_plot1–5_*.png`

### 8. `src/agent/ai_agent.py` — LLM-агент
- **`DiagnosticAgent`** — принимает `SymptomVector` + 4 канала RAG-контекста, строит промпт, вызывает Ollama
- **`AgentResponse`** dataclass: `pump_id`, `raw_text`, `model_name`, `used_context`, `sources`, `latency_sec`, `gen_time_sec`, `eval_count`, `prompt_eval_count`, `tokens_per_sec`, `format_ok`, `error`
- **`SYSTEM_PROMPT`** загружается из `config/prompts/diagnostic_agent.md` (anti-hallucination, строгий формат 5 разделов: СТАТУС / ДИАГНОЗ / ПРЕДПИСАНИЕ / РЕКОМЕНДАЦИИ ТОиР / ПЛАНОВЫЙ РЕМОНТ)
- Qwen 3.x: `think=False` через нативный Ollama-клиент отключает chain-of-thought

**4-канальная архитектура промпта (`_build_full_prompt`):**
- `rag_results` → блок СПРАВОЧНЫЙ КОНТЕКСТ (мануал/ГОСТ/диагностика) — только для ДИАГНОЗА
- `operator_results` → блок ДЕЙСТВИЯ ОПЕРАТОРА (регламент) — единственный источник ПРЕДПИСАНИЯ
- `repair_results` → блок РАБОТЫ ТОиР — для РЕКОМЕНДАЦИЙ ТОиР
- `schedule_results` → блок ГРАФИК ТОиР — для ПЛАНОВОГО РЕМОНТА
- Детерминированный **вердикт по внеплановому выводу** (P(аварии) ≥ 80% → вывод, иначе нет) вставляется в промпт перед блоком графика, модель переписывает его дословно

**Дополнительные методы:**
- `_event_header(sv)` — детерминированная шапка `АГРЕГАТ: ... | ВРЕМЯ: ...`, всегда в начале ответа (не доверяется LLM)
- `_unknown_response(sv)` — детерминированный ответ при `resolve_stage == 'unknown'`; LLM не вызывается
- `generate_prescription_stream(...)` — потоковый генератор (Ollama `stream=True`): сначала отдаёт шапку, затем стримит токены. В UI потребляется фоновым daemon-потоком (см. раздел 17), который аккумулирует чанки в потокобезопасный store; частичный текст показывается в тосте по мере накопления
- `_display_source(filename)` — имя файла → читаемое имя документа (из `DOC_DISPLAY_NAMES`)

- **`__main__`:** цепочка XAI → RAG (4 канала) → LLM с последовательным прогоном трёх моделей
- Три тестируемые модели: `qwen3.5:9b`, `phi4:14b`, `second_constantine/yandex-gpt-5-lite:8b`; рабочая модель платформы задаётся `DEFAULT_LLM_MODEL` в `config/settings` (сейчас — `second_constantine/yandex-gpt-5-lite:8b`)

### 9. `experiments/logo_cv/xgboost_benchmark.py` — LOGO CV бенчмарк
- **Leave-One-Group-Out Cross-Validation:** 5 фолдов, каждый раз один насос — тест, остальные 4 — обучение
- **Цель:** доказать, что XGBoost обобщается на любой новый агрегат парка (не переобучен на MNHV_005)
- Метрики: F1-Macro, Recall(Critical), PR-AUC по каждому фолду
- Визуализация: `ML_plot5_logo_cv_comparison.png`
- Содержит собственную копию `evaluate_model()` (дублирует логику из `severity_classifier_pipeline.py`)

### 10. `experiments/llm_benchmark/ai_agent_benchmark.py` — Бенчмарк LLM-агента
- **6 сценариев:** 3 типа отказа × 2 стадии (Предупреждение / Авария) — строит `build_scenarios(xai, df, preprocessor, kb)`
- Каждый сценарий получает полный 4-канальный вход (rag, operator, repair, schedule)
- **5 автометрик:**
  - `format_ok` — все 4 обязательных раздела (из `REQUIRED_SECTIONS`)
  - `attribution_ok` — явная ссылка на норматив (из `SOURCE_MARKERS`)
  - `action_steps` — число нумерованных пунктов в ПРЕДПИСАНИИ
  - `toir_is_works` — в разделе ТОиР ремонтные работы, а не даты (из `TOIR_WORK_MARKERS`)
  - `stage_appropriate` — стадийная уместность: на Предупреждение нет команды аварийного останова (`EMERGENCY_MARKERS`), на Аварию есть решительное действие (`DECISIVE_MARKERS`)
- Прогрев модели перед замером (первый сценарий)
- Усреднение по 6 сценариям × 3 повтора = 18 прогонов на модель
- **Выходы:** `agent_benchmark_multi.csv`, `agent_summary_table.csv` (index=model), `agent_summary_by_stage.csv` (model × stage), `agent_plot1-5_*.png`

### 11. `experiments/validation/rag_regression_guard_test.py` — Регрессионный guard RAG
- **Цель:** защита от регрессии в стадийной и сценарной привязке SOP-чанков после перестройки базы
- **3 теста:**
  1. `test_sop_chunks_tagged_with_fault_and_stage` — все 3 типа размечены `fault_type`; у operator-чанков есть `stage`; оба стадийных блока (warning + critical) присутствуют для каждого типа
  2. `test_operator_actions_locked_by_fault_and_stage` — `search_operator_actions(fault, stage)` возвращает чанки строго своего типа и стадии; позитивная и негативная лексическая проверка по `STAGE_OPERATOR_KW`
  3. `test_repair_works_locked_by_fault` — `search_repair_works(fault)` не смешивает сценарии
- Запуск: `pytest experiments/validation/rag_regression_guard_test.py -v` или `python experiments/validation/rag_regression_guard_test.py`
- Предусловие: база собрана с актуальной разметкой (`kb.build_database(reset=True)`) в `artifacts/chroma_db`

### 12. `experiments/validation/permutation_test.py` — Sanity check классификатора типа
- Загружает `fault_type_pumps_dataset.csv`, перемешивает `fault_target` случайным образом (`rng(42)`)
- Обучает XGBoost на перемешанных метках, проверяет balanced accuracy на тесте — должна упасть к ~1/3
- **Цель:** доказать, что модель учится реальным сигнатурам, а не паразитным паттернам в данных

### 13. `src/runtime/online_preprocessor.py` — Потоковый препроцессор (РЕАЛТАЙМ)
- **`OnlinePreprocessor`** — stateful-расчёт того же вектора `FEATURE_COLS`, что и offline `data_preprocessor.py`, но по одной строке через кольцевой буфер (`deque`) на каждый `pump_id`
- **Контракт паритета online == offline** (не менять без regression-теста): `shift(1)` (текущая строка не входит в окно), `min_periods=w`, `diff_30 = x[T-1] - x[T-31]`, `std` с `ddof=1` (как pandas)
- **Прогрев:** `WARMUP_ROWS = 60` строк; `push()` возвращает `None`, пока буфер не заполнен; именно поэтому демо-сценарий несёт 60-минутный warmup-префикс
- **`verify_parity(raw_df, offline_features, feature_cols)`** — сверка потокового расчёта с offline-матрицей (бросает `AssertionError` на первой расходящейся колонке)

### 14. `src/runtime/alarm_runtime.py` — Runtime-слой управления тревогами
- **`PumpAlarmFSM`** — дебаунс-автомат тревог по парку: переход подтверждается только после N одинаковых предсказаний подряд (`confirm_up=2`, `confirm_down=5` — эскалация быстрее деэскалации); практика рационализации ISA 18.2 / EEMUA 191
- **Гейтинг LLM:** `update()` возвращает `Incident` только на подтверждённой ЭСКАЛАЦИИ, требующей предписания — тяжёлая цепочка XAI→RAG→агент (~20 с) запускается один раз на стадию, а не на каждый тик
- **`Incident`** — жизненный цикл инцидента (Предупреждение→Авария→сброс); кеш предписаний, трасс извлечения и `SymptomVector` по стадиям
- **`AlarmJournal`** — журнал всех событий, включая подавленные state-based фильтром сигналы (требование ФЗ-116 / ГОСТ Р 22.1.12-2005: скрытые сигналы хранятся в архиве)

### 15. `src/runtime/platform_backend.py` — Адаптер UI ↔ аналитическое ядро
- **Единственная точка интеграции:** `app.py` не импортирует ML/XAI/RAG/агента напрямую — только этот адаптер; замена источника данных (демо-CSV → SCADA) и смена сигнатур модулей локализованы здесь
- **`PlatformBackend`** (боевой): грузит severity-модель + `AlarmManager`, `XAIExplainer` (обе модели), `KnowledgeBaseManager`, `DiagnosticAgent`; `process_tick()` (дешёвый ML на каждый тик), `explain()`, `prescription_stream()` (4 канала RAG + стрим агента), `retrieval_trace()`, `shap_figures()`
- **`ProtoBackend`** (отладка): тяжесть/тип берутся из меток датасета, предписание имитируется — позволяет верстать UI без Ollama/ChromaDB/моделей. UI откатывается на него автоматически, если боевой backend не инициализировался

### 16. `experiments/data_stream/demo_stream.py` — Детерминированный демо-сценарий
- **`extract_demo_scenario(dataset_path, fault_type, pump_id='MNHV_005', ...)`** — НЕ генерирует данные заново, а вырезает готовый эпизод `[warmup 60 мин → Healthy → Degradation → Critical]` заданного типа из размеченного датасета (воспроизводимость + честность: тот же unseen MNHV_005, на котором валидированы модели)
- **`ScenarioPlayer`** — плеер потока: выдаёт строки по одной, хранит позицию (живёт в `st.session_state`), `skip_warmup()` прогоняет warmup-префикс пакетом

### 17. `src/app/app.py` — Streamlit двухуровневый UI (NAMUR NE 129)
- **Мультистраничная навигация** (`st.Page` / `st.navigation`, `position="hidden"`): Оператор (`url_path="operator"`, default) и Инженер (`url_path="engineer"`) — настоящие независимые страницы (переключение кнопками в сайдбаре через `st.switch_page`); поддеревья изолированы, экраны не «протекают» друг в друга
- **Два источника данных** (радиокнопка в сайдбаре):
  - **Датасет (демо)** — `extract_demo_scenario` + `ScenarioPlayer` + строгий `OnlinePreprocessor` (60-мин прогрев). Путь к CSV резолвится от `_PROJECT_ROOT` (не от CWD Streamlit)
  - **Реальное время** — `LiveMultiPumpGenerator` + `RealtimePlayer` + `RealtimeProgressivePreprocessor` (признаки с 0-й минуты, холодный старт `warmup_rows=0`)
- **Оператор:** карта оборудования (плитки-статусы NAMUR NE 107), счётчики активных аварий/предупреждений/подавленных/переходов-в-час (скользящее окно 60 мин), drill-down в агрегат с интерактивными Plotly-графиками 4 параметров (пороги-линии, **ось реального времени**, фикс-диапазоны `YRANGE`, ховер); предписания — стопка «тостов» справа снизу (по одной карточке на насос, старшая стадия; цвет рамки по тяжести; сворачивается/квитируется)
- **Инженер:** список **ВСЕХ агрегатов**, отсортированных по тяжести (Авария→Предупреждение→Норма); для «Нормы» — снимок текущих параметров, для инцидента — вкладки: SHAP обеих моделей (живой с троттлингом ~5 c, либо замороженный при аварии), таблицы симптомов (live по текущему окну на предупреждении), трассировка RAG по каналам, план/история ТОиР
- **Сайдбар-«язычок»:** история предписаний + панель источника данных/воспроизведения + кнопка «Сохранить графики валидации» (real-time режим)

**Архитектура реального времени:**
- **Перерисовка через фрагменты:** экраны и история обёрнуты в `@st.fragment(run_every=REFRESH)` (`REFRESH=2.0 с`); движок (`_engine_tick`) гонит поток и опрашивает фон **внутри** фрагмента — нет глобального `time.sleep`/`st.rerun`, который перебивал бы клики (квитировать/развернуть) и кидал экран наверх
- **Поток данных не останавливается на эскалации:** `advance_stream()` на триггере вызывает `_on_escalation()` без `break`/снятия `playing` — ползунок и графики продолжают идти
- **Фоновая генерация предписания:** медленная цепочка RAG+LLM (~13 с) — в daemon-потоке; задания в потокобезопасном сторе `get_gen_store()` (`@st.cache_resource` — переживает rerun'ы Streamlit). Главный проход опрашивает store (`poll_generations()`) и дописывает готовый текст в `Incident.prescriptions`
- **Факт инцидента фиксируется сразу:** `_on_escalation()` синхронно считает `SymptomVector` → инцидент мгновенно появляется в истории/тосте; быстрый переход Предупреждение→Авария не теряет предупреждение
- **Журнал событий истории `ss.events`** — устойчивый append-only лог (одна запись = одно возникновение с уникальным `eid`), единственный источник левой панели. Не зависит от мутаций `stage_ts` инцидента в FSM (раньше история строилась из `fsm.all_incidents()` и хрупко склеивалась по временно́му окну — заменено). Кап `MAX_EVENTS=100` (`_cap_events` при каждом `_append_event`): выбрасываются старейшие ЗАВЕРШЁННЫЕ записи (acked/resolved), активные сохраняются; осиротевшие кэши `shap_frozen`/`_presc_ft` подчищаются (`_prune_incident_caches`)
- **Жизненный цикл алармов:** `ss.tripped` (авария держит «Отказ» + глушит повторный трип до квитирования), `ss.recovering` (после квитирования гасит стейл `fsm.state=2` до возврата в норму), `ss.pinned` (ссылка на инцидент переживает закрытие в FSM), `ss._deadzone` (мёртвая зона после пуска: окно обнуляется, пока пусковой ток выше порога), `_refine_warning_type` (уточнение типа отказа по мере развития слабой ранней сигнатуры)
- **Ре-наг (`poll_renags()`):** если после квитирования проблема **реально держится** (инцидент всё ещё текущий для агрегата) дольше `RENAG_MIN` (10 sim-мин) — в историю добавляется **новая строка** возникновения (`_append_event`), прежняя остаётся квитированной; предписание то же (текст в `Incident.prescriptions`)
- Запуск: `streamlit run src/app/app.py`

### 18. `experiments/realtime_validation/live_generator.py` — Живой генератор парка
- **`LiveMultiPumpGenerator`** — генерирует телеметрию ПОШАГОВО («здесь и сейчас»): на тик — по строке на каждый насос парка по единым модельным часам. Формулы AR(1), сигнатуры 3 типов отказа, startup-всплеск, аномалии и State Machine перенесены **один-в-один** из `data_generator.py` (те же μ, φ, σ, linspace-тренды); отличия только структурные (поминутный шаг, RNG-поток на насос, ускоренные тайминги)
- **`RealtimeConfig`** — ручки темпа/вероятностей (длительность деградации `degradation_min/max`, частота аномалий и т.д.), чтобы валидация шла минуты, а не дни
- **Квитирование как обратная связь:** `acknowledge(pump_id)` — Degradation+ack → 50% возврат в Healthy; Critical+ack → 100% останов (Off→Startup→Healthy, «режим предотвращённых сигналов»); `trip(pump_id)` — принудительный останов аварийного агрегата

### 19. `experiments/realtime_validation/realtime_player.py` — Драйвер тика реального времени
- **`RealtimePlayer`** — тонкая обёртка над `LiveMultiPumpGenerator`, повторяет интерфейс `ScenarioPlayer` (`next_rows`/`finished`/`progress`/`pos`/`__len__`/`skip_warmup`) → встаёт на место плеера в `app.py` без переписывания UI
- За тик отдаёт строку на КАЖДЫЙ насос (общий timestamp); `pos` считает sim-минуты; `acknowledge`/`trip` пробрасываются в генератор. С `warmup_rows=0` — честный холодный старт (в паре с прогрессивным препроцессором)

### 20. `experiments/realtime_validation/realtime_preprocessor.py` — Прогрессивный препроцессор
- **`RealtimeProgressivePreprocessor(OnlinePreprocessor)`** — выдаёт вектор `FEATURE_COLS` с ПЕРВОЙ строки (расширяющиеся → скользящие окна), снимая «слепое окно» 60-мин прогрева на старте демо. Наследует `__init__`/`feature_cols`, переопределяет только политику прогрева
- **Паритет на полном окне:** начиная с 60-й строки результат бит-в-бит совпадает со строгим `OnlinePreprocessor` (и, значит, с offline-обучением) — самотест в `__main__`. `shift(1)`, порядок колонок, `std` (ddof=1, 0 при <2 точках), `diff_30` (0 при <31 строке) сохранены
- `reset(pump_id=None)` совместим с базовым: без аргумента — полный сброс всех буферов (старт сценария), с `pump_id` — сброс конкретного агрегата (трип/пуск)
- `rows_seen(pump_id)` — для бейджа «прогрев»/«накопление истории» на плитке

### 21. `src/visualisation/realtime_val_visualisation.py` — Метрики и графики real-time валидации
- **`ValidationCollector`** — накапливает per-tick лог прогона (истина генератора + что выдала система); `from_settings()`, `add(...)`, `to_frame()`
- **`summarize(log)`** — сводные метрики; **`render_all(log, outdir)`** — публикационные графики в `artifacts/graphs/`:
  - `validation_avalanche.png` — **главное доказательство цели:** сколько тревог выдал бы наивный пороговый алармер vs сколько оставила система (гашение окнами/состоянием/дебаунсом)
  - `validation_alarm_rate.png` — темп тревог во времени (норматив ISA 18.2 ≤6/час), `validation_detection_latency.png` — латентность обнаружения, `validation_confusion.png` — матрица ошибок против истины генератора, `validation_timeline.png` — таймлайн состояний парка
- Полностью автономен от моделей (есть самотест на синтетическом логе)

### 22. `experiments/validation/online_parity_test.py` — Регрессия паритета препроцессоров
- **Цель:** гарантировать, что потоковый расчёт признаков бит-в-бит совпадает с offline-расчётом, на котором обучены модели; любое расхождение = рассинхрон train/inference
- **3 проверки:**
  1. `verify_parity`: строгий `OnlinePreprocessor` == offline `DataPreprocessor` на каждой строке после прогрева (тот же `shift(1)`, `min_periods=w`, `diff_30`, `std` ddof=1) — иначе `AssertionError` с первой расходящейся колонкой
  2. `RealtimeProgressivePreprocessor` == строгий препроцессор на ПОЛНОМ окне (≥ `WARMUP_ROWS`)
  3. На частичном окне прогрессивный ВЫДАЁТ признаки, а строгий молчит — подтверждение разной политики прогрева (фича, не баг)
- **Выравнивание индексов:** offline считается через `process(is_training=False)` — без dropna/reset_index, исходный индекс raw сохраняется
- **Скорость:** берутся первые `HEAD_ROWS` строк каждого насоса, а не весь датасет 648k строк
- Запуск: `pytest experiments/validation/online_parity_test.py -v` или `python experiments/validation/online_parity_test.py`
- Это тот самый regression-тест, без которого запрещено менять контракт паритета в `online_preprocessor.py`

### 23. `experiments/validation/protobackend_smoke_test.py` — Smoke-тест отката UI (ProtoBackend)
- **Цель:** `app.py` при сбое инициализации боевого `PlatformBackend` молча падает на `ProtoBackend`; если прототип отстал от того, что дёргает UI, — дашборд рушится вместо graceful-демо. Тест ЗАМОРАЖИВАЕТ контракт UI ↔ backend
- **Проверяемый контракт** (собран из `app.py`, держать в синхроне при правках UI):
  - методы backend: `process_tick`, `explain`, `prescription_stream`, `retrieval_trace`, `shap_figures`; атрибут `preproc` с `.feature_cols` / `.reset()`
  - поля SymptomVector: `probabilities[1]` (drill-down предупреждения!), `critical_probability`, `inferred_fault`, `fault_confidence`, `fault_probabilities`, `top_symptoms`, `fault_top_symptoms`; у элемента симптома — `feature/sensor/value/shap_weight`
  - `retrieval_trace` → `list[dict]` с ключами для таблицы инженера
- **Headless:** `app.py` не импортируется (там `st.set_page_config` на верхнем уровне) — контракт закодирован явными списками
- Запуск: `pytest experiments/validation/protobackend_smoke_test.py -v` или `python experiments/validation/protobackend_smoke_test.py`

### 24. Ветка C-MAPSS — валидация ядра на реальных данных NASA

**Цель:** доказать переносимость пайплайна `preprocessing → ML → XAI` на реальный
run-to-failure датасет другого класса оборудования (турбовентиляторные двигатели,
NASA PCoE, 4 подмножества FD001–FD004). Ядро переиспользуется, а не переписывается:
новые модули строго ОТДЕЛЕНЫ от насосных (свои файлы, свой конфиг), существующий
код не тронут.

**`src/data/cmapss_dataset.py`** — данные:
- `download_cmapss()` — идемпотентная загрузка с официального зеркала NASA (S3) +
  рекурсивная распаковка вложенных zip; датасет НЕ в git (`.gitignore`:
  `data/cmapss_dataset/`), оркестратор скачивает его сам при первом запуске
- `compute_rul()` — RUL: train `max(cycle) - cycle`; test — финальный RUL из
  `RUL_FD00X.txt` + смещение; `add_severity_target()` — piecewise-метки тяжести:
  RUL > 50 → 0, 20–50 → 1, ≤ 20 → 2 (границы `RUL_WARNING`/`RUL_CRITICAL`)
- Режимная нормализация FD002/FD004 (6 полётных режимов): id режима по ближайшему
  якорю op1 (`CMAPSS_REGIME_OP1_ANCHORS`), per-режимный z-score, статистики
  ТОЛЬКО по train
- **`CmapssPreprocessor(DataPreprocessor)`** — наследник насосного препроцессора:
  переопределены только список признаков (14 информативных сенсоров, окна
  [5, 10, 20] циклов, diff-лаг 10) — механика `shift(1)` / `groupby('pump_id')` /
  `min_periods=w` не тронута; 140 признаков, `ENGINE_ID_FMT` = `FD001_E001`

**`src/ml/cmapss_ml_pipeline.py`** — ML (запуск: `python -m src.ml.cmapss_ml_pipeline`, по умолчанию `all`):
- Переиспользует `evaluate_model()` из `severity_classifier_pipeline` и тройку
  моделей с теми же гиперпараметрами; отличие (задокументированное): LR в
  `make_pipeline(StandardScaler(), ...)` — сенсоры C-MAPSS различаются на 3 порядка
- **Два режима оценки:** `official` (train/test бенчмарка NASA; test обрезан до
  отказа — классы 1/2 редки, реалистичный продакшн-профиль) и `holdout`
  (группированный сплит по двигателям внутри train — контроль, что official-метрики
  не артефакт дисбаланса)
- `compute_lead_times()` — RUL при первом срабатывании стадии по каждому двигателю
  holdout-теста (все дожиты до отказа)
- Таблицы обновляются **слиянием** (`merge_update_csv` по ключу subset×split×model),
  а не перезаписью; PR-кривые интерполируются на единую сетку recall
  (`PR_RECALL_GRID`, 201 точка) и копятся в `cmapss_pr_curves.csv`
- **Результаты (XGBoost, mean±std по 4 сабсетам):** holdout PR-AUC(Авария)
  **0.95±0.02**, F1-Macro 0.84±0.02; official PR-AUC 0.77±0.12 (FD004 честно
  труднейший — 0.63); LogReg не уступает ансамблям. Медианное упреждение:
  Предупреждение за ~48–50, Авария за ~16–20 циклов до отказа; из 142 двигателей
  не пропущен ни один

**`src/xai/cmapss_xai_module.py`** — XAI (запуск: `python -m src.xai.cmapss_xai_module`, по умолчанию `all`):
- `XAIExplainer` переиспользуется БЕЗ изменений (`fault_model_path=None` — C-MAPSS
  не даёт метку типа); консольный `SymptomVector` по предотказной строке с
  физической расшифровкой сенсоров (`CMAPSS_SENSOR_DESC`)
- SHAP по предотказным строкам считается ОДИН раз на сабсет (`MAX_SHAP_ROWS=1200`);
  из него — и важность признаков, и блок точек наложенного beeswarm
- Топ-признаки физически осмысленны: Ps30, T50, phi, NRc — классические индикаторы
  деградации HPC

**`src/visualisation/cmapss_visualisation.py`** — визуализация:
- Философия «сводной подтверждённой картины»: по умолчанию строится ТОЛЬКО сводный
  слой из 6 фигур (`cmapss_summary_plot1–6`); пер-сабсетная детализация — по флагам
  `--detail-plots` / `--detail` (автоматически при прогоне одного сабсета)
- Сводные матрицы ошибок (`plot_summary_confusion`, plot6): сетка сплит × модель,
  агрегация СУММОЙ абсолютных счётчиков по сабсетам (не «средняя матрица») —
  нормировка по строке после суммирования даёт recall на объединении всех тестов
- `display_feature_name()`: `s11_mean_5` → `Ps30 | mean_5` — подписи SHAP-графиков
  в физических кодах сенсоров
- **Наложенный beeswarm** (`plot_summary_shap_beeswarm`): SHAP-матрицы всех сабсетов
  конкатенируются; значения признаков переведены в ПЕРЦЕНТИЛЬНЫЕ РАНГИ внутри своего
  сабсета — сырые (FD001/FD003) и z-нормированные (FD002/FD004) шкалы не смешиваются,
  цветовая семантика «high/low» остаётся честной
- `__main__` пересобирает сводные фигуры 1–3 из сохранённых CSV без переобучения

---

## Централизованная конфигурация (`config/`)

### `config/settings/settings.py`
Единственный источник истины для всех константных значений проекта.
Импортируется во все модули `src/` и `experiments/`:

| Константа | Содержание |
|---|---|
| `THRESHOLDS` | Пороги ГОСТ: vibration/temperature warning + critical |
| `PUMP_SEEDS` | `{pump_id: seed}` для воспроизводимости генерации |
| `FAULT_TYPES`, `FAULT_WEIGHTS` | Типы и доли отказов |
| `FAULT_LABELS` | Русские метки типов для графиков |
| `WINDOW_SIZES` | `[15, 30, 60]` — окна rolling-признаков |
| `PUMPS`, `TRAIN_PUMPS`, `TEST_PUMPS` | Разбивка парка на train/test |
| `EMBED_MODEL` | `intfloat/multilingual-e5-large` |
| `RELEVANCE_THRESHOLD` | `1.2` — порог L2-расстояния для RAG |
| `DOC_TYPES`, `CHUNK_CONFIG`, `DOC_TYPE_MAP` | Конфигурация базы знаний; `DOC_TYPE_MAP` содержит только `.md`-файлы |
| `DOC_TYPE_SHORT_RU` | Короткие русские подписи типов документов для осей/ячеек хитмапов RAG-визуализации |
| `DOC_DISPLAY_NAMES` | Читаемые имена документов для атрибуции в ответах агента |
| `FAULT_CONFIDENCE_THRESHOLD` | `0.5` — порог уверенности классификатора типа; ниже → `stage='unknown'` |
| `LLM_MODELS`, `DEFAULT_LLM_MODEL` | Список тестируемых Ollama-моделей |
| `MODEL_LABELS` | Читаемые названия моделей для графиков (`{model_id: label}`) |
| `REQUIRED_SECTIONS` | 4 обязательных раздела ответа агента для проверки формата |
| `SOURCE_MARKERS` | Ключевые слова атрибуции источника (для `check_attribution()`) |
| `TOIR_WORK_MARKERS` | Маркеры ремонтных работ в разделе ТОиР (для `check_toir_is_works()`) |
| `EMERGENCY_MARKERS` | Маркеры аварийного останова — недопустимы на стадии Предупреждение |
| `DECISIVE_MARKERS` | Маркеры решительного действия — обязательны на стадии Авария |
| `DIRECTIONS`, `FMT` | Направления метрик и форматы для хитмапа бенчмарка |
| `ISA_ALARM_RATE_LIMIT` | Предел тревог/час (ISA 18.2, `=6`); единый для дашборда и графиков валидации |
| `UI_REFRESH_SEC`, `UI_RENAG_MIN`, `UI_GEN_TIMEOUT_SEC`, `UI_MAX_EVENTS`, `UI_GRAPH_WINDOW_MIN` | Тюнинг runtime дашборда (`app.py`): период перерисовки, ре-наг, таймаут генерации, кап истории, окно трендов |
| `UI_YRANGE`, `UI_SEVERITY_COLORS`, `UI_ACK_COLOR` | Диапазоны осей графиков оператора и цвета статусов тяжести/квитирования |
| `SIM_AMBIENT_TEMP`, `SIM_HEALTHY_FIXED`, `SIM_START_DATE` | Физконстанты live-генератора парка (`experiments/realtime_validation/live_generator.py`) |

> `app.py` импортирует UI-константы из `config.settings` через алиасы (короткие
> локальные имена `REFRESH`/`RENAG_MIN`/… сохранены). `RealtimeConfig`
> (тайминги/вероятности симуляции) остаётся отдельным конфиг-датаклассом.

### `config/settings/settings_cmapss.py`
Источник истины констант ветки C-MAPSS. Отделён от `settings.py` намеренно —
константы насосного пайплайна и валидации на реальных данных не смешиваются:

| Константа | Содержание |
|---|---|
| `CMAPSS_URLS`, `CMAPSS_DATA_SUBDIR` | Зеркало NASA (S3) и папка данных (не в git) |
| `CMAPSS_SUBSETS`, `CMAPSS_SUBSET_INFO` | 4 подмножества: число режимов/типов отказа, подписи |
| `CMAPSS_RAW_COLUMNS`, `ENGINE_ID_FMT` | Схема сырых txt (26 колонок) и id двигателя |
| `CMAPSS_SENSORS` | 14 информативных сенсоров (константные исключены) |
| `CMAPSS_SENSOR_SHORT`, `CMAPSS_SENSOR_DESC` | Физические коды (Ps30, phi...) и расшифровки для SHAP/консоли |
| `RUL_WARNING=50`, `RUL_CRITICAL=20` | Piecewise-границы меток тяжести из RUL |
| `CMAPSS_WINDOW_SIZES=[5,10,20]`, `CMAPSS_DIFF_LAG=10` | Окна признаков в ЦИКЛАХ (не минутах) |
| `CMAPSS_MULTI_REGIME_SUBSETS`, `CMAPSS_REGIME_OP1_ANCHORS` | Режимная нормализация FD002/FD004 |
| `CMAPSS_MODELS_SUBDIR`, `CMAPSS_GRAPHS_PREFIX`, `CMAPSS_TABLES_PREFIX` | Пути/префиксы выходов |

### `config/prompts/diagnostic_agent.md`
Системный промпт `DiagnosticAgent`. Загружается при импорте `src/agent/ai_agent.py`. Редактируется без изменения Python-кода.
- 8 строгих правил: только из КОНТЕКСТА, источник по имени, разделение 4 каналов (справочный/оператор/ТОиР/график), стадийный тон (Авария vs Предупреждение), краткость
- Вердикт по внеплановому выводу в ремонт вставляется в промпт детерминированно агентом, модель переписывает его дословно
- Формат ответа: 5 разделов — СТАТУС / ДИАГНОЗ И ОБОСНОВАНИЕ / ПРЕДПИСАНИЕ (ДЕЙСТВИЯ ОПЕРАТОРА) / РЕКОМЕНДАЦИИ ТОиР (РЕМОНТНОЙ БРИГАДЕ) / ПЛАНОВЫЙ РЕМОНТ

---

## Визуализация (`src/visualisation/`)

Все функции реэкспортируются через `__init__.py` — импорт: `from src.visualisation import plot_all`.

| Модуль | Функции | Откуда вызывается |
|---|---|---|
| `simulation_visualisation.py` | `plot_smart_episode(df, hours, save_dir)` | `data_generator.py` |
| `ml_visualisation.py` | `plot_all(metrics, y_test, output_dir)`, `recall_plot(results, signatures, save_dir)`, `plot_fault_classifier(metrics, output_dir)` | `severity_classifier_pipeline.py`, `fault_recall_analysis.py`, `fault_classifier_pipeline.py` |
| `rag_visualisation.py` | `plot_all_rag(kb, test_queries, save_dir)` → 5 графиков: `plot_knowledge_base_stats`, `plot_chunk_length_distribution`, `plot_retrieval_quality`, `plot_fault_coverage_heatmap`, `plot_fault_section_sourcing` | `rag_database.py` |
| `xai_visualisation.py` | `plot_severity_waterfall()`, `plot_severity_summary_by_fault_type()` — severity; `plot_fault_waterfall()`, `plot_fault_summary_by_type()` — fault classifier | `xai_module.py`, `platform_backend.py` |
| `ai_visualisation.py` | `plot_performance(df, save_dir)`, `plot_quality_auto(df, save_dir)`, `plot_summary_heatmap(summary_df, directions, fmt, save_dir)`, `plot_expert_radar(expert_scores, save_dir)`, `plot_stage_breakdown(df, save_dir)` | `experiments/llm_benchmark/ai_agent_benchmark.py`, автономно |
| `realtime_val_visualisation.py` | `ValidationCollector`, `summarize(log)`, `render_all(log, outdir)` → `plot_avalanche`, `plot_alarm_rate`, `plot_detection_latency`, `plot_confusion`, `plot_state_timeline_all` | `app.py` (real-time режим), автономно (самотест) |
| `cmapss_visualisation.py` | Сводный слой: `plot_summary_generalisation`, `plot_summary_pr_curves`, `plot_summary_lead_time`, `plot_summary_shap_importance`, `plot_summary_shap_beeswarm`; пер-сабсетные (detail): confusion, PR, lead time, beeswarm, waterfall; `display_feature_name()` | `cmapss_ml_pipeline.py`, `cmapss_xai_module.py`, автономно (пересборка из CSV) |

---

## Статус проекта: ЗАВЕРШЁН

Проект **полностью готов**: аналитическое ядро (две модели, XAI, RAG, LLM-агент),
runtime-слой, двухуровневый Streamlit UI с двумя источниками данных (демо-датасет +
живой режим реального времени) и подсистема real-time валидации с публикационными
графиками — реализованы, валидированы и работают. Проведена финальная ревизия
кодовой базы: комментарии и docstring'и синхронизированы с кодом, мёртвый код и
неиспользуемые сущности удалены, все доменные подпакеты `src/` оформлены с
`__init__.py`, регрессионные тесты (`online_parity_test.py`,
`protobackend_smoke_test.py`) проходят.

**Дополнительный ключевой результат — валидация на реальных данных:** пайплайн
`preprocessing → ML → XAI` без изменения механики прогнан на run-to-failure датасете
NASA C-MAPSS (все 4 подмножества, два режима оценки, 3 модели) — см. раздел 24.
Holdout PR-AUC(Авария) 0.95±0.02, ни один из 142 тестовых двигателей не пропущен,
SHAP опирается на физику компрессора. Итоговый вывод — 6 сводных фигур
`cmapss_summary_plot1–6` в парадигме «усреднённая картина с отклонениями».

Возможные направления развития за рамками дипломной работы:

1. Подключение реального источника ТОиР-истории вместо демо-данных в инженерной вкладке.
2. Валидация на промышленных данных конкретного предприятия — насосный парк
   (переносимость методологии на реальные данные уже доказана на C-MAPSS;
   открытых промышленных датасетов насосов с разметкой отказов нет).

**Режим инференса в UI (по одной строке):**
```python
tick = backend.process_tick(pump_id, raw_row)      # online-препроцессор + AlarmManager
trigger = fsm.update(pump_id, ts, tick.severity,    # дебаунс-FSM, гейтинг LLM
                     suppressed=tick.suppressed, fault_type=tick.fault_type)
if trigger is not None:                             # подтверждённая эскалация
    sv = backend.explain(pump_id, ts, trigger.stage)
    stream = backend.prescription_stream(sv, STAGE_BY_SEVERITY[trigger.stage])
```

---

## Принятые архитектурные решения (не менять без явного обсуждения)

| Решение | Обоснование |
|---|---|
| AR(1) вместо i.i.d. шума | Физическая "память" сигнала необходима для rolling window и SHAP |
| Три типа отказа A/Б/В (55/30/15%) | Реалистичные сигнатуры; Тип В (электрика) без роста вибрации и температуры — модель вынуждена учиться по токовым признакам |
| `fault_type` сохраняется в processed CSV | В ML не попадает (нет в FEATURE_COLS), но нужен `fault_recall_analysis.py` для доказательства различения сигнатур |
| Отдельные флаги `anomaly_vibration/temperature/current` | Точечные маркеры на пострадавшем датчике без путаницы на графике |
| Group Split по pump_id | Доказывает обобщение на unseen equipment — строгий MLOps-стандарт |
| LOGO CV в `experiments/logo_cv/xgboost_benchmark.py` | Доказывает, что результат не случаен для конкретного тестового насоса |
| Отдельная модель типа отказа вместо SHAP-эвристики | SHAP-пороги «плывут» при переобучении; обучаемый классификатор даёт измеримую точность |
| `build_fault_dataset()` те же `FEATURE_COLS` | Согласует признаковое пространство train/inference для обеих моделей |
| `permutation_test.py` — sanity check | Метки перемешаны → balanced accuracy ~1/3 → доказывает отсутствие паразитных паттернов |
| `artifacts/` отдельно от `data/` | Сгенерированные графики/таблицы/ChromaDB изолированы от входных датасетов |
| `artifacts/tables/` отдельно от `artifacts/graphs/` | CSV-таблицы результатов не смешиваются с PNG-графиками |
| Пакетная раскладка `src/<domain>/` | Доменное разделение (data/ml/xai/rag/agent/runtime/app); явные импорты `src.<domain>.<module>` |
| `models/severity/` + `models/fault_type/` | Иерархия моделей отражена в файловой структуре; имена с префиксом домена |
| `shift(1)` перед `rolling()` | Предотвращает data leakage в признаках |
| `min_periods=w` для mean/max | Честный NaN вместо статистики по 1–2 точкам |
| `diff_30 = shift(1) - shift(31)` | Ровно 30 шагов, консистентно с rolling(30) |
| PR-AUC вместо ROC-AUC | При дисбалансе 95/5% ROC-AUC завышается из-за TN |
| AlarmManager поверх ML | State-based alarming — инженерный первый уровень фильтрации |
| SymptomVector dataclass | Строгий контракт между XAI и агентом (SRP) |
| `abs(shap_weight)` для сортировки | Признак -0.8 важнее +0.1 для диагностики |
| e5-large вместо MiniLM | Технический русский требует более мощной модели |
| FEATURE_COLS явный список | Единый контракт признаков для train и inference |
| `doc_type_map` как единственный allowlist | Нет нужды в `skip_pdfs` — загружаются только перечисленные файлы |
| `config/settings/settings.py` как единственный источник констант | Исключает дублирование порогов, seeds, имён файлов по модулям |
| Визуализация в `src/visualisation/` | SRP: модули данных не зависят от matplotlib; графики редактируются независимо |
| Промпт в `config/prompts/diagnostic_agent.md` | Редактируется без изменения Python-кода; читается при импорте |
| 4-канальный промпт (справочный/оператор/ТОиР/график) | Разделение источников по назначению: модель не смешивает диагноз и предписание |
| Детерминированный вердикт о внеплановом выводе | P(аварии) считается агентом; модель переписывает готовую фразу — не принимает решение сама |
| `_unknown_response()` при `stage='unknown'` | LLM не вызывается при низкой уверенности типа — нет галлюцинаций на неопределённых состояниях |
| `generate_prescription_stream()` для Streamlit | Потоковая отдача токенов: оператор видит ответ по мере генерации, не ждёт полного завершения |
| SOP-чанкинг с fault_type + stage | Позволяет детерминированно фильтровать operator-блоки по типу отказа И стадии |
| `resolve_stage()` в `rag_database.py` (не в `ai_agent.py`) | Единый резолвер стадии; импортируется и агентом, и бенчмарком |
| 6 сценариев в бенчмарке (fault × stage) | Доказывает стадийную дифференциацию, а не только различение типов |
| `stage_appropriate` как метрика | Автоматически проверяет, что на Предупреждение нет останова, на Аварию есть решительное действие |
| `rag_regression_guard_test.py` | Защита от регрессии при перестройке базы; тест запускается перед любым изменением разметки SOP |
| `online_parity_test.py` — регрессия паритета | Контракт online == offline препроцессора заморожен тестом; менять `shift(1)`/`min_periods`/`diff_30`/`ddof` без его прогона запрещено |
| `protobackend_smoke_test.py` — замороженный контракт UI ↔ backend | Методы backend и поля SymptomVector, которые читает UI, закодированы явными списками; ProtoBackend не может молча отстать от боевого |
| `agent_summary_table.csv` читается с `index_col=0` | `model` — индекс DataFrame; без этого хитмап показывает 0,1,2 вместо имён |
| Слой `src/runtime/` между UI и ядром | `app.py` зависит только от `platform_backend`; смена источника данных и сигнатур модулей локализована |
| `PlatformBackend` + `ProtoBackend` (один интерфейс) | UI тестируем и верстаем без Ollama/ChromaDB/моделей; автоматический откат на прототип при сбое инициализации |
| Гейтинг LLM в `PumpAlarmFSM` | Тяжёлая цепочка XAI→RAG→агент (~20 с) запускается раз на подтверждённую эскалацию, а не на каждый тик |
| Дебаунс FSM (`confirm_up=2`, `confirm_down=5`) | Анти-дребезг тревог (ISA 18.2 / EEMUA 191); эскалация подтверждается быстрее деэскалации |
| `AlarmJournal` хранит подавленные сигналы | ФЗ-116 / ГОСТ Р 22.1.12-2005: скрытые сигналы обязаны быть в архиве |
| `OnlinePreprocessor` с контрактом паритета online==offline | Один и тот же `FEATURE_COLS` в train и realtime; расхождение ловится `verify_parity` |
| Демо-сценарий вырезается из датасета, не генерится заново | Воспроизводимость + честность: те же unseen-данные MNHV_005, на которых валидированы модели |
| Фоновый поток генерации предписания (`get_gen_store` через `@st.cache_resource`) | Поток данных и графики не «замерзают» на ~13 с генерации RAG+LLM; store переживает rerun'ы Streamlit (модульные globals и `session_state` для этого не годятся) |
| Поток данных не останавливается на эскалации | `SymptomVector` считается синхронно (факт инцидента фиксируется сразу), текст дописывается в фоне — быстрый переход Предупреждение→Авария не теряет предупреждение |
| Мультистраничная навигация (`st.Page`/`st.navigation`) вместо `ss.role` | Реальная изоляция поддеревьев Оператор/Инженер; элементы одного экрана не протекают на другой |
| Перерисовка через `@st.fragment(run_every=REFRESH)`, без глобального `time.sleep`/`st.rerun` | Глобальный rerun перебивал фрагменты, съедал клики (квитировать/развернуть), кидал экран наверх и двоил тики |
| История из журнала `ss.events` (append-only, уникальный `eid`) | Устойчивость: не зависит от мутаций `stage_ts` инцидента в FSM; хрупкая склейка «близнецов» по временно́му окну больше не нужна |
| Ре-наг = НОВАЯ строка истории (а не мутация старой) | Квитированное событие остаётся завершённым; повторное возникновение через `RENAG_MIN` — отдельная строка с тем же предписанием; ре-наг только если инцидент всё ещё текущий для агрегата |
| `RealtimeProgressivePreprocessor` для живого режима | Признаки с 1-й минуты (нет «слепого окна»); паритет бит-в-бит со строгим препроцессором после 60-й строки сохранён |
| `LiveMultiPumpGenerator` повторяет формулы `data_generator.py` один-в-один | Живой режим валиден на той же физике, что и обучение; квитирование как обратная связь (50% восстановление / 100% останов) |
| `RealtimePlayer` повторяет интерфейс `ScenarioPlayer` | Живой режим встаёт на место демо-плеера без переписывания UI (advance_stream/тосты/графики как есть) |
| Мёртвая зона после пуска (`_deadzone`) + сброс окна на OFF/STARTUP | Пусковой ток/гидроудар не попадают в скоринг — переходник любой длины вырезан, а не первые 5 строк |
| Путь к CSV резолвится от `_PROJECT_ROOT` | Сборка демо-сценария не зависит от рабочей директории, из которой запущен `streamlit run` |
| Ветка C-MAPSS в ОТДЕЛЬНЫХ модулях (`cmapss_*`) + `settings_cmapss.py` | Насосный код не тронут; переработка = новый модуль, а не сотни строк в существующем |
| `CmapssPreprocessor` наследует `DataPreprocessor` | Переиспользование механики `shift(1)`/`groupby`/`min_periods` буквально: доказательство переносимости, а не переписывание |
| Piecewise-метки тяжести из RUL (50/20 циклов) | Стандартный подход литературы C-MAPSS; ровно та же трёхклассовая постановка, что у насосов |
| Два режима оценки: official + holdout по двигателям | Official сравним с литературой, но test обрезан (классы 1/2 редки); holdout доказывает, что метрики не артефакт дисбаланса |
| Per-режимный z-score для FD002/FD004 (статистики только по train) | 6 полётных режимов маскируют тренд деградации; нормализация внутри режима восстанавливает сигнал без утечки из test |
| Датасет C-MAPSS не в git, загрузка из сети в оркестраторе | Репозиторий не тяжелеет; прогон воспроизводим с чистого клона без ручной подготовки |
| Сводный слой по умолчанию, детализация по `--detail` | «Брать качеством, а не числом»: комиссии — 5 фигур mean±std, россыпь пер-сабсетных графиков — материал приложения |
| Таблицы C-MAPSS обновляются слиянием (`merge_update_csv`) | Прогон одного сабсета заменяет только свои строки — частный перезапуск не затирает общую картину |
| Наложенный beeswarm через перцентильные ранги внутри сабсета | Сырые (FD001/003) и z-нормированные (FD002/004) шкалы признаков нельзя смешивать в цветовой кодировке SHAP |

---

## Критические ограничения (задокументированы в дипломе)

1. Все 5 насосов имеют **идентичные номинальные параметры** — исследование разнотипного оборудования выходит за рамки работы
2. ГОСТ 32601-2013 (307 стр.) и мануал МНХВ **не загружаются как PDF** — заменены вручную подготовленными `gost_extract.md` и `mnhv_extract.md`
3. LLM работает **локально на M2** (Ollama) — зависимости от внешних API нет; сравниваются три модели: `qwen3.5:9b`, `phi4:14b`, `second_constantine/yandex-gpt-5-lite:8b`
4. Валидация на реальных данных выполнена на **NASA C-MAPSS** (турбовентиляторные двигатели) — доказана переносимость методологии на другой класс оборудования; открытых промышленных датасетов насосов с размеченными run-to-failure отказами не существует, данные конкретного предприятия — за рамками работы

---

## Технологический стек

| Компонент | Технология |
|---|---|
| Язык | Python 3.9+ (среда: 3.9.6) |
| ML | scikit-learn, XGBoost |
| XAI | shap (TreeExplainer) |
| RAG / Embeddings | LangChain, ChromaDB, intfloat/multilingual-e5-large |
| LLM | Ollama (локально, macOS M2) |
| PDF парсинг | pymupdf4llm, pdfplumber |
| UI | Streamlit (интерактивные графики — Plotly) |
| Визуализация | matplotlib, seaborn, plotly |
| Данные | pandas, numpy |

---

## Нормативная база (для контекста при работе с документами)

- **ГОСТ 32601-2013** — пороги вибрации и температуры подшипников
- **ANSI/ISA-18.2** — управление аварийными сигналами (≤150 аварий/день)
- **NAMUR NE 107** — цветовые статусы оборудования (UI)
- **NAMUR NE 129** — двухуровневый принцип HMI (оператор / инженер)
- **ФЗ № 116-ФЗ** — промышленная безопасность (скрытые сигналы хранятся в архиве)