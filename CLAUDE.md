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

```
predictive_alarm_platform/
│
├── src/
│   ├── data_generator.py           # ГОТОВО — AR(1) + State Machine + 3 типа отказа (A/Б/В)
│   ├── data_preprocessor.py        # ГОТОВО — rolling features, build_fault_dataset()
│   ├── ml_pipeline.py              # ГОТОВО — модель ТЯЖЕСТИ: LR / RF / XGBoost, AlarmManager
│   ├── fault_classifier_pipeline.py# ГОТОВО — модель ТИПА отказа: overheat/cavitation/electrical
│   ├── fault_recall_analysis.py    # ГОТОВО — recall по типам отказа, доказательство 3 сигнатур
│   ├── xai_module.py               # ГОТОВО — SHAP TreeExplainer, SymptomVector, SHAP-эвристика
│   ├── rag_database.py             # ГОТОВО — ChromaDB + multilingual-e5-large, PDF→чанки
│   ├── ai_agent.py                 # ГОТОВО — DiagnosticAgent (Ollama), сравнение 3 моделей
│   └── app.py                      # НЕ НАПИСАН — Streamlit двухуровневый UI
│
├── scripts/
│   ├── xgboost_benchmark.py        # ГОТОВО — LOGO CV (5 фолдов), доказательство обобщения
│   ├── ai_agent_benchmark.py       # ГОТОВО — многосценарный бенчмарк 3 LLM-моделей
│   └── permutation_test.py         # ГОТОВО — sanity check: метки перемешаны → accuracy ~1/3
│
├── visualisation_instruments/
│   ├── __init__.py              # Реэкспорт всех функций визуализации
│   ├── simulation_visualisation.py  # Графики генератора (plot_smart_episode)
│   ├── ml_visualisation.py          # Графики ML (confusion, metrics, PR, recall)
│   ├── rag_visualisation.py         # Графики RAG (состав базы, чанки, качество поиска)
│   └── xai_visualisation.py         # Графики XAI (waterfall, beeswarm по типам отказа)
│
├── config/
│   ├── __init__.py
│   ├── prompts/
│   │   └── diagnostic_agent.md  # Системный промпт LLM-агента (anti-hallucination)
│   └── settings/
│       ├── __init__.py
│       └── settings.py          # Централизованная конфигурация (пороги, seeds, RAG, LLM)
│
├── data/
│   ├── raw/
│   │   └── industrial_pumps_dataset.csv        # 648 000 строк, 5 насосов, 90 дней, шаг 1 мин
│   ├── processed/
│   │   ├── preprocessed_pumps_dataset.csv      # 40 rolling-признаков, target 0/1/2, fault_type
│   │   └── fault_type_pumps_dataset.csv        # выборка для классификатора типа (target=1&2)
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
│   │   ├── shap_plot1_waterfall_MNHV_005.png         # Severity-модель: waterfall
│   │   ├── shap_plot2_beeswarm_overheat.png        # Severity-модель: beeswarm
│   │   ├── shap_plot3_beeswarm_cavitation.png
│   │   ├── shap_plot4_beeswarm_electrical.png
│   │   ├── shap_fault_plot1_waterfall_MNHV_005.png # Fault-классификатор: waterfall
│   │   ├── shap_fault_plot2_beeswarm_overheat.png  # Fault-классификатор: beeswarm
│   │   ├── shap_fault_plot3_beeswarm_cavitation.png
│   │   ├── shap_fault_plot4_beeswarm_electrical.png
│   │   ├── rag_plot1-4_*.png
│   │   ├── agent_plot1-3_*.png
│   │   └── agent_benchmark_*.csv
│   └── tables/                                  # CSV-таблицы результатов
│       ├── ML_fault_recall_table.csv
│       └── ML_fault_classifier_summary.csv
│
├── models/
│   ├── lr_pump_model.joblib                     # Модель тяжести: LR
│   ├── rf_pump_model.joblib                     # Модель тяжести: RF
│   ├── xgboost_pump_model.joblib                # Модель тяжести: XGBoost (основная)
│   ├── fault_lr_model.joblib                    # Классификатор типа: LR
│   ├── fault_rf_model.joblib                    # Классификатор типа: RF
│   └── fault_xgboost_model.joblib               # Классификатор типа: XGBoost
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
├── chroma_db/                         # Локальная векторная БД ChromaDB
├── requirements.txt
└── CLAUDE.md                          # ← этот файл
```

---

## Что уже реализовано (детально)

### 1. `data_generator.py` — Генератор данных
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
- Визуализация вынесена в `visualisation_instruments/simulation_visualisation.py`; вызов: `plot_smart_episode(df, ...)` через импорт из `visualisation_instruments`
- Выход: `data/raw/industrial_pumps_dataset.csv`

### 2. `data_preprocessor.py` — Подготовка признаков
- **Rolling features** (15, 30, 60 мин): mean, std, max для каждого из 4 сенсоров
- **Gradient:** `shift(1) - shift(31)` — ровно 30 шагов, консистентно с rolling(30)
- **Data leakage защита:** `shift(1)` перед каждым `.rolling()` — в строке T используется только [T-W … T-1]
- **`groupby('pump_id')`:** окна не перетекают между насосами; `min_periods=w` для mean/max
- Флаги `anomaly_*` удаляются ДО подачи в ML; `fault_type` намеренно сохраняется
- **`FEATURE_COLS`** — явный контракт из 40 признаков для train и inference
- Пороги `VIB_*/TEMP_*`, `WINDOW_SIZES`, `FAULT_TYPES` импортируются из `config/settings/settings.py`
- **`build_fault_dataset(df_processed)`** — новый метод: формирует выборку для классификатора типа из строк `target==1` и `target==2`; добавляет `fault_target` (0=overheat, 1=cavitation, 2=electrical) и `severity_stage`; те же `FEATURE_COLS` без пересчёта
- Выходы: `preprocessed_pumps_dataset.csv` + `fault_type_pumps_dataset.csv`

### 3. `ml_pipeline.py` — Обучение моделей
- **Group Split:** train = MNHV_001–004, test = MNHV_005 (из `config/settings`: `TRAIN_PUMPS`, `TEST_PUMPS`)
- Три модели: Logistic Regression (baseline), Random Forest (n=300), XGBoost (n=300, lr=0.1, depth=6)
- Балансировка: `class_weight='balanced'` + `sample_weight` через `compute_sample_weight`
- **Метрики:** Confusion Matrix, F1-Macro, Recall(Авария) как KPI, PR-AUC
- **Результаты XGBoost:** F1-Macro=0.97, Recall(Critical)=0.949, PR-AUC(Critical)=0.990
- **`AlarmManager`:** если `raw_state in [0, 1]` → принудительно возвращает 0 (Alarm Shelving)
- Визуализация вынесена в `visualisation_instruments/ml_visualisation.py`; вызов `plot_all(metrics, y_test, output_dir)` — графики сохраняются с префиксом `ML_`
- **Шаг 10** в `__main__`: вызов `fault_recall_analysis.analyze_fault_recall(xgb_model, df_test, ...)`

### 4. `fault_recall_analysis.py` — Валидация по типам отказа
- **Цель:** доказать, что модель ТЯЖЕСТИ различает три физических сценария, а не работает по «всё выросло → авария»
- **`analyze_fault_recall(model, df_test, feature_cols, save_graphs_dir, save_tables_dir)`** — вызывается из `ml_pipeline.py` (шаг 10) и автономно
- **Recall(Critical):** доля строк `state=4` данного `fault_type`, предсказанных как класс 2
- **Recall(Warning):** доля строк `state=3` данного `fault_type`, предсказанных как ≥ 1 (угроза не пропущена)
- **Fallback:** если `fault_type` отсутствует в processed CSV — присоединяется из raw по `[timestamp, pump_id]`
- Визуализация вынесена в `visualisation_instruments/ml_visualisation.py` (функция `recall_plot`)
- **Выходы:** `data/graphs/ML_plot4_fault_recall_analysis.png`, `data/tables/ML_fault_recall_table.csv`

### 5. `fault_classifier_pipeline.py` — Классификатор типа отказа (вторая модель)
- **Второй этап иерархической классификации:** `ml_pipeline.py` определяет ТЯЖЕСТЬ (0/1/2); этот модуль определяет ФИЗИЧЕСКИЙ ТИП (overheat / cavitation / electrical)
- **Зачем отдельная модель:** SHAP-эвристика в `xai_module.py` давала точность ~0.61 из-за отсутствия физического смысла у абсолютных порогов SHAP; обучаемый классификатор даёт измеримую, защищаемую точность
- Вход: `fault_type_pumps_dataset.csv` (строки с `target==1 & 2`, `fault_target` 0/1/2)
- **Group Split:** train = MNHV_001–004, test = MNHV_005; `class_weight='balanced'`
- Три модели: LR, RF, XGBoost (n=300, lr=0.1, depth=6, `mlogloss`)
- **Метрики:** macro-F1, balanced accuracy, per-class recall, раздельно по стадии (Warning / Critical) через `severity_stage`
- Визуализация через `visualisation_instruments/ml_visualisation.py` (функция `plot_fault_classifier`)
- **Выходы:**
  - Модели: `fault_lr_model.joblib`, `fault_rf_model.joblib`, `fault_xgboost_model.joblib`
  - Графики: `ML_fault_plot1_confusion_matrix.png`, `ML_fault_plot2_model_comparison.png`, `ML_fault_plot3_stage_split.png`
  - Таблица: `data/tables/ML_fault_classifier_summary.csv`

### 6. `xai_module.py` — Объяснимый ИИ (SHAP)
- **`XAIExplainer`:** `shap.TreeExplainer` для XGBoost модели ТЯЖЕСТИ, `target_class_idx=2` (Авария)
- Возвращает **`SymptomVector`** (dataclass): `predicted_class`, `probabilities`, `critical_probability`, `top_symptoms`, `shap_base_value`, `inferred_fault`, `true_fault`
- Сортировка по `abs(shap_weight)` — признак -0.8 важнее признака +0.1
- **`_infer_fault_type(contributions)`** — SHAP-эвристика определения типа отказа по долям положительного вклада датчиков (масштабонезависима): температура≥40% → overheat; ток≥40% при молчащих остальных → electrical; иначе cavitation. Точность валидируется `validate_fault_inference()`
- Пороги ГОСТ из `config/settings/settings.py`
- Визуализация вынесена в `visualisation_instruments/xai_visualisation.py`:
  - `plot_waterfall()`, `plot_summary_by_fault_type()` → **severity-модель** (`shap_plot1-4_*`)
  - `plot_fault_waterfall()`, `plot_fault_summary_by_type()` → **fault-классификатор** (`shap_fault_plot1-4_*`)
- **Выходы:** `shap_plot1-4_*` (severity SHAP) + `shap_fault_plot1-4_*` (fault-classifier SHAP)

### 7. `rag_database.py` — База знаний (RAG)
- **Стек:** LangChain + ChromaDB (локально) + `intfloat/multilingual-e5-large` (MPS, M2)
- **`StructuredPDFLoader`:** pymupdf4llm → Markdown (сохраняет таблицы), fallback → pdfplumber
- **`TextKnowledgeLoader`:** загрузка `.md`/`.txt` файлов
- **`doc_type_map` — единственный allowlist:** загружаются только файлы, явно перечисленные в нём (из `config/settings`)
- Активный `doc_type_map` (все файлы — `.md`): `tm_regulation.md` (sop), `tm_schedule.md` (schedule), `gost_extract.md` (gost), `mnhv_extract.md` (manual), `diagnostics_extended.md` (diagnostics); PDF-версии в `knowledge_base/` хранятся как исходники, но в ChromaDB не загружаются
- **`StructuredPDFLoader`** оставлен как расширение для будущих PDF; в текущей версии не используется
- **`doc_type`:** manual / gost / sop / schedule / diagnostics; метаданные: `source`, `section`, `chunk_id`
- **Разные chunk_size по типу:** gost=700, manual=600, sop=550, schedule=350, diagnostics=500
- **e5 prefix:** `passage:` при загрузке, `query:` при поиске — обязательно для e5-large
- **Порог релевантности:** `RELEVANCE_THRESHOLD=1.2` (L2) из `config/settings`
- Батчевая запись по 500 чанков; `DOC_TYPES`, `CHUNK_CONFIG`, `RELEVANCE_THRESHOLD`, `EMBED_MODEL`, `DOC_TYPE_MAP` импортируются из `config/settings/settings.py`
- Визуализация вынесена в `visualisation_instruments/rag_visualisation.py`; вызов: `plot_all_rag(kb.chroma_dir, kb.embeddings, kb.EMBED_MODEL, test_queries, plots_dir)` в `__main__`
- **Выходы:** `rag_plot1_kb_composition.png`, `rag_plot2_chunk_distribution.png`, `rag_plot3_retrieval_quality.png`, `rag_plot4_fault_coverage_heatmap.png`

### 8. `ai_agent.py` — LLM-агент
- **`DiagnosticAgent`** — принимает `SymptomVector` + RAG-контекст, строит промпт, вызывает Ollama
- **`AgentResponse`** dataclass: `raw_text`, `model_name`, `latency_sec`, `gen_time_sec`, `eval_count`, `tokens_per_sec`, `format_ok`, `error`
- **`SYSTEM_PROMPT`** загружается из `config/prompts/diagnostic_agent.md` (anti-hallucination, строгий формат: СТАТУС / ДИАГНОЗ / ПРЕДПИСАНИЕ / ТОиР)
- Qwen 3.x: `think=False` через нативный Ollama-клиент отключает chain-of-thought
- **`__main__`:** цепочка XAI → RAG → LLM с последовательным прогоном трёх моделей
- Три тестируемые модели: `qwen3.5:9b` (default), `phi4:14b`, `second_constantine/yandex-gpt-5-lite:8b`
- Вывод: `ОТВЕТ АГЕНТА N (модель: ..., время: ... с)` для каждой модели

### 9. `scripts/xgboost_benchmark.py` — LOGO CV бенчмарк
- **Leave-One-Group-Out Cross-Validation:** 5 фолдов, каждый раз один насос — тест, остальные 4 — обучение
- **Цель:** доказать, что XGBoost обобщается на любой новый агрегат парка (не переобучен на MNHV_005)
- Метрики: F1-Macro, Recall(Critical), PR-AUC по каждому фолду
- Визуализация: `ML_plot5_logo_cv_comparison.png`
- Импортирует из `src.ml_pipeline` (функция `evaluate_model`)

### 10. `scripts/ai_agent_benchmark.py` — Бенчмарк LLM-агента
- **Три диагностических сценария:** Тип А (перегрев), Тип Б (кавитация), Тип В (электрика) — проверяет, различает ли LLM типы отказов
- **Метрики:** скорость генерации (tokens/sec), соответствие формату, атрибуция источника, автоматическая оценка качества
- Усреднение по 3 сценариям × 3 повтора = 9 прогонов на модель
- **Выходы:** `agent_benchmark_raw.csv`, `agent_benchmark_multi.csv`, `agent_summary_table.csv`, `agent_plot1-3_*.png`

### 11. `scripts/permutation_test.py` — Sanity check классификатора типа
- Загружает `fault_type_pumps_dataset.csv`, перемешивает `fault_target` случайным образом (`rng(42)`)
- Обучает XGBoost на перемешанных метках, проверяет balanced accuracy на тесте — должна упасть к ~1/3
- **Цель:** доказать, что модель учится реальным сигнатурам, а не паразитным паттернам в данных

---

## Централизованная конфигурация (`config/`)

### `config/settings/settings.py`
Единственный источник истины для всех константных значений проекта.
Импортируется во все модули `src/` и `scripts/`:

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
| `LLM_MODELS`, `DEFAULT_LLM_MODEL` | Список тестируемых Ollama-моделей |

### `config/prompts/diagnostic_agent.md`
Системный промпт `DiagnosticAgent`. Загружается при импорте `ai_agent.py`.
Редактируется без изменения Python-кода.

---

## Визуализация (`visualisation_instruments/`)

Все функции реэкспортируются через `__init__.py` — импорт: `from visualisation_instruments import plot_all`.

| Модуль | Функции | Откуда вызывается |
|---|---|---|
| `simulation_visualisation.py` | `plot_smart_episode(df, hours, save_dir)` | `data_generator.py` |
| `ml_visualisation.py` | `plot_all(metrics, y_test, output_dir)`, `recall_plot(results, signatures, save_dir)`, `plot_fault_classifier(metrics, output_dir)` | `ml_pipeline.py`, `fault_recall_analysis.py`, `fault_classifier_pipeline.py` |
| `rag_visualisation.py` | `plot_all_rag(chroma_dir, embeddings, model, queries, save_dir)` | `rag_database.py` |
| `xai_visualisation.py` | `plot_waterfall()`, `plot_summary_by_fault_type()` — severity; `plot_fault_waterfall()`, `plot_fault_summary_by_type()` — fault classifier | `xai_module.py` |

---

## Что предстоит сделать

### `app.py` — Streamlit UI
Двухуровневый интерфейс по стандарту NAMUR NE 129:

**Уровень 1 — Operator Dashboard:**
- Агрегированный статус парка (цвета NAMUR NE 107: зелёный / жёлтый / красный / серый)
- Счётчик активных тревог, лист отложенных (Alarm Shelving stash)
- Пошаговая инструкция от агента при аварии

**Уровень 2 — Engineer Tab:**
- Графики временных рядов с порогами
- SHAP Waterfall plot (из `xai_visualisation.plot_waterfall()`)
- SHAP Beeswarm plot (из `xai_visualisation.plot_summary_by_fault_type()`)
- История инцидентов, bad actors

**Режим инференса:**
```python
last_61_rows = df[df['pump_id'] == pump_id].tail(61)
features = preprocessor.process(last_61_rows, is_training=False)
feature_row = features.iloc[-1:][preprocessor.FEATURE_COLS]
raw_state = last_61_rows.iloc[-1]['state']
prediction = alarm_manager.predict_with_context(feature_row, raw_state)
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
| LOGO CV в `scripts/xgboost_benchmark.py` | Доказывает, что результат не случаен для конкретного тестового насоса |
| Отдельная модель типа отказа вместо SHAP-эвристики | SHAP-пороги «плывут» при переобучении; обучаемый классификатор даёт измеримую точность |
| `build_fault_dataset()` те же `FEATURE_COLS` | Согласует признаковое пространство train/inference для обеих моделей |
| `permutation_test.py` — sanity check | Метки перемешаны → balanced accuracy ~1/3 → доказывает отсутствие паразитных паттернов |
| `data/tables/` отдельно от `data/graphs/` | CSV-таблицы результатов не смешиваются с PNG-графиками |
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
| Визуализация в `visualisation_instruments/` | SRP: модули данных не зависят от matplotlib; графики редактируются независимо |
| Промпт в `config/prompts/diagnostic_agent.md` | Редактируется без изменения Python-кода; читается при импорте |

---

## Критические ограничения (задокументированы в дипломе)

1. Все 5 насосов имеют **идентичные номинальные параметры** — исследование разнотипного оборудования выходит за рамки работы
2. ГОСТ 32601-2013 (307 стр.) и мануал МНХВ **не загружаются как PDF** — заменены вручную подготовленными `gost_extract.md` и `mnhv_extract.md`
3. LLM работает **локально на M2** (Ollama) — зависимости от внешних API нет; сравниваются три модели: `qwen3.5:9b`, `phi4:14b`, `second_constantine/yandex-gpt-5-lite:8b`

---

## Технологический стек

| Компонент | Технология |
|---|---|
| Язык | Python 3.11+ |
| ML | scikit-learn, XGBoost |
| XAI | shap (TreeExplainer) |
| RAG / Embeddings | LangChain, ChromaDB, intfloat/multilingual-e5-large |
| LLM | Ollama (локально, macOS M2) |
| PDF парсинг | pymupdf4llm, pdfplumber |
| UI | Streamlit |
| Визуализация | matplotlib, seaborn |
| Данные | pandas, numpy |

---

## Нормативная база (для контекста при работе с документами)

- **ГОСТ 32601-2013** — пороги вибрации и температуры подшипников
- **ANSI/ISA-18.2** — управление аварийными сигналами (≤150 аварий/день)
- **NAMUR NE 107** — цветовые статусы оборудования (UI)
- **NAMUR NE 129** — двухуровневый принцип HMI (оператор / инженер)
- **ФЗ № 116-ФЗ** — промышленная безопасность (скрытые сигналы хранятся в архиве)