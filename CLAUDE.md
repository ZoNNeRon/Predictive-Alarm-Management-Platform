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
│   ├── data_generator.py        # ГОТОВО — AR(1) + State Machine + 3 типа отказа (A/Б/В)
│   ├── data_preprocessor.py     # ГОТОВО — rolling features, защита от data leakage
│   ├── ml_pipeline.py           # ГОТОВО — обучение LR / RF / XGBoost, AlarmManager
│   ├── fault_recall_analysis.py # ГОТОВО — recall по типам отказа, доказательство 3 сигнатур
│   ├── xai_module.py            # ГОТОВО — SHAP TreeExplainer, SymptomVector dataclass
│   ├── rag_database.py          # ГОТОВО — ChromaDB + multilingual-e5-large, PDF→чанки + TextKnowledgeLoader
│   ├── ai_agent.py              # ГОТОВО — DiagnosticAgent (Ollama), сравнение 3 моделей
│   └── app.py                   # НЕ НАПИСАН — Streamlit двухуровневый UI
│
├── data/
│   ├── raw/
│   │   └── enterprise_pump_fleet.csv   # 432 000 строк, 5 насосов, 60 дней, шаг 1 мин
│   ├── processed/
│   │  └── processed_features.csv      # 40 rolling-признаков, target 0/1/2
│   └── graphs/		# для хранения создаваемых кодом графиков
│
├── models/ 		# Обученные модели
│   └── ...
│
├── knowledge_base/
│   ├── gosts/
│   │   ├── gost_32601_2013.pdf		# ГОСТ 32601 (307 стр., НЕ загружается в БД)
│   │   └── gost_extract.md 		# ГОТОВО — выжимка: раздел 6.9, Таблицы 8-9, пороги температуры
│   ├── manuals/
│   │   ├── mnhv_manual.pdf 		# Мануал МНХВ (НЕ загружается — ЕСКД-штампы)
│   │   └── mnhv_extract.md 		# ГОТОВО — очищенный мануал: параметры, Таблица 4, ТО
│   ├── regulations/
│   │   └── tm_regulation.pdf           # Регламент ТО (sop)
│   ├── schedules/
│   │   └── tm_schedule.pdf             # График ППР (schedule)
│   └── diagnostics/
│       └── diagnostics_extended.md     # Расширенная вибродиагностика (doc_type: diagnostics)
│
├── chroma_db/                          # Локальная векторная БД ChromaDB
│
└── CLAUDE.md                           # ← этот файл
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
- **5 насосов** (`MNHV_001`–`MNHV_005`), `PUMP_SEEDS` — явный словарь seed'ов (42, 137, 2025, 31415, 99991)
- **432 000 строк**, 60 дней, шаг 1 минута; схема: `[timestamp, pump_id, state, state_name, fault_type, vibration, temperature, current, pressure, anomaly_vibration, anomaly_temperature, anomaly_current]`
- График `plot_smart_episode` — 5 панелей (vib/temp/curr/pressure/state), фоновая подсветка по `fault_type`, точечные маркеры помех только на пострадавшем датчике
- Выход: `data/raw/enterprise_pump_fleet.csv`

### 2. `data_preprocessor.py` — Подготовка признаков
- **Rolling features** (15, 30, 60 мин): mean, std, max для каждого из 4 сенсоров
- **Gradient:** `shift(1) - shift(31)` — ровно 30 шагов, консистентно с rolling(30)
- **Data leakage защита:** `shift(1)` перед каждым `.rolling()` — в строке T используется только [T-W … T-1]
- **`groupby('pump_id')`:** окна не перетекают между насосами
- `min_periods=w` для mean/max (честный NaN), `min_periods=2` для std
- Флаги `anomaly_vibration`, `anomaly_temperature`, `anomaly_current` удаляются ДО подачи в ML
- **`fault_type` намеренно сохраняется** — не входит в `FEATURE_COLS`, но нужен `fault_recall_analysis.py` для валидации
- **`FEATURE_COLS`** — явный контракт из 40 признаков, используется и в train, и в inference
- Фильтрация Off (0) и Startup (1) из обучающей выборки — ими занимается `AlarmManager`
- Выход: `data/processed/processed_features.csv` (содержит `fault_type`, `state`, `pump_id`, `timestamp` + 40 rolling-признаков)

### 3. `ml_pipeline.py` — Обучение моделей
- **Group Split:** train = MNHV_001–004 (321 615 строк), test = MNHV_005 (80 531 строки)
- Три модели: Logistic Regression (baseline), Random Forest (n=300), XGBoost (n=300, lr=0.1, depth=6)
- Балансировка: `class_weight='balanced'` + `sample_weight` через `compute_sample_weight`
- **Метрики:** Confusion Matrix, F1-Macro, Recall(Авария) как KPI, PR-AUC
- **Результаты XGBoost:** F1-Macro=0.97, Recall(Critical)=0.948, PR-AUC(Critical)=0.993
- **`AlarmManager`:** если `raw_state in [0, 1]` → принудительно возвращает 0 (Alarm Shelving)
- Три визуализации: confusion matrices heatmap, grouped bar chart, PR-кривые
- **Шаг 10** в `__main__`: вызов `fault_recall_analysis.analyze_fault_recall(xgb_model, df_test, ...)`

### 4. `fault_recall_analysis.py` — Валидация по типам отказа
- **Цель:** доказать, что модель различает три физических сценария, а не работает по «всё выросло → авария»
- **`analyze_fault_recall(model, df_test, feature_cols, save_dir)`** — основная функция, вызывается из `ml_pipeline.py` (шаг 10) и автономно
- **Recall(Critical):** доля строк `state=4` данного `fault_type`, предсказанных как класс 2
- **Recall(Warning):** доля строк `state=3` данного `fault_type`, предсказанных как ≥ 1 (угроза не пропущена)
- **Fallback:** если `fault_type` отсутствует в `processed_features.csv` — присоединяется из `enterprise_pump_fleet.csv` по `[timestamp, pump_id]`
- **Выходы:** `plot4_fault_recall_analysis.png` (grouped bar chart + тепловая карта сигнатур датчиков), `fault_recall_table.csv`

### 5. `xai_module.py` — Объяснимый ИИ (SHAP)
- **`XAIExplainer`:** `shap.TreeExplainer` для XGBoost, `target_class_idx=2` (Авария)
- Возвращает **`SymptomVector`** (dataclass) — строго типизированный контракт для агента
- Сортировка по `abs(shap_weight)` — признак -0.8 важнее признака +0.1
- Каждый `SymptomContribution` содержит: `sensor`, `window`, `value`, `shap_weight`, `direction`, пороги ГОСТ
- **Визуализация:** `plot_waterfall()` (локальное объяснение) + `plot_summary()` (beeswarm, глобальная важность)
- `generate_llm_prompt` **отсутствует** в этом модуле — намеренно (SRP). Промпт строит `agent_module.py`

### 6. `rag_database.py` — База знаний (RAG)
- **Стек:** LangChain + ChromaDB (локально) + `intfloat/multilingual-e5-large` (MPS, M2)
- **`StructuredPDFLoader`:** pymupdf4llm → Markdown (сохраняет таблицы), fallback → pdfplumber
- **`TextKnowledgeLoader`:** загрузка `.md`/`.txt` файлов (встроен в тот же модуль)
- **`doc_type_map` — единственный allowlist:** загружаются только файлы, явно перечисленные в нём; `skip_pdfs` удалён — лишняя фильтрация
- Активный `doc_type_map`: `tm_regulation.pdf` (sop), `tm_schedule.pdf` (schedule), `gost_extract.md` (gost), `mnhv_extract.md` (manual), `diagnostics_extended.md` (diagnostics)
- **`doc_type`:** manual / gost / sop / schedule / diagnostics; метаданные: `source`, `section`, `chunk_id`
- **Разные chunk_size по типу:** gost=700, manual=600, sop=500, schedule=350, diagnostics=500
- **e5 prefix:** `passage:` при загрузке, `query:` при поиске — обязательно для e5-large
- **Порог релевантности:** `RELEVANCE_THRESHOLD=1.2` (L2), нерелевантные результаты отсекаются
- Батчевая запись по 500 чанков; фильтрация по `doc_type` через `search_kwargs`
- **Текущее состояние базы:** ~59 чанков (17 gost + 17 manual + 11 sop + 10 schedule + diagnostics) — всё релевантный контент
- Три визуализации: `rag_plot1_kb_composition.png` (пай сортирован по убыванию, `startangle=90`), `rag_plot2_chunk_distribution.png`, `rag_plot3_retrieval_quality.png` (оба графика синхронизированы по порядку меток)

---

## Что уже реализовано (продолжение)

### 7. `ai_agent.py` — LLM-агент (ГОТОВО)
- **`DiagnosticAgent`** — принимает `SymptomVector` + RAG-контекст, строит промпт, вызывает Ollama
- **`AgentResponse`** dataclass: `raw_text`, `model_name`, `latency_sec`, `gen_time_sec`, `eval_count`, `tokens_per_sec`, `format_ok`, `error`
- **`SYSTEM_PROMPT`** — строгий anti-hallucination шаблон (СТАТУС / ДИАГНОЗ / ПРЕДПИСАНИЕ / ТОиР)
- Qwen 3.x: `think=False` через нативный Ollama-клиент отключает chain-of-thought
- **`__main__`:** цепочка XAI → RAG → LLM с последовательным прогоном трёх моделей
- Три тестируемые модели: `qwen3.5:9b` (default), `phi4:14b`, `second_constantine/yandex-gpt-5-lite:8b`
- Вывод: `ОТВЕТ АГЕНТА N (модель: ..., время: ... с)` для каждой модели

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
- SHAP Waterfall plot (из `xai_module.plot_waterfall()`)
- SHAP Beeswarm plot (из `xai_module.plot_summary()`)
- История инцидентов, bad actors

**Режим инференса:**
```python
# Правильный паттерн для real-time предсказания
last_61_rows = df[df['pump_id'] == pump_id].tail(61)
features = preprocessor.process(last_61_rows, is_training=False)
feature_row = features.iloc[-1:][preprocessor.FEATURE_COLS]
raw_state = last_61_rows.iloc[-1]['state']
prediction = alarm_manager.predict_with_context(feature_row, raw_state)
```

### Финальный этап: `3.6. Валидация и анализ Use Cases`
- 3 сценария: истинная авария / ложное срабатывание при запуске / нормальная работа
- Финальный скриншот UI для диплома

---

## Принятые архитектурные решения (не менять без явного обсуждения)

| Решение | Обоснование |
|---|---|
| AR(1) вместо i.i.d. шума | Физическая "память" сигнала необходима для rolling window и SHAP |
| Три типа отказа A/Б/В (55/30/15%) | Реалистичные сигнатуры; Тип В (электрика) без роста вибрации и температуры — модель вынуждена учиться по токовым признакам |
| `fault_type` сохраняется в processed CSV | В ML не попадает (нет в FEATURE_COLS), но нужен `fault_recall_analysis.py` для доказательства различения сигнатур |
| Отдельные флаги `anomaly_vibration/temperature/current` | Точечные маркеры на пострадавшем датчике без путаницы на графике |
| Group Split по pump_id | Доказывает обобщение на unseen equipment — строгий MLOps-стандарт |
| `shift(1)` перед `rolling()` | Предотвращает data leakage в признаках |
| `min_periods=w` для mean/max | Честный NaN вместо статистики по 1–2 точкам |
| `diff_30 = shift(1) - shift(31)` | Ровно 30 шагов, консистентно с rolling(30) |
| PR-AUC вместо ROC-AUC | При дисбалансе 95/5% ROC-AUC завышается из-за TN |
| AlarmManager поверх ML | State-based alarming — инженерный первый уровень фильтрации |
| SymptomVector dataclass | Строгий контракт между XAI и агентом (SRP) |
| abs(shap_weight) для сортировки | Признак -0.8 важнее +0.1 для диагностики |
| e5-large вместо MiniLM | Технический русский требует более мощной модели |
| FEATURE_COLS явный список | Единый контракт признаков для train и inference |
| `doc_type_map` как единственный allowlist | Нет нужды в `skip_pdfs` — загружаются только перечисленные файлы |

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
| LLM | Ollama + Phi3 14b (локально, macOS M2) |
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
