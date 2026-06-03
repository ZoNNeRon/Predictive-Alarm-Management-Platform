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
│   ├── data_generator.py      # ГОТОВО — генератор синтетических данных (AR(1), State Machine)
│   ├── data_preprocessor.py   # ГОТОВО — rolling features, защита от data leakage
│   ├── ml_pipeline.py         # ГОТОВО — обучение LR / RF / XGBoost, AlarmManager
│   ├── xai_module.py          # ГОТОВО — SHAP TreeExplainer, SymptomVector dataclass
│   ├── rag_database.py        # ТРЕБУЮТСЯ ДОРАБОТКИ — ChromaDB + multilingual-e5-large, PDF→чанки
│   ├── kb_text_loader.py      # НЕ НАПИСАН — загрузчик ручных текстовых файлов базы знаний
│   ├── agent_module.py        # НЕ НАПИСАН — LLM-агент (Ollama + Phi3), сборка промпта
│   └── app.py                 # НЕ НАПИСАН — Streamlit двухуровневый UI
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
│   │   └── gost_32601_2013.pdf		# ГОСТ 32601
│   ├── manuals/
│   │  └── mnhv_manual.pdf 			# Мануал МНХВ (ООО «НК «Крон», 2023)
│   ├── regulations/
│   │   └── tm_regulation.pdf           # Регламент ТО (sop)
│   ├── schedules/
│         └── tm_schedule.pdf             # График ППР (schedule)
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
- Деградация: AR(1) **поверх нарастающего linspace-тренда** — именно это XAI распознаёт как cumulative signal
- Помехи датчиков: вероятность 0.001, флаги `anomaly_vibration` / `anomaly_temperature` изолированы от физической модели
- **5 насосов** (`MNHV_001`–`MNHV_005`), seed через явный словарь (не умножение)
- **432 000 строк**, 60 дней, шаг 1 минута
- Выход: `data/raw/enterprise_pump_fleet.csv`

### 2. `data_preprocessor.py` — Подготовка признаков
- **Rolling features** (15, 30, 60 мин): mean, std, max для каждого из 4 сенсоров
- **Gradient:** `shift(1) - shift(31)` — ровно 30 шагов, консистентно с rolling(30)
- **Data leakage защита:** `shift(1)` перед каждым `.rolling()` — в строке T используется только [T-W … T-1]
- **`groupby('pump_id')`:** окна не перетекают между насосами
- `min_periods=w` для mean/max (честный NaN), `min_periods=2` для std
- Флаги `anomaly_*` удаляются ДО подачи в ML — модель учится игнорировать шум через сглаживание
- **`FEATURE_COLS`** — явный контракт из 40 признаков, используется и в train, и в inference
- Фильтрация Off (0) и Startup (1) из обучающей выборки — ими занимается `AlarmManager`
- Выход: `data/processed/processed_features.csv`

### 3. `ml_pipeline.py` — Обучение моделей
- **Group Split:** train = MNHV_001–004 (321 615 строк), test = MNHV_005 (80 531 строки)
- Три модели: Logistic Regression (baseline), Random Forest (n=300), XGBoost (n=300, lr=0.1, depth=6)
- Балансировка: `class_weight='balanced'` + `sample_weight` через `compute_sample_weight`
- **Метрики:** Confusion Matrix, F1-Macro, Recall(Авария) как KPI, PR-AUC
- **Результаты XGBoost:** F1-Macro=0.97, Recall(Critical)=0.948, PR-AUC(Critical)=0.993
- **`AlarmManager`:** если `raw_state in [0, 1]` → принудительно возвращает 0 (Alarm Shelving)
- Три визуализации: confusion matrices heatmap, grouped bar chart, PR-кривые

### 4. `xai_module.py` — Объяснимый ИИ (SHAP)
- **`XAIExplainer`:** `shap.TreeExplainer` для XGBoost, `target_class_idx=2` (Авария)
- Возвращает **`SymptomVector`** (dataclass) — строго типизированный контракт для агента
- Сортировка по `abs(shap_weight)` — признак -0.8 важнее признака +0.1
- Каждый `SymptomContribution` содержит: `sensor`, `window`, `value`, `shap_weight`, `direction`, пороги ГОСТ
- **Визуализация:** `plot_waterfall()` (локальное объяснение) + `plot_summary()` (beeswarm, глобальная важность)
- `generate_llm_prompt` **отсутствует** в этом модуле — намеренно (SRP). Промпт строит `agent_module.py`

### 5. `rag_database.py` — База знаний (RAG)
- **Стек:** LangChain + ChromaDB (локально) + `intfloat/multilingual-e5-large` (MPS, M2)
- **`StructuredPDFLoader`:** pymupdf4llm → Markdown (сохраняет таблицы), fallback → pdfplumber
- **Метаданные чанков:** `doc_type` (manual/gost/sop/schedule), `source`, `section`, `chunk_id`
- **Разные chunk_size по типу:** gost=400, manual=600, sop=500, schedule=350
- **e5 prefix:** `passage:` при загрузке, `query:` при поиске — обязательно для e5-large
- **Порог релевантности:** `RELEVANCE_THRESHOLD=1.2` (L2), нерелевантные результаты отсекаются
- Батчевая запись по 500 чанков
- Фильтрация по `doc_type` через `search_kwargs`
- **Текущее состояние базы:** 1851 чанк (ГОСТ — 94% из-за проблемы с PDF, см. ниже)
- Три визуализации для диплома: состав базы, распределение длин чанков, качество поиска

---

## Что предстоит сделать

### Завершить формирование базы знаний для RAG
PDF-файлы очень плохо воспринимаются инструментами системы, необходимо подготовить ГОСТ и мануал вручную для записи в БД. Необходимо решение, которое позволит переводить такие файлы в ручной формат.

### Модуль агента `agent_module.py`
LLM-агент, который принимает `SymptomVector` из `xai_module.py` и возвращает текстовое предписание.

**Логика модуля:**
1. Получить `SymptomVector` (уже реализован в `xai_module.py`)
2. Сформировать поисковый запрос из симптомов → `rag_database.search_by_symptoms()`
3. Получить 3–4 релевантных чанка из ChromaDB
4. Собрать промпт: SHAP-симптомы + пороги ГОСТ + контекст из базы знаний
5. Передать в LLM (Ollama + Phi3 или llama3) → получить текстовую инструкцию
6. Вернуть структурированный объект `AgentResponse` для передачи в UI

**Технологии:** Ollama (локально, M2), модель Phi3 14b (уже установлена), LangChain или прямой API Ollama.

**Важно:** промпт строится здесь, не в `xai_module.py`.

### После агента: `app.py` — Streamlit UI
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

---

## Критические ограничения (задокументированы в дипломе)

1. Все 5 насосов имеют **идентичные номинальные параметры** — исследование разнотипного оборудования выходит за рамки работы
2. ГОСТ 32601-2013 занимает 94% базы знаний в текущей сборке — **требует пересборки** через `kb_text_loader.py` + `gost_extract.md`
3. LLM работает **локально на M2** (Ollama + Phi3) — зависимости от внешних API нет

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
