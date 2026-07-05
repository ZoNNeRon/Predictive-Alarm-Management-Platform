"""
Централизованная конфигурация проекта Predictive Alarm Platform
================================================================
config/settings/settings.py

Источник истины для всех константных значений.
"""

# Физические пороги по ГОСТ 32601-2013 и мануалу МНХВ 
THRESHOLDS = {
    'vibration': {'warning': 3.0,  'critical': 8.0},
    'temperature': {'warning': 82.0, 'critical': 93.0},
    'current': {'warning': None, 'critical': None}, # нет нормируемых значений в ГОСТ
    'pressure': {'warning': None, 'critical': None}, # нет нормируемых значений в ГОСТ
}

# Парк насосов: сиды для воспроизводимости 
PUMP_SEEDS = {
    'MNHV_001': 42,
    'MNHV_002': 137,
    'MNHV_003': 2026,
    'MNHV_004': 54321,
    'MNHV_005': 99999,
}

# Типы отказа (генератор + анализ) 
FAULT_WEIGHTS = [0.55, 0.30, 0.15] # доли среди аварийных циклов

FAULT_LABELS = {
    'overheat': 'Тип А: Перегрев',
    'cavitation': 'Тип Б: Кавитация',
    'electrical': 'Тип В: Электрика',
}

# Признаки скользящего окна (DataPreprocessor) 
WINDOW_SIZES = [15, 30, 60] # минуты

# ML: разбивка по насосам 
PUMPS = ['MNHV_001', 'MNHV_002', 'MNHV_003', 'MNHV_004', 'MNHV_005']
TRAIN_PUMPS = ['MNHV_001', 'MNHV_002', 'MNHV_003', 'MNHV_004']
TEST_PUMPS = ['MNHV_005']


# RAG: модель эмбеддингов 
EMBED_MODEL = 'intfloat/multilingual-e5-large'

# RAG: порог релевантности (L2; > значения = нерелевантно) 
RELEVANCE_THRESHOLD = 1.2

# RAG: типы документов и метки для визуализации
DOC_TYPES = {
    'manual': 'Технический мануал (руководство по эксплуатации)',
    'gost': 'ГОСТ / Нормативный документ',
    'sop': 'Регламент технического обслуживания (SOP)',
    'schedule': 'График технического обслуживания',
    'diagnostics': 'Расширенная вибродиагностика (аналитическое дополнение)',
}

# RAG: короткие русские подписи типов документов для осей/ячеек хитмапов
# (полные названия из DOC_TYPES слишком длинны для компактных меток).
DOC_TYPE_SHORT_RU = {
    'manual':      'Руководство',
    'gost':        'ГОСТ',
    'diagnostics': 'Вибродиагностика',
    'sop':         'Регламент ТО',
    'schedule':    'График ТО',
}

# RAG: параметры чанкинга по типу документа 
CHUNK_CONFIG = {
    'manual': {'chunk_size': 600, 'chunk_overlap': 120},
    'gost': {'chunk_size': 700, 'chunk_overlap': 150},
    'sop': {'chunk_size': 550, 'chunk_overlap': 100},
    'schedule': {'chunk_size': 350, 'chunk_overlap':  50},
    'diagnostics': {'chunk_size': 500, 'chunk_overlap': 100},
}

# RAG: карта загружаемых файлов (явный allowlist) 
# Только эти файлы попадают в ChromaDB; остальные игнорируются.
DOC_TYPE_MAP = {
    'tm_regulation.md': 'sop',
    'tm_schedule.md': 'schedule',
    'gost_extract.md': 'gost',
    'mnhv_extract.md': 'manual',
    'diagnostics_extended.md': 'diagnostics',
}

FAULT_TYPES = ('overheat', 'cavitation', 'electrical')

DOC_DISPLAY_NAMES = {
    'mnhv_extract.md': 'Руководство по эксплуатации МНХВ',
    'gost_extract.md': 'ГОСТ 32601-2013',
    'tm_regulation.md': 'Регламент ТО предприятия',
    'diagnostics_extended.md':'Методика вибродиагностики',
    'tm_schedule.md': 'График ППР',
}

# Порог уверенности классификатора ТИПА отказа, ниже которого стадия = 'unknown'.
FAULT_CONFIDENCE_THRESHOLD = 0.5

# LLM-агент 
LLM_MODELS = [
    'qwen3.5:9b',
    'phi4:14b',
    'second_constantine/yandex-gpt-5-lite:8b',
]
DEFAULT_LLM_MODEL = LLM_MODELS[2]

MODEL_LABELS = {
    'qwen3.5:9b':                              'Qwen 3.5 9B',
    'phi4:14b':                                'Phi-4 14B',
    'second_constantine/yandex-gpt-5-lite:8b': 'YandexGPT-5 8B',
}

# Обязательные разделы для проверки формата
REQUIRED_SECTIONS = ['СТАТУС', 'ДИАГНОЗ', 'ПРЕДПИСАНИЕ', 'ТОиР']

# Маркеры явной атрибуции источника (модель ссылается на документ, а не на "Сценарий А")
SOURCE_MARKERS = ['регламент', 'tm_regulation', 'гост', 'мануал',
                  'руководств', 'источник', 'документ',
                  'mnhv_extract']

# Ключевые слова ремонтных работ (а не даты планового ремонта)
TOIR_WORK_MARKERS = ['дефектоскоп', 'балансиров', 'замена', 'центров', 'изоляц',
                     'подшипник', 'смазк', 'то-1', 'то-2', 'фильтр', 'уплотнен']

# Для ai_agent_benchmark heatmap графика
DIRECTIONS = {'Время_с':'lower', 'Токен_с':'higher', 'Токенов':'lower'}
FMT = {'Время_с':'{:.1f}с','Токен_с':'{:.1f}'}

# Маркеры аварийного останова - недопустимы в предписании на стадии Предупреждение.
EMERGENCY_MARKERS = ('немедленно останов', 'немедленно остановить',
                     'обесточить')
# Маркеры решительного действия - ожидаемы на стадии Авария.
DECISIVE_MARKERS = ('останов', 'обесточ', 'изолир', 'снизить', 'прикрыть')

# Норматив ANSI/ISA-18.2: целевой предел подтверждённых тревог на оператора.
# Единый источник для дашборда (метрика «Переходов/час») и графиков валидации.
ISA_ALARM_RATE_LIMIT = 6   # тревог в час

# UI / runtime дашборда
UI_REFRESH_SEC = 2.0        # период авто-перерисовки фрагментов (с)
UI_RENAG_MIN = 10           # ре-наг: повтор оповещения, если стадия держится N sim-мин
UI_GEN_TIMEOUT_SEC = 90.0   # страховочный таймаут фоновой генерации предписания (с)
UI_MAX_EVENTS = 100         # кап журнала истории событий (защита от роста памяти)
UI_GRAPH_WINDOW_MIN = 120   # окно трендов оператора: последние N точек/минут

# Диапазоны осей Y графиков оператора (физические пределы датчиков)
UI_YRANGE = {
    'vibration': (0, 12), 'temperature': (0, 98),
    'current': (0, 150), 'pressure': (0, 2),
}

# Цвета статусов тяжести (0=Норма, 1=Предупреждение, 2=Авария) и «квитировано»
UI_SEVERITY_COLORS = {0: '#1E8A3C', 1: '#E0A800', 2: '#C62828'}
UI_ACK_COLOR = '#1E8A3C'

# ── Симуляция реального времени (experiments/realtime_validation) ──────
SIM_AMBIENT_TEMP = 20.0                 # °C - температура окружающей среды (затухание)
SIM_START_DATE = '2026-04-01 00:00:00'  # старт модельных часов парка
# Устоявшиеся значения Healthy-AR(1) (неподвижные точки) - старт в Норме без скачка
SIM_HEALTHY_FIXED = {'temperature': 70.0, 'vibration': 1.8,
                     'current': 50.0, 'pressure': 1.5}