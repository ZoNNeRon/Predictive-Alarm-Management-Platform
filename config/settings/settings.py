"""
Централизованная конфигурация проекта Predictive Alarm Platform
================================================================
Источник истины для всех константных значений.
"""

# Физические пороги по ГОСТ 32601-2013 и мануалу МНХВ 
THRESHOLDS = {
    'vibration': {'warning': 3.0,  'critical': 8.0},
    'temperature': {'warning': 82.0, 'critical': 93.0},
    'current': {'warning': None, 'critical': None},  # нет нормируемых значений в ГОСТ
    'pressure': {'warning': None, 'critical': None},  # нет нормируемых значений в ГОСТ
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
FAULT_TYPES = ['overheat', 'cavitation', 'electrical']
FAULT_WEIGHTS = [0.55, 0.30, 0.15]   # доли среди аварийных циклов

FAULT_LABELS = {
    'overheat': 'Тип А: Перегрев',
    'cavitation': 'Тип Б: Кавитация',
    'electrical': 'Тип В: Электрика',
}

# Признаки скользящего окна (DataPreprocessor) 
WINDOW_SIZES = [15, 30, 60]  # минуты

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

# LLM-агент 
LLM_MODELS = [
    'qwen3.5:9b',
    'phi4:14b',
    'second_constantine/yandex-gpt-5-lite:8b',
]
DEFAULT_LLM_MODEL = LLM_MODELS[0]

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
                  'mnhv_extract', 'tm_regulation']

# Ключевые слова ремонтных работ (а не даты планового ремонта)
TOIR_WORK_MARKERS = ['дефектоскоп', 'балансиров', 'замена', 'центров', 'изоляц',
                     'подшипник', 'смазк', 'то-1', 'то-2', 'фильтр', 'уплотнен']

# Для ai_agent_benchmark heatmap графика
DIRECTIONS = {'Время_с':'lower', 'Токен_с':'higher', 'Токенов':'lower', 
              'Формат':'higher', 'Атрибуция':'higher'}
FMT = {'Время_с':'{:.1f}с','Токен_с':'{:.1f}','Формат':'{:.0%}',
       'Атрибуция':'{:.0%}','ТОиР_работы':'{:.0%}'}