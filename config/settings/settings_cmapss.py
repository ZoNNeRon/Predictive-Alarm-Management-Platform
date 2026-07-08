"""
Конфигурация валидации на реальных данных NASA C-MAPSS
=======================================================
config/settings/settings_cmapss.py

Источник истины для констант ветки CMAPSS-валидации (preprocessing -> ML -> XAI).
Отделён от config/settings/settings.py намеренно: константы насосного пайплайна
и CMAPSS-валидации не смешиваются.

Датасет: Turbofan Engine Degradation Simulation (NASA PCoE, PHM08).
Ссылка: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling
for Aircraft Engine Run-to-Failure Simulation", PHM08, Denver CO, Oct 2008.
"""

# Загрузка датасета
# Официальное зеркало NASA PCoE (S3). Архив содержит вложенный CMAPSSData.zip.
CMAPSS_URLS = [
    'https://phm-datasets.s3.amazonaws.com/NASA/'
    '6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip',
]

# Папка данных относительно корня репозитория (в git не публикуется - .gitignore)
CMAPSS_DATA_SUBDIR = ('data', 'cmapss_dataset')

# Четыре подмножества и их параметры (для загрузчика, отчётов и подписей графиков)
CMAPSS_SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
CMAPSS_SUBSET_INFO = {
    #        полётных режимов | типов отказа | описание для отчёта
    'FD001': {'conditions': 1, 'fault_modes': 1,
              'label': 'FD001: 1 режим, отказ HPC'},
    'FD002': {'conditions': 6, 'fault_modes': 1,
              'label': 'FD002: 6 режимов, отказ HPC'},
    'FD003': {'conditions': 1, 'fault_modes': 2,
              'label': 'FD003: 1 режим, отказы HPC/Fan'},
    'FD004': {'conditions': 6, 'fault_modes': 2,
              'label': 'FD004: 6 режимов, отказы HPC/Fan'},
}

# Файлы, обязанные лежать в data/cmapss_dataset/ после загрузки
CMAPSS_REQUIRED_FILES = tuple(
    f'{kind}_{s}.txt'
    for s in CMAPSS_SUBSETS
    for kind in ('train', 'test', 'RUL')
)

# Схема сырого файла: 26 колонок через пробел, без заголовка
# (1) номер юнита, (2) цикл, (3-5) режимные настройки, (6-26) 21 сенсор
CMAPSS_RAW_COLUMNS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + \
                     [f's{i}' for i in range(1, 22)]

# Идентификатор агрегата в терминах платформы (группировка препроцессора)
ENGINE_ID_FMT = '{subset}_E{unit:03d}'

# Информативные сенсоры (классический отбор для C-MAPSS):
# s1, s5, s6, s10, s16, s18, s19 константны в одном режиме и не несут сигнала
# деградации; для мультирежимных FD002/FD004 сигнал восстанавливается
# per-режимной нормализацией, набор сенсоров сохраняется единым.
CMAPSS_SENSORS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12',
                  's13', 's14', 's15', 's17', 's20', 's21']

# Короткие физические коды сенсоров - для подписей признаков на SHAP-графиках
# (s11_mean_5 -> 'Ps30 · mean_5'): обозначения из документации C-MAPSS,
# без них beeswarm/waterfall нечитаемы.
CMAPSS_SENSOR_SHORT = {
    's2': 'T24', 's3': 'T30', 's4': 'T50', 's7': 'P30',
    's8': 'Nf', 's9': 'Nc', 's11': 'Ps30', 's12': 'phi',
    's13': 'NRf', 's14': 'NRc', 's15': 'BPR', 's17': 'htBleed',
    's20': 'W31', 's21': 'W32',
}

# Физическая расшифровка используемых сенсоров (Saxena et al., PHM08) -
# для консольного отчёта XAI и подписей в тексте диплома.
CMAPSS_SENSOR_DESC = {
    's2':  'T24, температура на выходе LPC, °R',
    's3':  'T30, температура на выходе HPC, °R',
    's4':  'T50, температура на выходе LPT, °R',
    's7':  'P30, давление на выходе HPC, psia',
    's8':  'Nf, физические обороты вентилятора, rpm',
    's9':  'Nc, физические обороты ядра, rpm',
    's11': 'Ps30, статическое давление на выходе HPC, psia',
    's12': 'phi, отношение расхода топлива к Ps30',
    's13': 'NRf, приведённые обороты вентилятора, rpm',
    's14': 'NRc, приведённые обороты ядра, rpm',
    's15': 'BPR, степень двухконтурности',
    's17': 'htBleed, энтальпия отбора воздуха',
    's20': 'W31, отбор охлаждения HPT, lbm/s',
    's21': 'W32, отбор охлаждения LPT, lbm/s',
}

# Метки тяжести из RUL (остатка циклов до отказа), piecewise:
#   RUL >  RUL_WARNING                  -> 0 Норма
#   RUL_CRITICAL < RUL <= RUL_WARNING   -> 1 Предупреждение
#   RUL <= RUL_CRITICAL                 -> 2 Авария
RUL_WARNING = 50
RUL_CRITICAL = 20

CMAPSS_STAGE_LABELS = {0: 'Норма', 1: 'Предупреждение', 2: 'Авария'}

# Окна rolling-признаков в ЦИКЛАХ (не минутах): траектории 128-360 циклов,
# насосные [15, 30, 60] съедали бы слишком длинный прогрев на агрегат.
CMAPSS_WINDOW_SIZES = [5, 10, 20]

# Лаг градиентного признака diff (в циклах). Прогрев на двигатель =
# max(CMAPSS_WINDOW_SIZES, CMAPSS_DIFF_LAG + 1) строк; самые короткие
# обрезанные test-траектории (19-38 циклов) при этом сохраняют часть строк.
CMAPSS_DIFF_LAG = 10

# Мультирежимные подмножества: перед расчётом признаков сенсоры приводятся
# к z-score внутри полётного режима (статистики - только по train).
CMAPSS_MULTI_REGIME_SUBSETS = ('FD002', 'FD004')

# Идентификация режима: op1 группируется вокруг шести якорей (высотность),
# однозначно разделяющих режимы; ближайший якорь = id режима.
CMAPSS_REGIME_OP1_ANCHORS = (0.0, 10.0, 20.0, 25.0, 35.0, 42.0)

# Пути вывода (относительно корня репозитория)
CMAPSS_MODELS_SUBDIR = ('models', 'cmapss')
CMAPSS_GRAPHS_PREFIX = 'cmapss'          # префикс имён PNG в artifacts/graphs
CMAPSS_TABLES_PREFIX = 'cmapss'          # префикс имён CSV в artifacts/tables

# Воспроизводимость
CMAPSS_RANDOM_STATE = 42
