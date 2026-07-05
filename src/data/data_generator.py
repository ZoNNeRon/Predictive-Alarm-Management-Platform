"""
Генератор имитационных данных насоса МНХВ
==========================================
src/data/data_generator.py

Конечный автомат (State Machine) + AR(1)-процессы + три типа отказа.

Состояния:
    0 - Off         : насос выключен
    1 - Startup     : пуск (2-3 мин), пусковые токи и гидроудар
    2 - Healthy     : штатная работа с AR(1) шумом
    3 - Degradation : деградация, AR(1) поверх нарастающего тренда
    4 - Critical    : критический отказ, устойчивое нарушение порогов

Типы отказа (fault_type):
    overheat    (Тип А, ~55%) - перегрев + рост нагрузки:
        температура растёт к 93+°C, ток растёт и волатилен,
        вибрация умеренно растёт, давление в норме.
        Причина: износ подшипника, несоосность.

    cavitation  (Тип Б, ~30%) - критическая вибрация + падение давления:
        вибрация растёт к 8+ мм/с, давление падает/пульсирует,
        температура и ток в норме.
        Причина: кавитация, повреждение рабочего колеса.

    electrical  (Тип В, ~15%) - аномалии тока без роста вибрации и температуры:
        ток скачет, растёт его дисперсия, вибрация и температура в зелёной зоне.
        Причина: электрика, изменение сопротивления сети.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config.settings import THRESHOLDS, FAULT_TYPES, FAULT_WEIGHTS, PUMP_SEEDS
from src.visualisation.simulation_visualisation import plot_smart_episode


class PumpDataGenerator:
    """
    Генератор данных на основе конечного автомата (State Machine).
    Каждый физический параметр моделируется AR(1)-процессом:
        x[t] = μ + φ·(x[t-1] - μ) + ε[t]
    где φ - инерция сигнала, μ - целевое среднее, ε - белый шум,
    t - текущий момент времени, t-1 - предыдущий момент времени.
    """

    VIB_WARNING = THRESHOLDS['vibration']['warning']
    VIB_CRITICAL = THRESHOLDS['vibration']['critical']
    TEMP_WARNING = THRESHOLDS['temperature']['warning']
    TEMP_CRITICAL = THRESHOLDS['temperature']['critical']

    FAULT_TYPES = FAULT_TYPES
    FAULT_WEIGHTS = FAULT_WEIGHTS

    def __init__(self, pump_id, start_date='2026-04-01 00:00:00', total_days=90):
        self.pump_id = pump_id
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        self.total_days = total_days
        self.total_minutes = total_days * 24 * 60

        self.ambient_temp = 20.0
        self.current_temp = self.ambient_temp

        self.data = []
        self.current_time = self.start_date
        self.minutes_generated = 0

        # AR(1)-состояние: инициализация нулями для плавного старта
        self.last_vib = 0.0     # вибрация
        self.last_curr = 0.0    # ток
        self.last_press = 0.0   # давление

    # Утилиты

    @staticmethod
    def _get_state_name(state: int) -> str:
        """
        Наименование текущего режима:
            - Off (0) - Оборудование выключено;
            - Startup (1) - Запуск оборудования;
            - Healthy (2) - Нормальная работа;
            - Degradation (3) - Деградация значений, рост/падение критических показателей;
            - Critical (4) - Работа оборудования на уровне отказа/поломки.
        """

        return {0: 'Off', 1: 'Startup', 2: 'Healthy',
                3: 'Degradation', 4: 'Critical'}.get(state, 'Unknown')

    def _append_row(self, state, vib, temp, curr, press,
                    fault_type='none',
                    anomaly_vibration=0, anomaly_temperature=0, anomaly_current=0):
        """Запись одной минуты показаний в датасет."""

        self.data.append({
            'timestamp': self.current_time,
            'pump_id': self.pump_id,
            'state': state,
            'state_name': self._get_state_name(state),
            'fault_type': fault_type,
            'vibration': round(max(0.0, vib), 3),
            'temperature': round(max(self.ambient_temp, temp), 2),
            'current': round(max(0.0, curr), 2),
            'pressure': round(max(0.0, press), 3),
            'anomaly_vibration': anomaly_vibration,     # 1 = аппаратный сбой датчика вибрации
            'anomaly_temperature': anomaly_temperature, # 1 = аппаратный сбой термопары
            'anomaly_current': anomaly_current,         # 1 = аппаратный сбой датчика тока
        })
        self.current_time += timedelta(minutes=1)
        self.minutes_generated += 1

    # Генераторы состояний

    def generate_off(self, duration_mins: int):
        """
        Состояние 0 - оборудование выключено.
        Температура экспоненциально остывает.
        Механические сигналы затухают через AR(1) с коэффициентом 0.5.
        """

        for _ in range(duration_mins):
            self.current_temp = (self.ambient_temp
                                 + (self.current_temp - self.ambient_temp) * 0.95)
            self.last_vib = self.last_vib * 0.5 + np.random.normal(0, 0.04)
            self.last_curr = self.last_curr * 0.5 + np.random.normal(0, 0.08)
            self.last_press = self.last_press * 0.5 + np.random.normal(0, 0.008)
            self._append_row(0, self.last_vib, self.current_temp,
                             self.last_curr, self.last_press)

    def generate_startup(self):
        """
        Состояние 1 - пуск (2–3 мин).
        Резкий всплеск тока (~150 А, потом спадает), гидроудар давления,
        кратковременный рост вибрации до 4–5 мм/с.
        """

        duration = np.random.randint(2, 4)
        for i in range(duration):
            self.current_temp += np.random.normal(4.0, 0.8)
            self.last_vib = np.random.normal(4.5, 0.5)
            self.last_curr = np.random.normal(150 * (1 - 0.6 * i / duration), 10)
            self.last_press = np.random.normal(1.8, 0.12)
            self._append_row(1, self.last_vib, self.current_temp,
                             self.last_curr, self.last_press)

    def generate_healthy(self, duration_mins: int):
        """
        Состояние 2 - штатная работа.
        AR(1): x[t] = μ + φ·(x[t-1] - μ) + ε[t]
        φ = 0.9 (сильная инерция), μ - физические базовые значения агрегата.
        С вероятностью 0.1% - аппаратный сбой одного датчика (изолированный выброс).
        """

        PHI = 0.9
        for _ in range(duration_mins):
            self.current_temp = (self.current_temp * PHI 
                                 + 70.0 * (1 - PHI) + np.random.normal(0, 0.4))
            self.last_vib = (self.last_vib * 0.88 
                             + 1.8 * 0.12 + np.random.normal(0, 0.08))
            self.last_curr = (self.last_curr * 0.9 
                              + 50.0 * 0.1 + np.random.normal(0, 0.8))
            self.last_press = (self.last_press * 0.85 
                               + 1.5  * 0.15 + np.random.normal(0, 0.018))

            anom_vib = anom_temp = anom_curr = 0
            vib_out  = self.last_vib
            if np.random.rand() <= 0.001:
                # Разовый аппаратный сбой - один из трёх датчиков
                sensor = np.random.choice(['vib', 'temp', 'curr'])
                if sensor == 'vib':
                    vib_out = np.random.normal(9.5, 0.5); anom_vib = 1
                elif sensor == 'temp':
                    self.current_temp = np.random.normal(97.0, 1.5); anom_temp = 1
                else:
                    self.last_curr = np.random.normal(90.0, 3.0); anom_curr = 1

            self._append_row(2, vib_out, self.current_temp, self.last_curr, self.last_press,
                             anomaly_vibration=anom_vib,
                             anomaly_temperature=anom_temp,
                             anomaly_current=anom_curr)

    def generate_degradation(self, duration_mins: int, fault_type: str):
        """
        Состояние 3 - деградация по одному из трёх типов.
        Параметры, не участвующие в данном типе отказа, удерживаются у нормы.
        Это обеспечивает разные сигнатуры, которые XAI (SHAP) должен распознать.
        """

        temp_start = self.current_temp

        if fault_type == 'overheat':
            # Тип А: рост температуры, рост тока (волатильный), 
            # умеренный рост вибрации, давление слабо падает по мере износа
            temp_trend = np.linspace(temp_start, 91.0, duration_mins)
            vib_trend  = np.linspace(2.0, 5.5, duration_mins)
            curr_trend = np.linspace(50.0, 66.0, duration_mins)
            press_trend = np.linspace(1.5, 1.4, duration_mins)
            for i in range(duration_mins):
                self.current_temp = temp_trend[i] + np.random.normal(0, 0.5)
                self.last_vib = vib_trend[i] + np.random.normal(0, 0.18)
                self.last_curr = curr_trend[i] + np.random.normal(0, 2.0)   # высокая волатильность
                self.last_press = (self.last_press * 0.85 
                                   + press_trend[i] * 0.15 + np.random.normal(0, 0.03))
                self._append_row(3, self.last_vib, self.current_temp,
                                 self.last_curr, self.last_press, fault_type='overheat')

        elif fault_type == 'cavitation':
            # Тип Б: сильный рост вибрации, падение давления (пульсация), 
            # температура и ток в норме
            vib_trend = np.linspace(2.5, 7.5, duration_mins)
            press_trend = np.linspace(1.5, 0.9, duration_mins)  # падение напора
            for i in range(duration_mins):
                self.current_temp = (self.current_temp * 0.9
                                     + 70.0 * 0.1 + np.random.normal(0, 0.5))
                self.last_vib = vib_trend[i] + np.random.normal(0, 0.3)
                self.last_curr = (self.last_curr * 0.9
                                  + 50.0 * 0.1 + np.random.normal(0, 1.0))
                # Пульсация давления - высокая дисперсия на падающем тренде
                self.last_press = press_trend[i] + np.random.normal(0, 0.08)
                self._append_row(3, self.last_vib, self.current_temp,
                                 self.last_curr, self.last_press, fault_type='cavitation')

        elif fault_type == 'electrical':
            # Тип В: сильный рост тока + скачки (10%), 
            # вибрация, температура и давление в норме
            curr_trend = np.linspace(50.0, 72.0, duration_mins)
            for i in range(duration_mins):
                self.current_temp = (self.current_temp * 0.9
                                     + 72.0 * 0.1 + np.random.normal(0, 0.6))
                self.last_vib = (self.last_vib * 0.88
                                 + 2.0 * 0.12 + np.random.normal(0, 0.15))
                spike = np.random.normal(0, 6.0) if np.random.rand() < 0.1 else 0.0
                self.last_curr = curr_trend[i] + np.random.normal(0, 3.0) + spike
                self.last_press = (self.last_press * 0.85
                                   + 1.45 * 0.15 + np.random.normal(0, 0.04))
                self._append_row(3, self.last_vib, self.current_temp,
                                 self.last_curr, self.last_press, fault_type='electrical')

    def generate_critical(self, duration_mins, fault_type):
        """Критический отказ - устойчивое нарушение порогов с сохранением AR(1)-памяти."""

        PHI = 0.8   # память сигнала: значения инерционны, не скачут случайно
        for _ in range(duration_mins):
            # AR(1) с привязкой к типу аварии
            if fault_type == 'overheat':
                self.current_temp = (self.current_temp * PHI 
                                        + 96.0 * (1-PHI) + np.random.normal(0, 0.8))
                self.last_vib = (self.last_vib * PHI 
                                    + 8.5 * (1-PHI) + np.random.normal(0, 0.4))
                self.last_curr = (self.last_curr * PHI 
                                    + 78.0 * (1-PHI) + np.random.normal(0, 2.0))
                self.last_press = (self.last_press * PHI 
                                    + 1.3 * (1-PHI) + np.random.normal(0, 0.05))

            elif fault_type == 'cavitation':
                self.current_temp = (self.current_temp * 0.9 
                                     + 74.0 * 0.1 + np.random.normal(0, 0.8))   # норма
                self.last_vib = (self.last_vib * PHI 
                                 + 9.0 * (1-PHI) + np.random.normal(0, 0.5))
                self.last_curr = (self.last_curr * 0.9 
                                  + 52.0 * 0.1 + np.random.normal(0, 1.0))      # норма
                self.last_press = (self.last_press * PHI 
                                   + 0.7 * (1-PHI) + np.random.normal(0, 0.06))

            elif fault_type == 'electrical':
                self.current_temp = (self.current_temp * 0.9 
                                     + 75.0 * 0.1 + np.random.normal(0, 0.8))   # норма
                self.last_vib = (self.last_vib * 0.85 
                                 + 2.2 * 0.15 + np.random.normal(0, 0.2))       # норма
                self.last_curr = (self.last_curr * PHI 
                                  + 95.0 * (1-PHI) + np.random.normal(0, 3.5))
                self.last_press = (self.last_press * 0.85 
                                   + 1.4 * 0.15 + np.random.normal(0, 0.05))

            self._append_row(4, self.last_vib, self.current_temp,
                             self.last_curr, self.last_press, fault_type=fault_type)
            
    # Основной метод генерации

    def generate_dataset(self) -> pd.DataFrame:
        print(f"Генерация данных: {self.total_days} дней,"
              f"шаг 1 мин → {self.total_minutes:,} строк")

        while self.minutes_generated < self.total_minutes:
            cycle_type = np.random.choice(['normal', 'failure'], p=[0.80, 0.20])

            self.generate_off(np.random.randint(60, 180))
            if self.minutes_generated >= self.total_minutes:
                break

            self.generate_startup()
            if self.minutes_generated >= self.total_minutes:
                break

            if cycle_type == 'normal':
                self.generate_healthy(np.random.randint(12 * 60, 48 * 60))
            else:
                # Тип отказа по реалистичному распределению A/Б/В
                fault = np.random.choice(self.FAULT_TYPES, p=self.FAULT_WEIGHTS)
                self.generate_healthy(np.random.randint(12 * 60, 24 * 60))
                if self.minutes_generated >= self.total_minutes:
                    break
                self.generate_degradation(np.random.randint(5 * 60, 10 * 60), fault)
                if self.minutes_generated >= self.total_minutes:
                    break
                self.generate_critical(np.random.randint(10, 60), fault)

        df = pd.DataFrame(self.data).head(self.total_minutes)
        self._print_state_distribution(df)
        return df

    @staticmethod
    def _print_state_distribution(df: pd.DataFrame):
        """Статистика по состояниям и типам отказа."""

        print("\nРаспределение состояний:")
        counts = df['state_name'].value_counts()
        total = len(df)
        for name, cnt in counts.items():
            print(f"  {name:<12}: {cnt:>7,} мин. ({100 * cnt / total:5.1f}%)")

        fault_rows = df[df['fault_type'] != 'none']
        if not fault_rows.empty:
            print("\nРаспределение типов отказа (Degradation + Critical):")
            for ft, cnt in fault_rows['fault_type'].value_counts().items():
                pct = 100 * cnt / len(fault_rows)
                print(f"  {ft:<12}: {cnt:>7,} мин. ({pct:5.1f}%)")


# Точка входа

if __name__ == "__main__":
    fleet_dfs = []

    for pump_id, seed in PUMP_SEEDS.items():
        np.random.seed(seed)
        print(f"\n--- Генерация данных для {pump_id} (seed={seed}) ---")
        generator = PumpDataGenerator(pump_id=pump_id,
                                      start_date='2026-04-01 00:00:00',
                                      total_days=90)
        df = generator.generate_dataset()

        # Демонстрационный график - только для первого насоса
        if pump_id == 'MNHV_001':
            print(f"Построение демонстрационного графика для {pump_id}...")
            plot_smart_episode(df, hours=60)

        fleet_dfs.append(df)

    print("\nСборка единого датасета имитированного предприятия...")
    enterprise_df = pd.concat(fleet_dfs, ignore_index=True)

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))))
    os.makedirs(os.path.join(project_root, 'data', 'raw'), exist_ok=True)
    enterprise_path = os.path.join(project_root, 'data', 'raw', 'industrial_pumps_dataset.csv')
    enterprise_df.to_csv(enterprise_path, index=False)

    print(f"\nПарк оборудования сгенерирован.")
    print(f"Всего насосов: {len(PUMP_SEEDS)} | Всего строк: {len(enterprise_df)}")

    # Валидация: средние значения по типам отказа в Critical-состоянии
    crit = enterprise_df[enterprise_df['state'] == 4]
    if not crit.empty:
        print(f"\nСредние значения по типам отказа (Critical, state=4):")
        print(crit.groupby('fault_type')[
            ['vibration', 'temperature', 'current', 'pressure']
        ].mean().round(1))