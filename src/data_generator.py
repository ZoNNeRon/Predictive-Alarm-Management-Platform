import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


class PumpDataGenerator:
    """
    Генератор данных на основе конечного автомата (State Machine).

    Состояния:
        0 - Off         : насос выключен
        1 - Startup     : пуск (2-3 мин), пусковые токи и гидроудар
        2 - Healthy     : штатная работа с AR(1) шумом
        3 - Degradation : деградация, AR(1) поверх нарастающего тренда
        4 - Critical    : критический отказ, устойчивое нарушение порогов
    """

    # Физические пороги по ГОСТ 32601-2013 и мануалу МНХВ
    VIB_WARNING = 3.0   # мм/с
    VIB_CRITICAL = 8.0   # мм/с
    TEMP_WARNING = 82.0  # °C
    TEMP_CRITICAL = 93.0 # °C

    def __init__(self, pump_id, start_date='2026-04-01 00:00:00', total_days=60):
        self.pump_id = pump_id # уникальный идентификатор агрегата
        
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        self.total_days = total_days
        self.total_minutes = total_days * 24 * 60

        # Пути к файлам (определяются относительно корня проекта)
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raw_dir = os.path.join(self.project_root, 'data', 'raw')
        self.processed_dir = os.path.join(self.project_root, 'data', 'processed')
        self.graph_dir = os.path.join(self.project_root, 'data', 'graphs')

        # Физическое состояние
        self.ambient_temp = 20.0
        self.current_temp = self.ambient_temp

        self.data = []
        self.current_time = self.start_date
        self.minutes_generated = 0

    # Утилиты
    @staticmethod
    def _get_state_name(state: int) -> str:
        return {0: 'Off', 1: 'Startup', 2: 'Healthy',
                3: 'Degradation', 4: 'Critical'}.get(state, 'Unknown')

    def _append_row(self, state, vib_sensor, temp_sensor, curr_sensor, press_sensor, anom_vib=0, anom_temp=0):
        """Запись показаний в датасет, без изменения физики"""
        self.data.append({
            'timestamp': self.current_time,
            'pump_id': self.pump_id, # тег оборудования
            'state': state, # код текущего состояния
            'state_name': self._get_state_name(state), # текущее состояние 
            'vibration': round(max(0, vib_sensor), 2), # текущая вибрация
            'temperature': round(max(self.ambient_temp, temp_sensor), 1), # текущая температура
            'current': round(max(0, curr_sensor), 1), # текущий ток
            'pressure': round(max(0, press_sensor), 2), # текущее давление
            'anomaly_vibration': anom_vib, # Флаг аппаратной помехи датчика вибрации
            'anomaly_temperature': anom_temp # Флаг аппаратной помехи термопары
        })
        self.current_time += timedelta(minutes=1)
        self.minutes_generated += 1

    # Генераторы состояний

    def generate_off(self, duration_mins: int):
        """
        Состояние 0 — насос выключен.
        Температура экспоненциально остывает, все прочие параметры ~0.
        """
        for _ in range(duration_mins):
            self.current_temp = self.ambient_temp + (self.current_temp - self.ambient_temp) * 0.95
            
            # Физическое состояние: микро-вибрации трубы или наводки в кабеле (всегда положительные)
            self.last_vib = abs(np.random.normal(0, 0.02))
            self.last_curr = abs(np.random.normal(0, 0.05))
            self.last_press = abs(np.random.normal(0, 0.01))
            
            self._append_row(0, self.last_vib, self.current_temp, self.last_curr, self.last_press)

    def generate_startup(self):
        """
        Состояние 1 — пуск (2–3 мин).
        Резкий всплеск тока (пусковой ток ~3x номинала), гидроудар давления,
        кратковременный рост вибрации до 4–5 мм/с.
        """
        duration = np.random.randint(2, 4)
        for i in range(duration):
            # Фактор падает от 1.0 до ~0.3-0.5
            startup_factor = 1.0 - (i / duration) 
            
            # Ток плавно падает со ~150 А (50 + 100*1.0) до номинальных 50 А
            curr = 50 + (100 * startup_factor) + np.random.normal(0, 5.0)
            
            # Давление (гидроудар) спадает с 2.0 МПа (1.5 + 0.5*1.0) до 1.5 МПа
            press = 1.5 + (0.5 * startup_factor) + np.random.normal(0, 0.1)
            
            # Вибрация при прохождении критических частот
            vib = 1.8 + (2.5 * startup_factor) + np.random.normal(0, 0.3)
            
            # Резкий скачок температуры (физика трения)
            self.current_temp += np.random.normal(2, 0.5) 
            
            self.last_curr = curr
            self.last_press = press
            self.last_vib = vib
            
            self._append_row(1, vib, self.current_temp, curr, press)

    def generate_healthy(self, duration_mins: int):
        """
        Состояние 2 — штатная работа.

        Каждый параметр моделируется авторегрессионным процессом AR(1):
            x[t] = μ + φ·(x[t-1] - μ) + ε[t]

        - x[t] – текущее значение (например, температура насоса на текущей минуте).
        - x[t-1] – значение системы на предыдущем шаге, «память системы» 
            (например, температура на прошлой минуте). 
        - μ (Мю) – базовое среднее / Set-point, целевое значение, к которому система 
            естественным образом стремится (например, рабочая температура 70°C).
        - φ (Фи) – коэффициент авторегрессии / инерция, определяющая силу «памяти 
            системы», представленное числом от 0 до 1 (в стационарных процессах):
        - Если φ = 0,9 – у системы сильная инерция (значение будет меняться очень плавно);
        - Если φ = 0,1 – инерция слабая, система мгновенно «забудет» прошлое и прыгнет к норме.
        - ε[t] (Эпсилон) – белый шум / случайный шок, непредсказуемое физическое воздействие в 
            текущий момент времени (например, наводки, вибрация от соседнего станка или порыв ветра). 

        """
        for _ in range(duration_mins):
            # AR(1) возмущения
            self.current_temp = self.current_temp * 0.9 + 70.0 * 0.1 + np.random.normal(0, 0.2)
            self.last_vib = self.last_vib * 0.88 + 1.8 * 0.12 + np.random.normal(0, 0.08)
            self.last_curr = self.last_curr * 0.7 + 50.0 * 0.3 + np.random.normal(0, 0.5)
            self.last_press = self.last_press * 0.85 + 1.5 * 0.15 + np.random.normal(0, 0.02)

            # Снятие показаний с датчиков
            sensor_vib = self.last_vib
            sensor_temp = self.current_temp
            sensor_curr = self.last_curr
            sensor_press = self.last_press
            anom_vib = 0
            anom_temp = 0

            # Помехи на датчиках
            if np.random.rand() < 0.001:
                sensor_vib += np.random.normal(7.0, 1.0) # Разовый всплеск вибрации
                anom_vib = 1
            if np.random.rand() < 0.001:
                sensor_temp += np.random.normal(15.0, 2.0) # Разовый сбой термопары
                anom_temp = 1
                
            self._append_row(2, sensor_vib, sensor_temp, sensor_curr, sensor_press, anom_vib, anom_temp)

    def generate_degradation(self, duration_mins: int):
        """
        Состояние 3 — деградация (износ подшипника).

        Нарастающий тренд моделируется как:
            x[t] = trend[t] + AR(1)-шум поверх тренда

        Это обеспечивает и плавное нарастание (trend), и реалистичный "шевелящийся" 
        сигнал вокруг тренда (AR-компонента).
        Именно такую структуру XAI (SHAP) должен обнаружить как cumulative feature.
        """
        vib_target_trend = np.linspace(1.8, 7.9, duration_mins)
        temp_target_trend = np.linspace(70.0, 92.0, duration_mins)
        curr_target_trend = np.linspace(50.0, 65.0, duration_mins)

        for i in range(duration_mins):
            # AR-процесс плавно подтягивается к растущему тренду
            self.current_temp = self.current_temp * 0.9 + temp_target_trend[i] * 0.1 + np.random.normal(0, 0.2)
            self.last_vib = self.last_vib * 0.88 + vib_target_trend[i] * 0.12 + np.random.normal(0, 0.1)
            self.last_curr = self.last_curr * 0.7 + curr_target_trend[i] * 0.3 + np.random.normal(0, 0.5)
            self.last_press = self.last_press * 0.85 + 1.4 * 0.15 + np.random.normal(0, 0.02)
            
            # В период деградации аппаратные помехи не генерируются, чтобы XAI видел "чистую" проблему
            self._append_row(3, self.last_vib, self.current_temp, self.last_curr, self.last_press, 0)

    def generate_critical(self, duration_mins: int):
        """
        Состояние 4 — критический отказ.
        Устойчивое превышение обоих порогов (вибрация > 8.0, температура > 93°C).
        AR(1) сохраняет "дрожание" сигнала, характерное для сильного износа.
        """

        for _ in range(duration_mins):
            # Система "бьется" в критических значениях, сохраняя инерцию
            self.current_temp = self.current_temp * 0.9 + 96.0 * 0.1 + np.random.normal(0, 0.5)
            self.last_vib = self.last_vib * 0.88 + 9.5 * 0.12 + np.random.normal(0, 0.4)
            self.last_curr = self.last_curr * 0.7 + 75.0 * 0.3 + np.random.normal(0, 2.0)
            self.last_press = self.last_press * 0.85 + 1.1 * 0.15 + np.random.normal(0, 0.1)
            
            self._append_row(4, self.last_vib, self.current_temp, self.last_curr, self.last_press, 0)

    # Основной метод генерации

    def generate_dataset(self) -> pd.DataFrame:
        print(f"Генерация данных: {self.total_days} дней, "
              f"шаг 1 мин → {self.total_minutes:,} строк")

        while self.minutes_generated < self.total_minutes:
            # 80% нормальных циклов, 20% — с отказом
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
                # Цикл с деградацией: здоровый → деградация → критический
                self.generate_healthy(np.random.randint(12 * 60, 24 * 60))
                if self.minutes_generated >= self.total_minutes:
                    break
                self.generate_degradation(np.random.randint(5 * 60, 10 * 60))
                if self.minutes_generated >= self.total_minutes:
                    break
                self.generate_critical(np.random.randint(10, 60))

        df = pd.DataFrame(self.data).iloc[:self.total_minutes]

        self._print_state_distribution(df) # type: ignore
        return df # type: ignore

    @staticmethod
    def _print_state_distribution(df: pd.DataFrame):
        """Краткая статистика по распределению состояний."""
        print("\nРаспределение состояний:")
        counts = df['state_name'].value_counts()
        total  = len(df)
        for name, cnt in counts.items():
            print(f"  {name:<12}: {cnt:>7,} мин  ({100*cnt/total:5.1f}%)")

    # Визуализация

    def plot_smart_episode(self, df: pd.DataFrame, hours: int = 30):
        """
        Строит график окна вокруг последнего зафиксированного отказа.
        Показывает переход Healthy → Degradation → Critical, что является
        ключевым визуальным доказательством работы предиктивной логики.
        """
        critical_indices = df[df['state'] == 4].index
        if len(critical_indices) > 0:
            last_idx = critical_indices[-1]
            end_idx   = min(len(df), last_idx + 60)
            start_idx = max(0, end_idx - hours * 60)
            sample    = df.iloc[start_idx:end_idx]
            print(f"График: окно вокруг отказа (индекс {last_idx})")
        else:
            sample = df.iloc[-hours * 60:]
            print("Отказов не найдено. Построен график последних часов.")

        fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(
            'Имитационная модель МНХВ: окно развития дефекта\n'
            '(AR(1)-процессы обеспечивают реалистичную "память" сигнала)',
            fontsize=14
        )

        # Вибрация
        axs[0].plot(sample['timestamp'], sample['vibration'], color='steelblue', lw=0.8)
        axs[0].axhline(self.VIB_WARNING,  color='orange', ls='--', lw=1.2,
                       label=f'Warning ({self.VIB_WARNING} мм/с)')
        axs[0].axhline(self.VIB_CRITICAL, color='red',    ls='--', lw=1.2,
                       label=f'Critical ({self.VIB_CRITICAL} мм/с)')
        # Помечаем точки-помехи
        noise_pts_vib = sample[sample['anomaly_vibration'] == 1]
        axs[0].scatter(noise_pts_vib['timestamp'], noise_pts_vib['vibration'],
                       color='purple', zorder=5, s=20, label='Помеха датчика')
        axs[0].set_ylabel('Вибрация (мм/с)')
        axs[0].legend(fontsize=8)
        axs[0].grid(True, alpha=0.4)

        # Температура
        axs[1].plot(sample['timestamp'], sample['temperature'], color='tomato', lw=0.8)
        axs[1].axhline(self.TEMP_WARNING,  color='orange', ls='--', lw=1.2,
                       label=f'Warning ({self.TEMP_WARNING} °C)')
        axs[1].axhline(self.TEMP_CRITICAL, color='darkred', ls='--', lw=1.2,
                       label=f'Critical ({self.TEMP_CRITICAL} °C)')
        noise_pts_temp = sample[sample['anomaly_temperature'] == 1]
        axs[1].scatter(noise_pts_temp['timestamp'], noise_pts_temp['temperature'],
                       color='purple', zorder=5, s=20, label='Помеха датчика')
        axs[1].set_ylabel('Температура (°C)')
        axs[1].legend(fontsize=8)
        axs[1].grid(True, alpha=0.4)

        # Ток
        axs[2].plot(sample['timestamp'], sample['current'], color='darkorange', lw=0.8)
        axs[2].set_ylabel('Ток (А)')
        axs[2].grid(True, alpha=0.4)

        # Состояние
        color_map = {0: 'grey', 1: 'blue', 2: 'green', 3: 'orange', 4: 'red'}
        for _, row in sample.iterrows():
            axs[3].axvline(row['timestamp'],
                           color=color_map.get(row['state'], 'black'),
                           alpha=0.15, lw=1)
        axs[3].plot(sample['timestamp'], sample['state'], color='black',
                    drawstyle='steps-post', lw=1.5)
        axs[3].set_yticks([0, 1, 2, 3, 4])
        axs[3].set_yticklabels(['Off', 'Startup', 'Healthy', 'Degradation', 'Critical'])
        axs[3].set_ylabel('Состояние')
        axs[3].grid(True, alpha=0.4)

        plt.tight_layout()

        os.makedirs(self.graph_dir, exist_ok=True)
        plot_path = os.path.join(self.graph_dir, 'simulation_plot_targeted.png')
        plt.savefig(plot_path, dpi=150)
        print(f"График сохранён: {plot_path}")
        plt.show()


# Точка входа

# Генерация датасета для 1 насоса
# if __name__ == "__main__":
#     np.random.seed(42)  # фиксируем seed для воспроизводимости
 
#     gen = PumpDataGenerator(start_date='2026-04-01 00:00:00', total_days=60)
#     df  = gen.generate_dataset()
#     gen.plot_smart_episode(df, hours=30)

# Генерация датасета для n-числа насосов
if __name__ == "__main__":
    n = 5  # изменяемый параметр - количество насосов в парке
    fleet_dfs = [] # список для хранения датафреймов каждого насоса
    
    for i in range(1, n + 1):
        # Меняем seed для каждого насоса (42, 84, 126...)
        np.random.seed(42 * i)
        
        # Динамическое имя: :03d сделает 1 -> 001, 10 -> 010
        pump_id = f'MNHV_{i:03d}' 
        
        print(f"\n--- Генерация данных для {pump_id} ---")
        generator = PumpDataGenerator(pump_id=pump_id, start_date='2026-04-01 00:00:00', total_days=60)
        df = generator.generate_dataset()
        
        # График только для первого насоса
        if i == 1:
            print(f"Построение демонстрационного графика для {pump_id}...")
            generator.plot_smart_episode(df, hours=30)
            
        # Сохранение датафрейма насоса в общий список
        fleet_dfs.append(df)
        
    # Объединение списка датафреймов в единый лог предприятия
    print("\nСборка единого датасета предприятия...")
    enterprise_df = pd.concat(fleet_dfs, ignore_index=True)
    
    # Сохранение финального файла
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(project_root, 'data', 'raw'), exist_ok=True)
    enterprise_path = os.path.join(project_root, 'data', 'raw', 'enterprise_pump_fleet.csv')
    enterprise_df.to_csv(enterprise_path, index=False)
    
    print(f"Парк оборудования успешно сгенерирован!")
    print(f"Всего насосов: {n} | Всего строк: {len(enterprise_df)}")