"""
Блок визуализации фрагмента сырых данных из data_generator.py
=============================================================
Строит 5-панельный график окна вокруг последнего зафиксированного отказа.
Показывает переход Healthy → Degradation → Critical с аннотацией типа отказа
и фоновой подсветкой по fault_type.
Точками на графиках отображаются аппаратные сбои.
Датчик давления не сбоит, т.к. промышленные мембранные преобразователи
конструктивно защищенносты, а вязкая среда нефтепроводов обладает  
демпфирующими свойствами, физически сглаживающими любые мгновенные помехи.

Панели: Вибрация | Температура | Ток | Давление | Состояние
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config.settings import THRESHOLDS

VIB_WARNING = THRESHOLDS['vibration']['warning']
VIB_CRITICAL = THRESHOLDS['vibration']['critical']
TEMP_WARNING = THRESHOLDS['temperature']['warning']
TEMP_CRITICAL = THRESHOLDS['temperature']['critical']

project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
graph_dir = os.path.join(project_root, 'artifacts', 'graphs')

def plot_smart_episode(df: pd.DataFrame, hours: int = 60):

    critical_indices = df[df['state'] == 4].index
    if len(critical_indices) > 0:
        last_idx = critical_indices[-1]
        end_idx = min(len(df), last_idx + 60)
        start_idx = max(0, end_idx - hours * 60)
        sample = df.iloc[start_idx:end_idx]
        print(f"График: окно вокруг отказа (индекс {last_idx})")
    else:
        sample = df.iloc[-hours * 60:]
        print("Отказов не найдено. Построен график последних часов.")

    # Определяется тип отказа в окне для заголовка
    fault_mode = sample[sample['fault_type'] != 'none']['fault_type'].mode()
    fault_label = fault_mode.iloc[0] if not fault_mode.empty else 'none'
    fault_display = {
        'overheat':   'Тип А - перегрев (износ подшипника)',
        'cavitation': 'Тип Б — кавитация (повреждение колеса)',
        'electrical': 'Тип В — аномалия тока (электрика)',
    }.get(fault_label, 'нет отказа')

    fig, axs = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    fig.suptitle(
        f'Имитационная модель МНХВ: окно развития дефекта\n'
        f'{fault_display} | AR(1)-процессы с типизированными сигнатурами',
        fontsize=13, fontweight='bold'
    )

    # Фоновая подсветка по типу отказа
    fault_shade = {
        'overheat': '#FFD0D0',
        'cavitation': '#D0E4FF',
        'electrical': '#EDD0FF',
    }
    for ft, fc in fault_shade.items():
        mask = (sample['fault_type'] == ft).values.astype(int)
        diffs = np.diff(mask, prepend=0, append=0)
        for s, e in zip(np.where(diffs == 1)[0], np.where(diffs == -1)[0]):
            ts_s = sample['timestamp'].iloc[min(s, len(sample) - 1)]
            ts_e = sample['timestamp'].iloc[min(e, len(sample) - 1)]
            for ax in axs:
                ax.axvspan(ts_s, ts_e, color=fc, alpha=0.28, zorder=0)

    anom_vib = sample[sample['anomaly_vibration'] == 1]
    anom_temp = sample[sample['anomaly_temperature'] == 1]
    anom_curr = sample[sample['anomaly_current'] == 1]

    # Панель 0 — Вибрация
    axs[0].plot(sample['timestamp'], sample['vibration'], color='steelblue', lw=0.8)
    axs[0].axhline(VIB_WARNING,  color='orange', ls='--', lw=1.2,
                    label=f'Warning ({VIB_WARNING} мм/с)')
    axs[0].axhline(VIB_CRITICAL, color='red',    ls='--', lw=1.2,
                    label=f'Critical ({VIB_CRITICAL} мм/с)')
    if not anom_vib.empty:
        axs[0].scatter(anom_vib['timestamp'], anom_vib['vibration'],
                        color='purple', zorder=5, s=30, label='Помеха датчика')
    axs[0].set_ylabel('Вибрация (мм/с)')
    axs[0].legend(fontsize=8)
    axs[0].grid(True, alpha=0.4)

    # Панель 1 — Температура
    axs[1].plot(sample['timestamp'], sample['temperature'], color='tomato', lw=0.8)
    axs[1].axhline(TEMP_WARNING,  color='orange',  ls='--', lw=1.2,
                    label=f'Warning ({TEMP_WARNING} °C)')
    axs[1].axhline(TEMP_CRITICAL, color='darkred', ls='--', lw=1.2,
                    label=f'Critical ({TEMP_CRITICAL} °C)')
    if not anom_temp.empty:
        axs[1].scatter(anom_temp['timestamp'], anom_temp['temperature'],
                        color='purple', zorder=5, s=30, label='Помеха датчика')
    axs[1].set_ylabel('Температура (°C)')
    axs[1].legend(fontsize=8)
    axs[1].grid(True, alpha=0.4)

    # Панель 2 — Ток
    axs[2].plot(sample['timestamp'], sample['current'], color='darkorange', lw=0.8)
    if not anom_curr.empty:
        axs[2].scatter(anom_curr['timestamp'], anom_curr['current'],
                        color='purple', zorder=5, s=30, label='Помеха датчика')
    axs[2].set_ylabel('Ток (А)')
    axs[2].legend(fontsize=8)
    axs[2].grid(True, alpha=0.4)

    # Панель 3 — Давление (ключевой сигнал для Типа Б)
    axs[3].plot(sample['timestamp'], sample['pressure'], color='seagreen', lw=0.8)
    axs[3].axhline(1.5, color='grey', ls=':', lw=1.0, label='Норма (1.5 МПа)')
    axs[3].set_ylabel('Давление (МПа)')
    axs[3].legend(fontsize=8)
    axs[3].grid(True, alpha=0.4)

    # Панель 4 — Состояние конечного автомата
    color_map = {0: 'grey', 1: 'blue', 2: 'green', 3: 'orange', 4: 'red'}
    for _, row in sample.iterrows():
        axs[4].axvline(row['timestamp'],
                        color=color_map.get(row['state'], 'black'),
                        alpha=0.15, lw=1)
    axs[4].plot(sample['timestamp'], sample['state'],
                color='black', drawstyle='steps-post', lw=1.5)
    axs[4].set_yticks([0, 1, 2, 3, 4])
    axs[4].set_yticklabels(['Off', 'Startup', 'Healthy', 'Degradation', 'Critical'])
    axs[4].set_ylabel('Состояние')
    axs[4].grid(True, alpha=0.4)

    axs[0].set_xlim(sample['timestamp'].iloc[0], sample['timestamp'].iloc[-1])

    plt.tight_layout()

    os.makedirs(graph_dir, exist_ok=True)
    plot_path = os.path.join(graph_dir, 'DG_plot_alarm_detection.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"График сохранён: {plot_path}")