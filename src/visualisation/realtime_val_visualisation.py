"""
Метрики и графики real-time валидации (Validation Viz)
======================================================
src/visualisation/realtime_val_visualisation.py

Назначение: из лога прогона реального времени строить ПУБЛИКАЦИОННЫЕ графики
для текста диссертации и защиты, а также считать сводные метрики. Главный из
них - «лавина тревог»: сколько сигналов сгенерировал бы наивный пороговый
алармер и сколько из них система погасила окнами / состоянием / дебаунсом,
оставив оператору лишь подтверждённые тревоги. Это прямое доказательство цели
работы.

КОНТРАКТ ВХОДА - единый per-tick лог (pandas.DataFrame), колонки:
    timestamp        datetime  - модельное время тика
    pump_id          str
    true_state       int 0..4  - ИСТИНА генератора (Off/Startup/Healthy/
                                 Degradation/Critical)
    true_fault       str       - overheat/cavitation/electrical/none
    sev_detected     int       - что выдала модель: 0/1/2; -1 если тик не готов
    suppressed       bool      - сигнал подавлен (пуск/простой и т.п.)
    suppress_reason  str       - 'anomaly'/'startup_idle'/'debounce'/'' 
    raw_fire         bool      - наивный пороговый алармер сработал бы на этом тике
    escalated        bool      - FSM подтвердил НОВЫЙ переход в тревогу на этом тике

Лог накапливается коллектором ValidationCollector по ходу прогона (хедлесс-
харнесс или advance_stream), затем передаётся в построители. Визуализатор
полностью автономен от моделей - на нём же гоняется самотест на синтетике.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, cast

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import ISA_ALARM_RATE_LIMIT   # предел тревог/час (ISA 18.2)

# Палитра, согласованная с интерфейсом (NE 107 / тяжесть)
C_NORM, C_WARN, C_ALARM = "#1E8A3C", "#E0A800", "#C62828"
C_GRAY, C_BLUE = "#5A5F66", "#1565C0"
SEV_COLORS = {0: C_NORM, 1: C_WARN, 2: C_ALARM}
SEV_LABELS = {0: "Норма", 1: "Предупреждение", 2: "Авария"}
# Истинное состояние генератора → класс тяжести (как в обучении)
STATE_TO_SEV = {2: 0, 3: 1, 4: 2}   # Off/Startup (0,1) - не оцениваются

LOG_COLUMNS = ["timestamp", "pump_id", "true_state", "true_fault",
               "sev_detected", "suppressed", "suppress_reason",
               "raw_fire", "escalated", "anomaly"]


def _style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 165, "savefig.bbox": "tight",
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    })


# Коллектор лога

@dataclass
class ValidationCollector:
    """Накопитель per-tick лога. Вызывается по ходу прогона.

    Пороговые значения - для расчёта «наивного» raw_fire (что сработало бы без
    окон/дебаунса/состояния). Берутся из config.settings.THRESHOLDS.
    """
    vib_warn: float = 3.0
    temp_warn: float = 82.0
    _rows: List[dict] = field(default_factory=list)

    @classmethod
    def from_settings(cls) -> "ValidationCollector":
        try:
            from config.settings import THRESHOLDS
            return cls(vib_warn=THRESHOLDS["vibration"]["warning"],
                       temp_warn=THRESHOLDS["temperature"]["warning"])
        except Exception:
            return cls()

    def add(self, raw_row: dict, sev_detected: int, suppressed: bool,
            escalated: bool, suppress_reason: str = "") -> None:
        # наивный алармер: касание варн-порога ИЛИ флаг аномалии датчика
        anomaly = bool(raw_row.get("anomaly_vibration") or
                       raw_row.get("anomaly_temperature") or
                       raw_row.get("anomaly_current"))
        raw_fire = (float(raw_row.get("vibration", 0)) >= self.vib_warn or
                    float(raw_row.get("temperature", 0)) >= self.temp_warn or
                    anomaly)
        if not suppress_reason:
            if suppressed:
                suppress_reason = "startup_idle"
            elif anomaly and not escalated:
                suppress_reason = "anomaly"
            elif raw_fire and not escalated:
                # повторное срабатывание в уже активной тревоге: наивный алармер
                # фыркал бы каждую минуту, система держит один сигнал на инцидент
                suppress_reason = "debounce"
        self._rows.append({
            "timestamp": pd.Timestamp(raw_row["timestamp"]),
            "pump_id": str(raw_row["pump_id"]),
            "true_state": int(raw_row.get("state", -1)),
            "true_fault": str(raw_row.get("fault_type", "none")),
            "sev_detected": int(sev_detected),
            "suppressed": bool(suppressed),
            "suppress_reason": suppress_reason,
            "raw_fire": bool(raw_fire),
            "escalated": bool(escalated),
            "anomaly": bool(anomaly),
        })

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=LOG_COLUMNS)

# Сводные метрики

def summarize(log: pd.DataFrame) -> dict:
    raw = int(log["raw_fire"].sum())
    confirmed = int(log["escalated"].sum())
    suppressed = max(0, raw - confirmed)
    ratio = (100.0 * suppressed / raw) if raw else 0.0

    # пик тревог в час (ISA 18.2: ≤6/ч на оператора)
    by_min = (log.assign(_a=log["escalated"].astype(int))
              .set_index("timestamp")["_a"].sort_index())
    per_hour = by_min.rolling("60min").sum() if not by_min.empty else by_min
    peak_rate = float(per_hour.max()) if len(per_hour) else 0.0

    lat = detection_latencies(log)
    miss = [e for e in lat if e["latency_min"] is None]
    det = [e for e in lat if e["latency_min"] is not None]
    return {
        "raw_fire": raw, "confirmed": confirmed, "suppressed": suppressed,
        "suppress_ratio_pct": round(ratio, 1),
        "peak_alarms_per_hour": peak_rate,
        "isa_18_2_ok": peak_rate <= ISA_ALARM_RATE_LIMIT,
        "fault_episodes": len(lat),
        "detected_episodes": len(det), "missed_episodes": len(miss),
        "median_latency_min": (round(float(np.median([e["latency_min"] for e in det])), 1)
                               if det else None),
    }


def detection_latencies(log: pd.DataFrame) -> List[dict]:
    """Задержка детектирования по каждому эпизоду деградации.

    Эпизод - непрерывный участок true_state==3 (Degradation) на насосе.
    Задержка = время от начала эпизода до первого sev_detected>=1.
    None - пропуск (FN).
    """
    out: List[dict] = []
    for pid, g in log.sort_values("timestamp").groupby("pump_id"):
        g = g.reset_index(drop=True)
        in_ep = g["true_state"] == 3
        starts = g.index[in_ep & ~in_ep.shift(1, fill_value=False)]
        for s in starts:
            e = s
            while e + 1 < len(g) and g.loc[e + 1, "true_state"] in (3, 4):
                e += 1
            seg = g.loc[s:e]
            onset = seg.iloc[0]["timestamp"]
            warned = seg[seg["sev_detected"] >= 1]
            lat = (None if warned.empty
                   else (warned.iloc[0]["timestamp"] - onset).total_seconds() / 60.0)
            out.append({"pump_id": pid, "fault": seg.iloc[0]["true_fault"],
                        "onset": onset, "latency_min": lat})
    return out

# Построение графиков

def plot_avalanche(log: pd.DataFrame):
    """ГЛАВНЫЙ график: воронка лавины + разбивка подавленного по причинам."""

    _style()
    raw = int(log["raw_fire"].sum())
    confirmed = int(log["escalated"].sum())
    suppressed = max(0, raw - confirmed)
    ratio = (100.0 * suppressed / raw) if raw else 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    bars = ax1.bar(["Сырые\nсрабатывания", "Подтверждённые\nтревоги"],
                   [raw, confirmed], color=[C_GRAY, C_ALARM], width=0.6)
    for b, v in zip(bars, [raw, confirmed]):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:,}".replace(",", " "),
                 ha="center", va="bottom", fontweight="bold")
    ax1.set_title("Гашение «лавины тревог»")
    ax1.set_ylabel("Количество сигналов")
    ax1.set_ylim(0, raw * 1.18 if raw else 1)
    ax1.annotate(f"подавлено {ratio:.1f}%",
                 xy=(1, confirmed + raw * 0.1), xytext=(1, raw * 0.52),
                 ha="center", color=C_BLUE, fontweight="bold", fontsize=12,
                 arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.6))

    by_reason = (log[log["raw_fire"] & ~log["escalated"]]
                 .groupby("suppress_reason").size()
                 .sort_values(ascending=False))   # по часовой: от большего к меньшему
    names = {"anomaly": "Аномалии датчиков (сглажены окнами)",
             "startup_idle": "Пуск / простой (состояние, ФЗ-116)",
             "debounce": "Удержание тревоги (один сигнал на инцидент)",
             "": "Прочее"}
    reason_color = {"anomaly": C_WARN, "startup_idle": C_BLUE,
                    "debounce": C_GRAY, "": "#9AA0A6"}
    if by_reason.sum() > 0:
        total = int(by_reason.sum())
        colors = [reason_color.get(k, "#9AA0A6") for k in by_reason.index]
        # Проценты - на крупных секторах; подписи причин - в легенде снизу
        # (вертикальный список): мелкие сектора 1–6% радиальными подписями
        # налезали друг на друга. Легенда исключает наложение by design.
        wedges, _t, _a = ax2.pie(
            by_reason.values, colors=colors, startangle=90, counterclock=False,
            autopct=lambda p: f"{p:.0f}%" if p >= 8 else "",
            pctdistance=0.62, wedgeprops={"linewidth": 1, "edgecolor": "white"},
            textprops={"fontsize": 13, "fontweight": "bold", "color": "white"})
        legend_labels = [f"{names.get(str(k), str(k))} - {v / total * 100:.0f}%"
                         for k, v in by_reason.items()]
        ax2.legend(wedges, legend_labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.02), fontsize=11.5, frameon=False,
                   handlelength=1.1, labelspacing=0.5)
        ax2.set_title("Какой сигнал подавлен")
    else:
        ax2.axis("off")
    fig.tight_layout()
    return fig


def plot_alarm_rate(log: pd.DataFrame):
    """Интенсивность тревог в час против норматива ISA 18.2 (≤6/ч)."""

    _style()
    s = (log.assign(_a=log["escalated"].astype(int))
         .set_index("timestamp")["_a"].sort_index())
    per_hour = s.rolling("60min").sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(per_hour.index, per_hour.values, color=C_BLUE, lw=1.8,
            label="Подтверждённые тревоги / час")
    lim = ISA_ALARM_RATE_LIMIT
    ax.axhline(lim, color=C_ALARM, ls="--", lw=1.6,
               label=f"Предел ISA 18.2 ({lim}/ч)")
    ax.fill_between(per_hour.index, per_hour.values, lim,
                    where=(per_hour.values > lim), color=C_ALARM, alpha=0.18)
    ax.set_title("Интенсивность тревог (ISA 18.2)")
    ax.set_ylabel("Тревог за скользящий час")
    ax.set_xlabel("Время")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_detection_latency(log: pd.DataFrame):
    """Задержка детектирования по эпизодам, сгруппированная по типу отказа."""

    _style()
    lat = [e for e in detection_latencies(log) if e["latency_min"] is not None]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    if not lat:
        ax.axis("off"); ax.set_title("Задержка детектирования: эпизодов нет")
        return fig
    fault_names = {"overheat": "Перегрев", "cavitation": "Кавитация",
                   "electrical": "Электрика"}
    order = ["overheat", "cavitation", "electrical"]
    data, labels, colors = [], [], []
    pal = {"overheat": C_ALARM, "cavitation": C_BLUE, "electrical": C_WARN}
    for f in order:
        vals = [e["latency_min"] for e in lat if e["fault"] == f]
        if vals:
            data.append(vals); labels.append(f"{fault_names[f]}\n(n={len(vals)})")
            colors.append(pal[f])
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", lw=1.4))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    ax.set_title("Задержка детектирования предупреждения")
    ax.set_ylabel("Минут от начала деградации")
    fig.tight_layout()
    return fig


def plot_confusion(log: pd.DataFrame):
    """Матрица ошибок: истинная тяжесть vs детектированная (по неподавленным)."""

    _style()
    d = log[(~log["suppressed"]) & (log["sev_detected"] >= 0) &
            (log["true_state"].isin(STATE_TO_SEV))].copy()
    d["true_sev"] = d["true_state"].map(STATE_TO_SEV)
    m = np.zeros((3, 3), dtype=int)
    for t, p in zip(d["true_sev"], d["sev_detected"]):
        if 0 <= int(p) <= 2:
            m[int(t), int(p)] += 1
    mn = m / m.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.imshow(mn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([SEV_LABELS[i] for i in range(3)], rotation=15, ha="center")
    ax.set_yticklabels([SEV_LABELS[i] for i in range(3)])
    ax.set_xlabel("Детектировано", fontsize=14)
    ax.set_ylabel("Истина", fontsize=14)
    ax.set_title("Матрица ошибок тяжести", fontsize=16)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mn[i, j]*100:.0f}%\n{m[i, j]}", ha="center",
                    va="center", color="white" if mn[i, j] > 0.5 else "black",
                    fontsize=10)
    ax.grid(False)
    fig.tight_layout()
    return fig


def plot_state_timeline_all(log: pd.DataFrame):
    """Таймлайн по ВСЕМ насосам парка: на каждый насос - пара полос истина/детект."""

    _style()
    pumps = sorted(log["pump_id"].unique())
    n = len(pumps)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.45 * n + 1.0), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, pid in zip(axes, pumps):
        g = cast(pd.DataFrame, log[log["pump_id"] == pid]).sort_values("timestamp")
        t = g["timestamp"].values
        true_sev = g["true_state"].map(STATE_TO_SEV).fillna(-1).values
        det = g["sev_detected"].astype(float).values
        # на остановленном агрегате детект честно отображается как «отключён»:
        # после аварии/останова модель не «читает норму» - насос физически off
        det = np.where(true_sev < 0, -1.0, det)

        def band(y, series, h=0.40):
            for i in range(len(series)):
                v = series[i]
                color = SEV_COLORS.get(int(v), C_GRAY) if v >= 0 else "#2B2F36"
                ax.axvspan(t[i], t[min(i + 1, len(t) - 1)],
                           ymin=y, ymax=y + h, color=color, lw=0)
        band(0.55, true_sev)
        band(0.05, det)
        # фиолетовые точки аномалий на полосе «истина» (как на графике датасета):
        # штатная работа, но система зафиксировала аномалии → повод для ручной проверки
        if "anomaly" in g.columns:
            an = g[g["anomaly"].astype(bool)]
            if len(an):
                ax.scatter(an["timestamp"].values, [0.75] * len(an), s=16,
                           color="#8E24AA", marker="o", zorder=6,
                           edgecolors="white", linewidths=0.4)
        ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.75])
        ax.set_yticklabels(["детект", "истина"], fontsize=12)
        ax.set_ylabel(pid, fontsize=14, rotation=0, ha="right",
                      va="center", labelpad=34, fontweight="bold")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[-1].set_xlabel("Дата (мм-дд) Время (час)", fontsize=14)
    leg: List[Artist] = [Patch(facecolor=SEV_COLORS[k], label=SEV_LABELS[k])
                         for k in (0, 1, 2)]
    leg.append(Patch(facecolor=C_GRAY, label="Пуск / простой"))
    leg.append(Line2D([0], [0], marker="o", color="white", markerfacecolor="#8E24AA",
                      markersize=8, label="Аномалия (истина)", linewidth=0))
    fig.legend(handles=leg, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=10)
    fig.suptitle("Таймлайн состояния по насосам парка", fontweight="bold", fontsize=16)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def render_all(log: pd.DataFrame, outdir: str) -> Dict[str, str]:
    """Строит и сохраняет все графики; возвращает {имя: путь}."""

    os.makedirs(outdir, exist_ok=True)
    figs = {
        "avalanche": plot_avalanche(log),
        "alarm_rate": plot_alarm_rate(log),
        "detection_latency": plot_detection_latency(log),
        "confusion": plot_confusion(log),
        "timeline": plot_state_timeline_all(log),       # все насосы парка
    }
    paths = {}
    for name, fig in figs.items():
        p = os.path.join(outdir, f"validation_{name}.png")
        fig.savefig(p, bbox_inches="tight")   # захватить вынесенную вниз легенду
        plt.close(fig)
        paths[name] = p
    return paths

# Самотест на синтетическом логе

def _synthetic_log(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pumps = [f"MNHV_00{i}" for i in range(1, 6)]
    faults = ["overheat", "cavitation", "electrical"]
    t0 = pd.Timestamp("2026-04-01 00:00:00")
    rows = []
    for pid in pumps:
        state, fault, sev, minute = 2, "none", 0, 0
        seg_left = int(rng.integers(60, 160))
        warned_at = None
        for k in range(600):
            minute += 1
            ts = t0 + pd.Timedelta(minutes=minute)
            if seg_left <= 0:
                if state == 2:
                    if rng.random() < 0.5:
                        state, fault = 3, rng.choice(faults); seg_left = int(rng.integers(90, 140))
                        warned_at = None
                    else:
                        state, fault, seg_left = 0, "none", int(rng.integers(5, 14))
                elif state == 3:
                    state, seg_left = 4, int(rng.integers(10, 28))
                elif state == 4:
                    state, fault, seg_left = 0, "none", int(rng.integers(5, 14))
                elif state == 0:
                    state, seg_left = 1, 3
                elif state == 1:
                    state, fault, seg_left = 2, "none", int(rng.integers(80, 170))
            # детектирование с задержкой
            if state == 3:
                if warned_at is None and rng.random() < 0.12:
                    warned_at = k
                sev = 1 if warned_at is not None else 0
            elif state == 4:
                sev = 2 if rng.random() < 0.96 else 1
            elif state == 2:
                sev = 1 if rng.random() < 0.01 else 0   # редкий FP
            else:
                sev = -1
            suppressed = state in (0, 1)
            escalated = (state == 3 and warned_at == k) or \
                        (state == 4 and sev == 2 and rng.random() < 0.06)
            anomaly = (state == 2 and rng.random() < 0.02)
            vib = 9.5 if anomaly else (8.6 if state == 4 else 1.8)
            temp = 70.0 if state != 4 else 95.0
            rows.append({"timestamp": ts, "pump_id": pid, "state": state,
                         "fault_type": fault, "vibration": vib, "temperature": temp,
                         "anomaly_vibration": int(anomaly), "anomaly_temperature": 0,
                         "anomaly_current": 0, "_sev": sev, "_supp": suppressed,
                         "_esc": escalated})
            seg_left -= 1
    raw = pd.DataFrame(rows)
    coll = ValidationCollector()
    for _, r in raw.iterrows():
        coll.add(r.to_dict(), int(r["_sev"]), bool(r["_supp"]), bool(r["_esc"]))
    return coll.to_frame()


if __name__ == "__main__":
    log = _synthetic_log()
    print(f"Синтетический лог: {len(log):,} строк, {log['pump_id'].nunique()} насосов")
    s = summarize(log)
    print("\nСводные метрики:")
    for k, v in s.items():
        print(f"  {k:<22}: {v}")
    out = os.path.join(_THIS_DIR, "_demo_figures")
    paths = render_all(log, out)
    print("\nГрафики сохранены:")
    for n, p in paths.items():
        print(f"  {n:<18} → {p}")