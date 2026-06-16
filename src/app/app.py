"""
app.py — пользовательский интерфейс платформы предиктивного управления.

ДВА РАЗНЫХ ИНТЕРФЕЙСА (NAMUR NE 129):

  ОПЕРАТОР — процессная осведомлённость по всему парку в реальном времени.
    Карта оборудования (плитки-статусы NE 107); drill-down в агрегат с
    интерактивными графиками всех параметров (Plotly, ховер со значением);
    предписание агента — потоковый тост в правом нижнем углу (сворачивается).

  ИНЖЕНЕР — событийный диагностический console: SHAP обеих моделей,
    симптомы, трассировка RAG, план/история ТОиР. Появляется при инциденте.

Левый нативный сайдбар = «язычок», сжимающий экран: история предписаний +
сценарий (валидация). Запуск:  streamlit run src/app/app.py
"""

import os
import re
import sys
import html
import time

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# разрешение путей (app.py лежит в src/app/) 
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.runtime.platform_backend import (PlatformBackend, ProtoBackend,
                                          THRESHOLDS, PARAM_LABELS, TickResult)
from src.runtime.alarm_runtime import (PumpAlarmFSM, SEVERITY_LABELS, 
                                       FAULT_LABELS, STAGE_BY_SEVERITY)
from experiments.data_stream.demo_stream import (extract_demo_scenario, 
                                                 ScenarioPlayer)

NE107 = {
    "good":        {"color": "#1E8A3C", "icon": "●", "label": "Норма"},
    "out_of_spec": {"color": "#E0A800", "icon": "▲", "label": "Выход за пределы"},
    "maintenance": {"color": "#1565C0", "icon": "⬒", "label": "Требуется ТО"},
    "failure":     {"color": "#C62828", "icon": "✖", "label": "Отказ"},
    "check":       {"color": "#EF6C00", "icon": "◌", "label": "Пуск / простой"},
    "offline":     {"color": "#5A5F66", "icon": "—", "label": "Нет данных"},
}
SEV_TO_NE107 = {0: "good", 1: "out_of_spec", 2: "failure"}
WINDOW_MIN = 120

st.set_page_config(page_title="Платформа предиктивного управления",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container {padding-top:.8rem !important; padding-bottom:1.5rem !important;}
header[data-testid="stHeader"] {height:0; background:transparent;}

.app-title {font-size:1.1rem; font-weight:800; line-height:1.12; margin:0 0 1px;}
.app-sub   {font-size:.7rem; color:#9aa0a6; margin:0 0 6px;}

/* кнопки ролей: без зазора; активная = primary (синяя) */
.role-wrap {margin-bottom:2px;}
.role-wrap [data-testid="stButton"] {margin-bottom:0 !important;}
.role-wrap [data-testid="stButton"] button {border-radius:0; font-weight:700;}
.role-wrap [data-testid="stButton"]:first-of-type button {border-radius:7px 7px 0 0;}
.role-wrap [data-testid="stButton"]:last-of-type  button {border-radius:0 0 7px 7px; border-top:none;}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background:#3d6fb4 !important; border-color:#3d6fb4 !important; color:#fff !important;}

/* плитки оборудования */
.tile {border-radius:10px; padding:13px 14px 11px; color:#fff; min-height:92px;
       box-shadow:0 1px 3px rgba(0,0,0,.35); margin-bottom:4px;}
.tile .tid {font-size:1.0rem; font-weight:800; letter-spacing:.4px;}
.tile .tst {font-size:.8rem; opacity:.95; margin-top:2px;}
.tile .tx  {font-size:.72rem; opacity:.9; margin-top:8px;}

.vbadge {font-size:.72rem; color:#c8ccd1; margin:0 0 2px 2px;}
.vbadge b {color:#fff;}

/* потоковый тост предписания — фиксирован справа снизу, сворачивается */
.st-key-presctoast {position:fixed; right:16px; bottom:16px; width:430px;
    max-width:44vw; max-height:62vh; overflow-y:auto; z-index:1000;
    background:#10151a; border:1px solid #44515c; border-left:5px solid #C62828;
    border-radius:10px; box-shadow:0 8px 28px rgba(0,0,0,.55); padding:8px 11px;}
.st-key-presctoast.warn {border-left-color:#E0A800;}
.presc-head {font-family:monospace; font-size:.74rem; color:#e8eef2;
             white-space:pre-wrap; margin-bottom:6px;}
.presc-sec {margin-bottom:7px;}
.presc-sh  {color:#9fc6ff; font-weight:700; font-size:.74rem;}
.presc-sb  {color:#dfe4e8; font-size:.8rem; white-space:pre-wrap;}
.st-key-presctoast [data-testid="stButton"] button {padding:.12rem .5rem; font-size:.74rem;}
</style>
""", unsafe_allow_html=True)


def init_state():
    ss = st.session_state
    ss.setdefault("backend", None)
    ss.setdefault("backend_kind", None)
    ss.setdefault("fsm", PumpAlarmFSM())
    ss.setdefault("player", None)
    ss.setdefault("playing", False)
    ss.setdefault("speed", 5)
    ss.setdefault("history", {})
    ss.setdefault("last_tick", {})
    ss.setdefault("sim_ts", None)
    ss.setdefault("role", "Оператор")
    ss.setdefault("selected_pump", None)
    ss.setdefault("pending_stream", None)    # (pump, stage) — ждёт генерации
    ss.setdefault("toast_collapsed", False)


def get_backend():
    ss = st.session_state
    if ss.backend is None:
        try:
            ss.backend = PlatformBackend()
            ss.backend_kind = "боевой"
        except Exception as e:
            ss.backend = ProtoBackend()
            ss.backend_kind = f"прототип ({type(e).__name__})"
    return ss.backend


def push_history(pump_id, row, keep=600):
    h = st.session_state.history.setdefault(pump_id, [])
    h.append({k: row.get(k) for k in
              ("timestamp", "vibration", "temperature", "current",
               "pressure", "state")})
    if len(h) > keep:
        del h[: len(h) - keep]


def advance_stream(n_rows):
    """Прогон n строк. На эскалации НЕ генерирует здесь (иначе экран замёрзнет
    до отрисовки) — ставит pending_stream, дашборд рисуется, тост стримит позже."""

    ss = st.session_state
    backend, fsm, player = ss.backend, ss.fsm, ss.player
    if player is None or player.finished:
        ss.playing = False
        return
    for row in player.next_rows(n_rows):
        pump_id, ts = str(row["pump_id"]), str(row["timestamp"])
        ss.sim_ts = ts
        push_history(pump_id, row)
        tick = backend.process_tick(pump_id, row)
        ss.last_tick[pump_id] = tick
        if not tick.ready:
            continue
        trigger = fsm.update(pump_id, ts, tick.severity,
                             suppressed=tick.suppressed,
                             fault_type=tick.fault_type)
        if trigger is not None:
            ss.playing = False
            ss.pending_stream = (pump_id, trigger.stage)
            ss.toast_collapsed = False
            break
    if player.finished:
        ss.playing = False


def minutes_since(ts_from, ts_to):
    try:
        d = pd.Timestamp(ts_to) - pd.Timestamp(ts_from)
        return max(0, int(d.total_seconds() // 60))
    except Exception:
        return 0


def status_key_for(pump_id):
    ss = st.session_state
    tick = ss.last_tick.get(pump_id)
    if tick is None:
        return "offline"
    raw_state = ss.history[pump_id][-1]["state"] if ss.history.get(pump_id) else 2
    if int(raw_state) in (0, 1):
        return "check"
    if not tick.ready:
        return "offline"
    return SEV_TO_NE107[ss.fsm.state(pump_id)]


_SECTION_RE = re.compile(
    r"(ОБОРУДОВАНИЕ:|СТАТУС:|ДИАГНОЗ[^:]*:|ПРЕДПИСАНИЕ[^:]*:|"
    r"РЕКОМЕНДАЦИ[ИЯ][^:]*:|ПЛАНОВЫЙ РЕМОНТ:|ИСТОЧНИКИ:|РЕКОМЕНДАЦИЯ:)", re.I)


def split_sections(text):
    if not text:
        return []
    idxs = [(m.start(), m.group(1)) for m in _SECTION_RE.finditer(text)]
    if not idxs:
        return [("", text.strip())]
    out = []
    if idxs[0][0] > 0:
        out.append(("__header__", text[:idxs[0][0]].strip()))
    for i, (pos, _lab) in enumerate(idxs):
        end = idxs[i + 1][0] if i + 1 < len(idxs) else len(text)
        chunk = text[pos:end].strip()
        m = re.match(r"([^:]+:)\s*(.*)", chunk, re.S)
        out.append((m.group(1).strip(), m.group(2).strip()) if m else ("", chunk))
    return out


def sections_html(text):
    parts = []
    for h, b in split_sections(text):
        if h == "__header__":
            continue
        if not h:
            parts.append(f"<div class='presc-sb'>{html.escape(b)}</div>")
            continue
        parts.append(f"<div class='presc-sec'><span class='presc-sh'>"
                     f"{html.escape(h)}</span><div class='presc-sb'>"
                     f"{html.escape(b)}</div></div>")
    return "".join(parts)


# Интерактивные графики оператора (Plotly: ховер со значением)
def _plotly_axis(series, thr, label, unit):
    n = len(series)
    x = list(range(n))
    ymin, ymax = float(min(series)), float(max(series))
    pad = max(1e-6, (ymax - ymin) * 0.08)
    lo, hi = ymin - pad, ymax + pad
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=list(series), mode="lines",
                             line=dict(color="#e3e7ea", width=1.6),
                             hovertemplate="%{y:.2f} " + unit + "<extra></extra>"))
    for key, col in (("warning", "#E0A800"), ("critical", "#C62828")):
        tv = thr[key]
        if lo <= tv <= hi:
            fig.add_hline(y=tv, line=dict(color=col, width=1, dash="dash"))
    fig.update_layout(template="plotly_dark", height=190,
                      margin=dict(l=44, r=10, t=6, b=22), showlegend=False,
                      paper_bgcolor="#0e1318", plot_bgcolor="#0e1318",
                      xaxis=dict(range=[0, max(1, n - 1)], gridcolor="#23282e",
                                 title=None),
                      yaxis=dict(range=[lo, hi], gridcolor="#23282e",
                                 title=dict(text=f"{label}, {unit}",
                                            font=dict(size=10))))
    return fig


def value_badge(pump_id, param):
    h = st.session_state.history.get(pump_id, [])
    if not h:
        return ""
    cur = h[-1].get(param)
    thr = THRESHOLDS[param]
    return (f"<div class='vbadge'>{PARAM_LABELS[param]}: <b>{cur:.2f}</b> "
            f"{thr['unit']} &nbsp;·&nbsp; пред {thr['warning']} · "
            f"авар {thr['critical']}</div>")


def render_pump_graphs(pump_id, params):
    h = st.session_state.history.get(pump_id, [])
    if len(h) < 2:
        st.info("Поток данных ещё не запущен.")
        return
    df = pd.DataFrame(h).tail(WINDOW_MIN)
    cells = st.columns(2)
    for i, p in enumerate(params):
        with cells[i % 2]:
            st.markdown(value_badge(pump_id, p), unsafe_allow_html=True)
            if HAS_PLOTLY:
                st.plotly_chart(_plotly_axis(df[p].values, THRESHOLDS[p],
                                             PARAM_LABELS[p], THRESHOLDS[p]["unit"]),
                                use_container_width=True,
                                config={"displayModeBar": False},
                                key=f"plt_{pump_id}_{p}")
            else:
                st.line_chart(df[[p]], height=190)


# Потоковый тост предписания (правый нижний угол; сворачивается)
def render_active_prescription():
    ss = st.session_state
    fsm = ss.fsm
    pend = ss.pending_stream

    if pend is not None:
        pump_id, stage = pend
        inc = fsm.incident(pump_id)
        if inc is None:
            ss.pending_stream = None
            return
        warn = "warn" if stage == 1 else ""
        box = st.container(key="presctoast")
        # тег-класс warn вешаем поверх контейнера
        st.markdown(f"<style>.st-key-presctoast{{}}</style>", unsafe_allow_html=True)
        with box:
            st.markdown(
                f"<div class='presc-head'>АГРЕГАТ {pump_id} · "
                f"{SEVERITY_LABELS[stage]} · "
                f"{FAULT_LABELS.get(inc.fault_type or '', '')} · {inc.stage_ts}"
                f"</div>", unsafe_allow_html=True)
            sv = ss.backend.explain(pump_id, inc.stage_ts, stage)
            inc.symptom_vectors[stage] = sv
            stage_key = STAGE_BY_SEVERITY[stage]
            full = st.write_stream(ss.backend.prescription_stream(sv, stage_key))
        inc.prescriptions[stage] = full
        inc.retrieval_traces[stage] = ss.backend.retrieval_trace(sv, stage_key)
        ss.pending_stream = None
        st.rerun()                      # переключиться на чистый секционный вид
        return

    active = [fsm.incident(p) for p in ss.history if fsm.incident(p)]
    active = [i for i in active if i and i.stage in i.prescriptions]
    if not active:
        return
    inc = max(active, key=lambda i: i.stage)
    box = st.container(key="presctoast")
    with box:
        if ss.toast_collapsed:
            c = st.columns([5, 2])
            c[0].markdown(
                f"<div class='presc-head'>📋 {inc.pump_id} · "
                f"{inc.stage_label}</div>", unsafe_allow_html=True)
            if c[1].button("Развернуть", key="toast_exp"):
                ss.toast_collapsed = False
                st.rerun()
        else:
            head = (f"АГРЕГАТ {inc.pump_id} · {inc.stage_label} · "
                    f"{inc.fault_label} · ПРОШЛО "
                    f"{minutes_since(inc.stage_ts, ss.sim_ts or inc.stage_ts)} мин")
            st.markdown(f"<div class='presc-head'>{html.escape(head)}</div>"
                        f"{sections_html(inc.prescriptions[inc.stage])}",
                        unsafe_allow_html=True)
            c = st.columns([3, 2, 2])
            if not inc.acknowledged:
                if c[0].button("Квитировать", key="toast_ack", type="primary"):
                    fsm.acknowledge(inc.pump_id, ss.sim_ts or "")
                    st.rerun()
            else:
                c[0].caption("Квитировано")
            if c[2].button("Свернуть", key="toast_col"):
                ss.toast_collapsed = True
                st.rerun()


# Сайдбар = левый «язычок» (сжимает экран): история + сценарий-валидация
def render_sidebar():
    ss = st.session_state
    with st.sidebar:
        st.markdown("### История предписаний")
        incs = [i for i in ss.fsm.all_incidents() if i.prescriptions]
        if not incs:
            st.caption("Предписаний пока нет.")
        for inc in incs:
            text = inc.prescriptions.get(inc.stage) or \
                   next(iter(inc.prescriptions.values()), "")
            with st.expander(f"№{inc.incident_id} · {inc.pump_id} · "
                             f"{inc.stage_label}"):
                for h, b in split_sections(text):
                    if h == "__header__":
                        continue
                    if h:
                        st.markdown(f"**{h}**")
                    st.text(b)            # st.text — без markdown-перенумерации

        st.divider()
        with st.expander("Сценарий и воспроизведение (валидация)",
                         expanded=ss.player is None):
            dataset = st.text_input("Сырой датасет",
                                    "data/raw/industrial_pumps_dataset.csv")
            fault = st.selectbox("Тип отказа",
                                 ["overheat", "cavitation", "electrical"],
                                 format_func=lambda k: FAULT_LABELS[k])
            if st.button("Собрать сценарий", use_container_width=True):
                try:
                    scen = extract_demo_scenario(dataset, fault)
                    ss.player = ScenarioPlayer(scen)
                    ss.fsm = PumpAlarmFSM()
                    ss.history, ss.last_tick = {}, {}
                    ss.selected_pump, ss.pending_stream = None, None
                    ss.backend.preproc.reset()
                    for row in ss.player.skip_warmup():
                        pid = str(row["pump_id"])
                        push_history(pid, row)
                        ss.last_tick[pid] = ss.backend.process_tick(pid, row)
                    st.success(f"Готово: {len(scen)} мин, {scen['pump_id'].iloc[0]}")
                except Exception as e:
                    st.error(f"Не удалось собрать сценарий: {e}")

            if ss.player is not None:
                st.progress(ss.player.progress,
                            text=f"Поток: {ss.player.pos}/{len(ss.player)} мин")
                b = st.columns(3)
                if b[0].button("▶"): ss.playing = True
                if b[1].button("⏸"): ss.playing = False
                if b[2].button("⏭"): advance_stream(1)
                ss.speed = st.slider("Минут за тик UI", 1, 20, ss.speed)


def render_header_and_roles():
    ss = st.session_state
    hc = st.columns([4, 6])
    with hc[0]:
        st.markdown("<div class='app-title'>Платформа<br>предиктивного "
                    "управления</div><div class='app-sub'>Интерфейс · "
                    "NAMUR NE 129</div>", unsafe_allow_html=True)
        st.markdown("<div class='role-wrap'>", unsafe_allow_html=True)
        if st.button("Оператор", key="role_op", use_container_width=True,
                     type="primary" if ss.role == "Оператор" else "secondary"):
            ss.role = "Оператор"; st.rerun()
        if st.button("Инженер", key="role_en", use_container_width=True,
                     type="primary" if ss.role == "Инженер" else "secondary"):
            ss.role = "Инженер"; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption(f"Ядро: {ss.backend_kind}")


def view_operator():
    ss = st.session_state
    fsm = ss.fsm
    pumps = sorted(ss.history.keys())
    st.subheader("Карта оборудования")

    if not pumps:
        st.info("Запусти демо-сценарий в левой панели (открой её язычком »).")
        render_active_prescription()
        return

    k = st.columns(4)
    k[0].metric("Активные аварии", fsm.active_alarm_count())
    k[1].metric("Предупреждения", fsm.active_warning_count())
    k[2].metric("Подавлено", fsm.journal.count("suppressed"),
                help="Сигналы пуска/простоя. Скрыты, но в архиве (ФЗ-116).")
    k[3].metric("Переходов", sum(1 for e in fsm.journal.events
                                 if e.kind == "transition"),
                help="Норматив ISA 18.2 — ≤6 тревог/час.")

    cols = st.columns(max(4, len(pumps)))
    for col, pid in zip(cols, pumps):
        s = NE107[status_key_for(pid)]
        inc = fsm.incident(pid)
        extra = FAULT_LABELS.get(inc.fault_type or "", "") if inc and inc.fault_type else ""
        col.markdown(f"<div class='tile' style='background:{s['color']}'>"
                     f"<div class='tid'>{s['icon']} {pid}</div>"
                     f"<div class='tst'>{s['label']}</div>"
                     f"<div class='tx'>{extra}</div></div>", unsafe_allow_html=True)
        if col.button("Открыть", key=f"open_{pid}", use_container_width=True):
            ss.selected_pump = pid
            st.rerun()

    if ss.selected_pump and ss.selected_pump in pumps:
        pid = ss.selected_pump
        st.divider()
        head = st.columns([6, 1.4])
        head[0].markdown(f"#### Агрегат {pid}")
        if head[1].button("← к парку", use_container_width=True):
            ss.selected_pump = None
            st.rerun()
        inc = fsm.incident(pid)
        if inc and inc.stage == 1:
            sv = inc.symptom_vectors.get(1)
            pw = (sv.probabilities[1] * 100 if sv and len(sv.probabilities) > 2 else None)
            drv = (sv.fault_top_symptoms[0].feature
                   if sv and getattr(sv, "fault_top_symptoms", None) else "—")
            st.warning(
                f"Ранний сигнал деградации · {FAULT_LABELS.get(inc.fault_type or '', '')}"
                f"{f' · P(деградация) {pw:.0f}%' if pw else ''}. "
                f"Абсолютные значения параметров в пределах нормы — обнаружена "
                f"статистическая сигнатура развивающегося дефекта "
                f"(ведущий признак: {drv}). Порог не достигнут; требуются "
                f"упреждающие действия, не аварийный останов.")
        render_pump_graphs(pid, ["vibration", "temperature", "current", "pressure"])

    render_active_prescription()


def view_engineer():
    ss = st.session_state
    fsm = ss.fsm
    st.subheader("Инженерная диагностика")
    incidents = [i for i in fsm.all_incidents() if i.symptom_vectors]
    if not incidents:
        st.info("Активных инцидентов нет. Диагностические данные появляются "
                "при возникновении предупреждения или аварии.")
        return
    labels = [f"№{i.incident_id} · {i.pump_id} · {i.stage_label} · {i.fault_label}"
              for i in incidents]
    idx = st.selectbox("Инцидент", range(len(incidents)),
                       format_func=lambda j: labels[j], key="eng_inc")
    inc = incidents[idx]
    sv = inc.symptom_vectors.get(inc.stage) or next(iter(inc.symptom_vectors.values()))

    c = st.columns(3)
    c[0].metric("Стадия", inc.stage_label)
    c[1].metric("Тип отказа", inc.fault_label)
    c[2].metric("Уверенность типа", f"{getattr(sv, 'fault_confidence', 0):.0f}%")
    if inc.stage == 1:
        st.caption("Стадия «Предупреждение»: абсолютные значения параметров могут "
                   "быть в норме — сигнал в статистической сигнатуре, не в пороге.")

    tabs = st.tabs(["Диагностика (SHAP)", "Симптомы", "Трассировка RAG",
                    "ТОиР: план и история"])
    with tabs[0]:
        f_sev, f_fault = ss.backend.shap_figures(inc.pump_id)
        if not (f_sev or f_fault):
            st.caption("SHAP-графики доступны в боевом режиме ядра.")
        gc = st.columns([1, 6, 1])      # центрируем и ограничиваем ширину
        with gc[1]:
            if f_fault:
                st.image(f_fault, use_container_width=True,
                         caption="Почему классификатор выбрал этот тип отказа")
            if f_sev:
                st.image(f_sev, use_container_width=True,
                         caption="Вклад признаков по классу «Авария» "
                                 "(удалённость от аварии)")
    with tabs[1]:
        for attr, title in (("fault_top_symptoms", "Признаки типа отказа"),
                            ("top_symptoms", "Признаки тяжести (класс «Авария»)")):
            items = getattr(sv, attr, None)
            if items:
                st.markdown(f"**{title}**")
                st.table(pd.DataFrame([
                    {"Признак": getattr(s, "feature", "—"),
                     "Датчик": getattr(s, "sensor", "—"),
                     "Значение": round(float(getattr(s, "value", 0)), 3),
                     "SHAP": round(float(getattr(s, "shap_weight", 0)), 3)}
                    for s in items]))
    with tabs[2]:
        trace = inc.retrieval_traces.get(inc.stage) or \
            ss.backend.retrieval_trace(sv, STAGE_BY_SEVERITY.get(inc.stage, "critical"))
        if trace:
            st.dataframe(pd.DataFrame(trace), use_container_width=True, hide_index=True)
            st.caption("Источники по разделам: диагноз — мануал/ГОСТ/вибродиагностика; "
                       "предписание и ТОиР — регламент; плановый ремонт — график ППР.")
        else:
            st.caption("Трассировка доступна в боевом режиме ядра.")
    with tabs[3]:
        st.markdown("**Плановый ремонт (из графика ППР)**")
        text = inc.prescriptions.get(inc.stage, "")
        m = re.search(r"ПЛАНОВЫЙ РЕМОНТ:\s*(.+)", text, re.S)
        st.write(m.group(1).strip()[:400] if m else "— нет данных графика —")
        st.markdown("**История работ по агрегату** _(подключается к системе ТОиР "
                    "предприятия; здесь — демонстрационные данные)_")
        st.dataframe(pd.DataFrame([
            {"Дата": "2026-02-11", "Работа": "ТО-1: замена смазки картера",
             "Статус": "выполнено"},
            {"Дата": "2025-11-03", "Работа": "ТО-2: лазерная центровка валов",
             "Статус": "выполнено"},
        ]), use_container_width=True, hide_index=True)


def main():
    init_state()
    get_backend()

    if st.session_state.playing and st.session_state.pending_stream is None:
        advance_stream(st.session_state.speed)

    render_sidebar()
    render_header_and_roles()
    st.divider()

    if st.session_state.role == "Оператор":
        view_operator()
    else:
        view_engineer()
        render_active_prescription()

    if st.session_state.playing and st.session_state.pending_stream is None:
        time.sleep(0.6)
        st.rerun()


if __name__ == "__main__":
    main()