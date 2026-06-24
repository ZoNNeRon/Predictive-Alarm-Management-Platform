"""
app.py — пользовательский интерфейс платформы предиктивного управления.

ДВА РАЗНЫХ ИНТЕРФЕЙСА (NAMUR NE 129):
  ОПЕРАТОР — процессная осведомлённость по парку в реальном времени: карта
    оборудования (NE 107), drill-down с интерактивными графиками (Plotly, ось
    времени), предписание агента — потоковый тост справа снизу.
  ИНЖЕНЕР — событийная диагностика: SHAP обеих моделей, симптомы, трассировка
    RAG, план/история ТОиР. Список ВСЕХ агрегатов сортируется по тяжести.

АРХИТЕКТУРА РЕАЛЬНОГО ВРЕМЕНИ (важно):
  • Поток данных НЕ останавливается при обнаружении предупреждения/аварии —
    ползунок и графики продолжают идти. advance_stream нигде не снимает playing
    по факту эскалации и не делает break.
  • Медленная генерация предписания (RAG+LLM, ~13 с) вынесена в ФОНОВЫЙ ПОТОК
    и пишет в потокобезопасный модульный словарь _GEN. Главный проход её только
    опрашивает — поэтому UI не «замораживается». Поток трогает RAG/Ollama;
    главный поток — модели/препроцессор (process_tick, explain). Подсистемы
    разные, общих изменяемых объектов нет.
  • Факт инцидента попадает в историю СРАЗУ при обнаружении (как только посчитан
    SymptomVector), ещё до завершения генерации. Текст дописывается потом —
    целиком, не потокенно. Быстрый переход Предупреждение→Авария не «теряет»
    предупреждение.
  • Экраны Оператор/Инженер рендерятся в РАЗНЫХ keyed-контейнерах — Streamlit
    держит их как независимые поддеревья, наложений при переключении нет.

Запуск:  streamlit run src/app/app.py

ОГРАНИЧЕНИЕ (честно): фоновый поток в Streamlit не может сам перерисовать
страницу — обновление идёт только когда главный проход делает rerun. Поэтому
пока идёт генерация ИЛИ воспроизведение, главный цикл переотрисовывается по
таймеру (REFRESH). Это и обеспечивает «живой» поток и дорисовку тоста.
"""

import os
import re
import sys
import html
import time
import threading

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
                                          THRESHOLDS, PARAM_LABELS)
from src.runtime.alarm_runtime import (PumpAlarmFSM, SEVERITY_LABELS, 
                                       FAULT_LABELS, STAGE_BY_SEVERITY)
from experiments.data_stream.demo_stream import (extract_demo_scenario, 
                                                 ScenarioPlayer)
from experiments.realtime_validation.realtime_player import RealtimePlayer
from experiments.realtime_validation.live_generator import LiveMultiPumpGenerator
from src.runtime.online_preprocessor import OnlinePreprocessor
from experiments.realtime_validation.realtime_preprocessor import RealtimeProgressivePreprocessor
from src.visualisation.realtime_val_visualisation import ValidationCollector, render_all, summarize

NE107 = {
    "good":        {"color": "#1E8A3C", "icon": "●", "label": "Норма"},
    "out_of_spec": {"color": "#E0A800", "icon": "▲", "label": "Выход за пределы"},
    "maintenance": {"color": "#1565C0", "icon": "⬒", "label": "Требуется ТО"},
    "failure":     {"color": "#C62828", "icon": "✖", "label": "Отказ"},
    "check":       {"color": "#EF6C00", "icon": "◌", "label": "Пуск / простой"},
    "offline":     {"color": "#5A5F66", "icon": "—", "label": "Нет данных"},
}
SEV_TO_NE107 = {0: "good", 1: "out_of_spec", 2: "failure"}
SEV_COLOR = {0: "#1E8A3C", 1: "#E0A800", 2: "#C62828"}   # норма/пред/авария
ACK_COLOR = "#1E8A3C"                                     # квитировано — зелёный
WINDOW_MIN = 120
RENAG_MIN = 10            # повторное оповещение, если стадия держится (sim-минуты)
REFRESH = 2.0            # период переотрисовки главного прохода (с)
GEN_TIMEOUT = 90.0       # страховочный таймаут генерации предписания (с)
YRANGE = {"vibration": (0, 12), "temperature": (0, 98),
          "current": (0, 150), "pressure": (0, 2)}

st.set_page_config(page_title="Платформа предиктивного управления",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container {padding-top:.4rem !important; padding-bottom:1rem !important;}
h2, .stSubheader {margin-top:.1rem !important; padding-top:0 !important;}
header[data-testid="stHeader"] {height:0; background:transparent;}

.app-title {font-size:1.35rem; font-weight:800; line-height:1.15; margin:0 0 1px;
            white-space:nowrap;}
.dash-switch-label {font-size:.7rem; color:#9aa0a6; font-weight:700;
                    letter-spacing:.5px; margin:6px 0 3px;}
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

/* контейнер тостов — фиксирован справа снизу, прозрачный; внутри карточки стопкой */
.st-key-presctoast {position:fixed; right:16px; bottom:16px; width:430px;
    max-width:44vw; max-height:82vh; overflow-y:auto; z-index:1000;
    background:transparent; padding:0;}
/* карточка-уведомление; цвет левой рамки задаётся динамически по тяжести */
[class*="st-key-toastcard_"] {background:rgba(16,21,26,.8); border:1px solid #44515c;
    border-left:5px solid #C62828; border-radius:9px; backdrop-filter:blur(2px);
    box-shadow:0 6px 20px rgba(0,0,0,.5); padding:5px 9px; margin-bottom:6px;}
.presc-head {font-family:monospace; font-size:.72rem; color:#e8eef2;
             white-space:pre-wrap; margin-bottom:3px; line-height:1.15;}
.presc-sec {margin-bottom:3px;}
.presc-sh  {color:#9fc6ff; font-weight:700; font-size:.72rem;}
.presc-sb  {color:#dfe4e8; font-size:.74rem; white-space:pre-wrap; line-height:1.2;}
.st-key-presctoast [data-testid="stButton"] button {padding:.12rem .5rem; font-size:.74rem;}

/* история предписаний — компактный шрифт заголовка карточки */
[data-testid="stSidebar"] [data-testid="stExpander"] summary p {font-size:.82rem;}
</style>
""", unsafe_allow_html=True)


# ===================================================================== #
# ФОНОВАЯ ГЕНЕРАЦИЯ (потокобезопасный модульный словарь; НЕ session_state)
# ===================================================================== #
@st.cache_resource
def get_gen_store():
    """ЕДИНОЕ хранилище фоновых заданий, переживающее rerun'ы Streamlit.
    Обычные модульные globals при каждом rerun пересоздаются (скрипт
    исполняется заново) — из-за этого поток писал в осиротевший словарь,
    и предписание не доходило ни в тост, ни в историю. cache_resource даёт
    единый объект на всё приложение."""
    return {"jobs": {}, "lock": threading.Lock()}


def _launch_generation(backend, sv, pump_id, stage, force=False):
    store = get_gen_store()
    jobs, lock = store["jobs"], store["lock"]          # стабильные объекты
    key = (pump_id, stage)
    stage_key = STAGE_BY_SEVERITY[stage]
    with lock:
        if key in jobs and not force:
            return
        job = {"partial": "", "text": None, "trace": None,
               "done": False, "started": time.time()}
        jobs[key] = job        # при force старое задание перезаписывается → его поток осиротевает

    def _run():
        acc = []
        try:
            for chunk in backend.prescription_stream(sv, stage_key):
                acc.append(chunk)
                with lock:
                    job["partial"] = "".join(acc)
            text = "".join(acc)
            try:
                trace = backend.retrieval_trace(sv, stage_key) or []
            except Exception:
                trace = []
            with lock:
                job.update(text=text, trace=trace)
        except Exception as e:
            with lock:
                job.update(
                    text=f"[ошибка генерации: {type(e).__name__}: {e}]",
                    trace=[])
        finally:
            with lock:
                job["done"] = True

    threading.Thread(target=_run, daemon=True).start()


def poll_renags():
    """Ре-наг: после квитирования, если проблема РЕАЛЬНО держится дольше RENAG_MIN
    (sim-мин) — новое возникновение (оператор видит, что параметры не в норме).
    Авария — пока агрегат в аварийном останове (tripped). Предупреждение — пока
    severity ещё >= 1. Вернулся в норму — не ре-нагаем."""
    ss = st.session_state
    if not ss.sim_ts:
        return
    inc_pid = {inc.incident_id: inc.pump_id for inc in ss.fsm.all_incidents()}
    for key, ack_ts in list(ss.acked.items()):
        incident_id, stage = key
        pid = inc_pid.get(incident_id)
        if pid is None:
            continue
        # Инцидент ещё ТЕКУЩИЙ для агрегата? Если его сменил новый (tripped/state
        # относятся к НАСОСУ, а acked — к ИНЦИДЕНТУ), старый квитированный инцидент
        # больше не «держится» — он закрыт и остаётся квитированным навсегда.
        # Без этой проверки новая авария на том же насосе ошибочно «ре-нагала» бы
        # старую квитированную (и та уходила в «возвращено в норму»).
        cur = ss.fsm.incident(pid) or ss.pinned.get(pid)
        if cur is None or cur.incident_id != incident_id:
            continue
        if stage == 2:
            still = pid in ss.tripped                 # авария: ещё в останове
        else:
            still = ss.fsm.state(pid) >= 1            # предупреждение: ещё не в норме
        if not still:
            continue
        if minutes_since(ack_ts, ss.sim_ts) >= RENAG_MIN:
            ss.acked.pop(key, None)                  # тост снова покажется
            ev = _find_event(incident_id, stage)     # та же запись — НЕ дублируем строку
            if ev is not None:
                ev["status"] = "active"
                ev["status_ts"] = None

def poll_generations():
    ss = st.session_state
    store = get_gen_store()
    jobs, lock = store["jobs"], store["lock"]
    now = time.time()
    finished = []
    with lock:
        for key, job in list(jobs.items()):
            if not job["done"] and now - job["started"] > GEN_TIMEOUT:
                job.update(done=True,
                           text=job["text"] or "[таймаут генерации]",
                           trace=job["trace"] or [])
            if job["done"]:
                finished.append((key, job.get("text"), job.get("trace")))
    for key, text, trace in finished:
        pump_id, stage = key
        inc = ss.fsm.incident(pump_id) or ss.pinned.get(pump_id)
        if inc is not None:
            inc.prescriptions[stage] = text if text is not None else "[нет текста]"
            inc.retrieval_traces[stage] = trace or []
        with lock:
            jobs.pop(key, None)


def has_pending():
    store = get_gen_store()
    with store["lock"]:
        return len(store["jobs"]) > 0


def gen_partial(key):
    store = get_gen_store()
    with store["lock"]:
        job = store["jobs"].get(key)
        return job["partial"] if job else None


# ===================================================================== #
def init_state():
    ss = st.session_state
    ss.setdefault("backend", None)
    ss.setdefault("backend_kind", None)
    ss.setdefault("fsm", PumpAlarmFSM())
    ss.setdefault("player", None)
    ss.setdefault("playing", False)
    ss.setdefault("speed", 1)
    ss.setdefault("history", {})
    ss.setdefault("last_tick", {})
    ss.setdefault("sim_ts", None)
    ss.setdefault("role", "Оператор")
    ss.setdefault("selected_pump", None)
    ss.setdefault("toast_expanded", set())
    ss.setdefault("toir_log", {})       # журнал выполненных работ по насосам
    ss.setdefault("acked", {})               # (incident_id, stage) -> sim_ts квитирования
    # Журнал событий истории — ЕДИНСТВЕННЫЙ источник левой панели «История».
    # Append-only, по одной записи на возникновение (incident_id, stage); статус
    # меняется на месте. Не зависит от мутаций stage_ts инцидента в FSM, поэтому
    # не нужна склейка «близнецов» по временному окну (была хрупкой).
    ss.setdefault("events", [])         # список dict: см. _log_event()
    ss.setdefault("validation", None)
    ss.setdefault("tripped", set())     # насосы в аварийном останове до квитирования
    ss.setdefault("pinned", {})         # закреплённые инциденты: живут до квитирования
    ss.setdefault("shap_frozen", {})    # SHAP-фигуры, замороженные на момент эскалации
    ss.setdefault("anomaly_suppressed", 0)   # сглаженные аномалии (для «Подавлено»)
    ss.setdefault("_presc_ft", {})   # (incident_id, stage) -> тип, под который собрано предписание
    ss.setdefault("_ft_cand", {})    # pid -> (тип-кандидат, тиков подряд)
    ss.setdefault("_deadzone", {})       # pid -> остаток жёсткой мёртвой зоны после пуска
    ss.setdefault("recovering", set())   # квитированные аварии: гасим стейл fsm.state=2


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


def _on_escalation(pump_id, stage):
    """Обнаружена эскалация. СРАЗУ: считаем SymptomVector (быстро, главный поток),
    помечаем инцидент как обнаруженный (→ он мгновенно появляется в истории и в
    тосте) и запускаем фоновую генерацию текста. Поток данных НЕ останавливаем."""
    ss = st.session_state
    inc = ss.fsm.incident(pump_id)
    if inc is None or stage in inc.symptom_vectors:
        return
    try:
        sv = ss.backend.explain(pump_id, inc.stage_ts, stage)
    except Exception as e:
        inc.symptom_vectors[stage] = None
        inc.prescriptions[stage] = f"[ошибка анализа: {type(e).__name__}: {e}]"
        return
    inc.symptom_vectors[stage] = sv          # факт инцидента зафиксирован сейчас
    try:
        ss.shap_frozen[(inc.incident_id, stage)] = ss.backend.shap_figures(pump_id)
    except Exception:
        pass
    # ОДНА запись истории на это возникновение (incident_id, stage). Guard выше
    # (`stage in inc.symptom_vectors`) гарантирует ровно один заход → дублей нет.
    _log_event(inc, stage)
    _launch_generation(ss.backend, sv, pump_id, stage)
    ss.pinned[pump_id] = inc            # держим ссылку — переживёт закрытие в FSM
    ss._presc_ft[(inc.incident_id, stage)] = inc.fault_type


def _log_event(inc, stage):
    """Добавить запись в журнал истории один раз на (incident_id, stage).

    ts фиксируется СЕЙЧАС (на момент возникновения) и больше не меняется — даже
    если FSM позже сдвинет inc.stage_ts при повторной эскалации. Это и есть
    устойчивость: каждая авария/предупреждение в истории ровно один раз."""
    ss = st.session_state
    key = (inc.incident_id, stage)
    if any((e["incident_id"], e["stage"]) == key for e in ss.events):
        return
    ss.events.append({
        "incident_id": inc.incident_id,
        "pump_id": inc.pump_id,
        "stage": stage,
        "ts": inc.stage_ts,            # время возникновения, неизменно
        "status": "active",           # active -> acked / resolved (на месте)
        "status_ts": None,
    })


def _find_event(incident_id, stage):
    for e in st.session_state.events:
        if e["incident_id"] == incident_id and e["stage"] == stage:
            return e
    return None


def _refine_warning_type(pump_id, warming):
    """Тип отказа на ранней (слабой) сигнатуре часто неверен (склонен к «электрике»).
    По мере развития дефекта классификатор уточняет тип; если он устойчиво сменился —
    пересобираем диагностику и текст предписания предупреждения под верный тип."""
    ss = st.session_state
    if warming:
        return
    inc = ss.fsm.incident(pump_id)
    tick = ss.last_tick.get(pump_id)
    if (inc is None or tick is None or not getattr(tick, "ready", False)
            or pump_id in ss.tripped or inc.stage != 1
            or 1 not in inc.symptom_vectors):
        return
    live_ft = tick.fault_type
    built_for = ss._presc_ft.get((inc.incident_id, 1))
    if not live_ft or built_for is None or live_ft == built_for:
        ss._ft_cand.pop(pump_id, None)
        return
    cand, cnt = ss._ft_cand.get(pump_id, (None, 0))
    cnt = cnt + 1 if cand == live_ft else 1
    ss._ft_cand[pump_id] = (live_ft, cnt)
    if cnt < 6:                          # дебаунс: тип держится ~6 мин подряд
        return
    ss._ft_cand.pop(pump_id, None)
    try:
        sv = ss.backend.explain(pump_id, ss.sim_ts, 1)   # пересчёт на ТЕКУЩЕМ окне
    except Exception:
        return
    inc.fault_type = live_ft
    ss._presc_ft[(inc.incident_id, 1)] = live_ft
    inc.symptom_vectors[1] = sv
    inc.prescriptions.pop(1, None)       # текст перегенерируется в фоне
    inc.retrieval_traces.pop(1, None)
    try:
        ss.shap_frozen[(inc.incident_id, 1)] = ss.backend.shap_figures(pump_id)
    except Exception:
        pass
    _launch_generation(ss.backend, sv, pump_id, 1, force=True)


def advance_stream(n_rows):
    """Прогон n строк потока. На эскалации НЕ останавливаемся и НЕ прерываемся —
    ползунок и графики продолжают идти; предписание формируется в фоне."""
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
        # Пуск/простой (OFF=0, STARTUP=1) + мёртвая зона: обнуляем окно, пока
        # пусковой ток не вернулся под порог, затем ещё короткий хвост — так весь
        # переходник (любой длины) вырезан из скоринга, а не только первые 5 строк.
        try:
            stt = int(float(row["state"]))
        except (KeyError, TypeError, ValueError):
            stt = None
        if stt in (0, 1):
            backend.preproc.reset(pump_id)
            ss._deadzone[pump_id] = 5
        elif ss._deadzone.get(pump_id, 0) > 0:
            backend.preproc.reset(pump_id)
            try:
                _cur = float(row["current"])
            except (KeyError, TypeError, ValueError):
                _cur = None
            spiking = _cur is not None and _cur > THRESHOLDS["current"]["warning"]
            ss._deadzone[pump_id] = 5 if spiking else ss._deadzone[pump_id] - 1
        if not tick.ready:
            continue
        warming = (hasattr(backend.preproc, "rows_seen")
                   and backend.preproc.rows_seen(pump_id) < 15)
        trigger = fsm.update(pump_id, ts,
                             0 if warming else tick.severity,
                             suppressed=tick.suppressed or warming,
                             fault_type=tick.fault_type)
        if ss.validation is not None:
            ss.validation.add(row, tick.severity if tick.ready else -1,
                              tick.suppressed, trigger is not None)
        # сглаженная аномалия датчика: наивная модель выдала бы тревогу, окна её
        # погасили → засчитываем в «Подавлено» на дашборде оператора
        if trigger is None and (row.get("anomaly_vibration") or
                                row.get("anomaly_temperature") or
                                row.get("anomaly_current")):
            ss.anomaly_suppressed = ss.get("anomaly_suppressed", 0) + 1
        # дедуп аварии НЕЗАВИСИМО от ss.tripped: пока по агрегату жив неквитированный
        # аварийный инцидент (закреплён в ss.pinned, стадия 2) ИЛИ идёт останов/
        # восстановление — повтор аварии НЕ заводим (это дрожание типа / повторный
        # трип того же события, а не новый инцидент).
        _pin = ss.pinned.get(pump_id)
        _alarm_live = _pin is not None and 2 in _pin.symptom_vectors
        if trigger is not None and not (
                trigger.stage == 2 and (pump_id in ss.tripped
                                        or pump_id in ss.recovering
                                        or _alarm_live)):
            _on_escalation(pump_id, trigger.stage)
            if trigger.stage == 2:
                ss.tripped.add(pump_id)        # держит «Отказ» и ГЛУШИТ повторный трип
                if hasattr(player, "trip"):
                    player.trip(pump_id)        # теперь ОТЛОЖЕННЫЙ останов (см. плеер)
        # восстановление завершено, когда FSM реально вернулась в норму
        if pump_id in ss.recovering and fsm.state(pump_id) == 0:
            ss.recovering.discard(pump_id)
        _refine_warning_type(pump_id, warming)
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
    if pump_id in ss.tripped:            # авария: останов держит «Отказ» до квитирования
        return "failure"
    tick = ss.last_tick.get(pump_id)
    if tick is None:
        return "offline"
    raw_state = ss.history[pump_id][-1]["state"] if ss.history.get(pump_id) else 2
    if int(raw_state) in (0, 1):
        return "check"
    if not tick.ready:
        return "offline"
    if pump_id in ss.recovering:         # квитировано: не показываем стейл-«Отказ»
        return "check"
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


# Интерактивные графики оператора (Plotly: ось времени + ховер со значением)
def _plotly_axis(xvals, series, thr, label, unit, time_axis, yrange=None):
    n = len(series)
    if yrange is not None:
        lo, hi = yrange
    else:
        ymin, ymax = float(min(series)), float(max(series))
        pad = max(1e-6, (ymax - ymin) * 0.08)
        lo, hi = ymin - pad, ymax + pad
    fig = go.Figure()
    hov = (("%{x|%d.%m.%Y %H:%M}<br>%{y:.2f} " + unit + "<extra></extra>") if time_axis
           else ("%{y:.2f} " + unit + "<extra></extra>"))
    fig.add_trace(go.Scatter(x=xvals, y=list(series), mode="lines",
                             line=dict(color="#e3e7ea", width=1.6),
                             hovertemplate=hov))
    for key, col in (("warning", "#E0A800"), ("critical", "#C62828")):
        tv = thr[key]
        if lo <= tv <= hi:
            fig.add_hline(y=tv, line=dict(color=col, width=1, dash="dash"))
    xaxis = dict(gridcolor="#23282e")
    if time_axis:
        x_end = pd.Timestamp(xvals.iloc[-1] if hasattr(xvals, "iloc") else list(xvals)[-1])
        x_start = x_end - pd.Timedelta(minutes=WINDOW_MIN)
        xaxis.update(type="date", tickformat="%d.%m %H:%M",
                     range=[x_start, x_end],        # окно фикс-ширины; линия едет влево    # type: ignore
                     title=dict(text="Дата · время", font=dict(size=10)))  # type: ignore
    else:
        xaxis.update(range=[0, max(1, n - 1)],                      # type: ignore
                     title=dict(text="мин потока", font=dict(size=10))) # type: ignore
    fig.update_layout(template="plotly_dark", height=196,
                      margin=dict(l=46, r=10, t=6, b=30), showlegend=False,
                      paper_bgcolor="#0e1318", plot_bgcolor="#0e1318",
                      xaxis=xaxis,
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
    # ось X — в единицах текущего (модельного) времени, если timestamp парсится
    try:
        xvals = pd.to_datetime(df["timestamp"])
        time_axis = bool(xvals.notna().all())
    except Exception:
        time_axis = False
    if not time_axis:
        xvals = list(range(len(df)))
    cells = st.columns(2)
    for i, p in enumerate(params):
        with cells[i % 2]:
            st.markdown(value_badge(pump_id, p), unsafe_allow_html=True)
            if HAS_PLOTLY:
                st.plotly_chart(_plotly_axis(xvals, df[p].values, THRESHOLDS[p],
                                             PARAM_LABELS[p], THRESHOLDS[p]["unit"],
                                             time_axis, YRANGE.get(p)),
                                use_container_width=True,
                                config={"displayModeBar": False},
                                key=f"plt_{pump_id}_{p}")
            else:
                st.line_chart(df[[p]], height=196)


def render_active_prescription():
    """ОПЕРАТОРСКИЙ элемент (только внутри view_operator). Одна карточка НА КАЖДЫЙ
    насос с активным инцидентом; в пределах насоса показывается СТАРШАЯ стадия —
    авария вытесняет предупреждение. Карточки стопкой друг над другом в фикс-
    контейнере справа снизу: для разных насосов не перекрываются. В историю
    предписание попадает отдельно, поэтому вытеснённое предупреждение не теряется."""
    ss = st.session_state
    fsm = ss.fsm

    # по одному кандидату на насос: старшая обнаруженная стадия
    cards = []
    for pid in ss.history:
        inc = fsm.incident(pid)
        if inc is None and pid in ss.tripped:
            inc = ss.pinned.get(pid)          # авария: показываем после закрытия в FSM
        if not (inc and inc.symptom_vectors):
            continue
        stage = max(inc.symptom_vectors.keys())
        if (inc.incident_id, stage) in ss.acked:     # квитировано; ре-наг — в poll_renags
            continue
        cards.append((inc, stage))
    if not cards:
        return

    # порядок в стопке: сначала более тяжёлые, внутри тяжести — более свежие
    cards.sort(key=lambda e: (e[1], e[0].incident_id), reverse=True)

    # динамическая подсветка рамки карточек по тяжести (жёлтый/красный)
    rules = [f".st-key-toastcard_{inc.pump_id}_{stage}{{border-left-color:"
             f"{SEV_COLOR.get(stage, '#C62828')} !important;}}"
             for inc, stage in cards]
    st.markdown("<style>" + "".join(rules) + "</style>", unsafe_allow_html=True)

    with st.container(key="presctoast"):
        for inc, stage in cards:
            _render_toast_card(inc, stage)


def _acknowledge_incident(inc, stage, text):
    ss = st.session_state
    ss.fsm.acknowledge(inc.pump_id, ss.sim_ts or "")
    ss.pinned.pop(inc.pump_id, None)     # снимаем закрепление
    ack_ts = ss.sim_ts or inc.stage_ts
    ss.acked[(inc.incident_id, stage)] = ack_ts       # время (тост-гейтинг)
    ev = _find_event(inc.incident_id, stage)          # помечаем ТУ ЖЕ запись истории
    if ev is not None:
        ev["status"] = "acked"
        ev["status_ts"] = ack_ts
    if hasattr(ss.player, "acknowledge"):
        ss.player.acknowledge(inc.pump_id)        # предупреждение: 50%; авария: рестарт
    if stage == 2:
        ss.tripped.discard(inc.pump_id)      # снимаем «Отказ», генератор перезапустит
        ss.recovering.add(inc.pump_id)       # fsm.state=2 ещё ~4 тика стейл — гасим «Отказ»
        when = pd.Timestamp(ss.sim_ts or inc.stage_ts).strftime("%Y-%m-%d %H:%M")
        ss.toir_log.setdefault(inc.pump_id, []).insert(
            0, {"Дата": when,
                "Работа": "Проведены внеплановые работы по ликвидации аварии",
                "Тип": inc.fault_label, "Статус": "выполнено"})
        ss._ft_cand.pop(inc.pump_id, None)   # снимаем «временную память» уточнителя типа
            

def _render_toast_card(inc, stage):
    ss = st.session_state
    pid = inc.pump_id
    stage_label = SEVERITY_LABELS.get(stage, stage)
    ready_text = inc.prescriptions.get(stage)
    expanded = (pid in ss.toast_expanded) or (pid in ss.tripped)   # авария — всегда развёрнута

    with st.container(key=f"toastcard_{pid}_{stage}"):
        if not expanded:
            status = "✓ готово — развернуть" if ready_text else "⏳ формируется…"
            c = st.columns([5, 2])
            c[0].markdown(
                f"<div class='presc-head'>📋 {pid} · {stage_label} · {inc.fault_label}"
                f"<br><span style='color:#9fc6ff'>{status}</span></div>",
                unsafe_allow_html=True)
            if c[1].button("Развернуть", key=f"toast_exp_{pid}_{stage}"):
                ss.toast_expanded.add(pid)
                st.rerun(scope="fragment")
            return

        head = (f"АГРЕГАТ {pid} · {stage_label} · {inc.fault_label} · "
                f"ПРОШЛО {minutes_since(inc.stage_ts, ss.sim_ts or inc.stage_ts)} мин")
        st.markdown(f"<div class='presc-head'>{html.escape(head)}</div>",
                    unsafe_allow_html=True)
        if ready_text:
            st.markdown(sections_html(ready_text), unsafe_allow_html=True)
            c = st.columns([3, 2])
            if c[0].button("Квитировать", key=f"toast_ack_{pid}_{stage}", type="primary"):
                _acknowledge_incident(inc, stage, ready_text)
                ss.toast_expanded.discard(pid)
                st.rerun(scope="fragment")
            if c[1].button("Свернуть", key=f"toast_col_{pid}_{stage}"):
                ss.toast_expanded.discard(pid)
                st.rerun(scope="fragment")
        else:
            partial = gen_partial((pid, stage)) or ""
            st.markdown(sections_html(partial) if partial else
                        "<div class='presc-sb'>⏳ формируется предписание…</div>",
                        unsafe_allow_html=True)


# Сайдбар = левый «язычок»: история (с подсветкой) + сценарий-валидация
def _hist_color(stage, acked):
    if acked:
        return ACK_COLOR
    return SEV_COLOR.get(int(stage), "#5A5F66")


def render_sidebar(op_page, en_page, nav):
    ss = st.session_state
    with st.sidebar:
        is_op = (nav.url_path != "engineer")
        st.markdown("<div class='dash-switch-label'>ДАШБОРД</div>", unsafe_allow_html=True)
        st.markdown("<div class='role-wrap'>", unsafe_allow_html=True)
        if st.button("🖥  Оператор", key="role_op", use_container_width=True,
                     type="primary" if is_op else "secondary"):
            st.switch_page(op_page)
        if st.button("🔧  Инженер", key="role_en", use_container_width=True,
                     type="primary" if not is_op else "secondary"):
            st.switch_page(en_page)
        st.markdown("</div>", unsafe_allow_html=True)
        st.divider()
        st.markdown("### История предписаний")
        _render_history()        # авто-обновление каждые REFRESH

        st.divider()
        with st.expander("Сценарий и воспроизведение (валидация)",
                         expanded=ss.player is None):
            mode = st.radio("Источник данных",
                            ["Датасет (демо)", "Реальное время"],
                            horizontal=True, key="data_mode")

            if mode == "Датасет (демо)":
                dataset = st.text_input("Сырой датасет",
                                        "data/raw/industrial_pumps_dataset.csv")
                fault = st.selectbox("Тип отказа",
                                     ["overheat", "cavitation", "electrical"],
                                     format_func=lambda k: FAULT_LABELS[k])
                build_label = "Собрать сценарий"
            else:
                horizon = st.number_input("Горизонт прогона, мин",
                                          min_value=60, max_value=10000,
                                          value=480, step=60)
                st.caption("Парк из 5 насосов; отказы по типам генерируются "
                           "случайно. Темп/вероятности — в RealtimeConfig.")
                build_label = "Запустить поток"

            if st.button(build_label, use_container_width=True):
                try:
                    fc = ss.backend.preproc.feature_cols      # единый контракт признаков
                    if mode == "Датасет (демо)":
                        ss.backend.preproc = OnlinePreprocessor(fc)   # строгий + 60-прогрев
                        scen = extract_demo_scenario(dataset, fault)
                        ss.player = ScenarioPlayer(scen)
                        done = f"Готово: {len(scen)} мин, {scen['pump_id'].iloc[0]}"
                    else:
                        ss.backend.preproc = RealtimeProgressivePreprocessor(fc)  # с 0-й мин
                        ss.player = RealtimePlayer(LiveMultiPumpGenerator(),
                                                   horizon_minutes=int(horizon),
                                                   warmup_rows=0)     # холодный старт
                        done = (f"Поток запущен: {len(ss.player)} мин, "
                                f"{len(ss.player.gen.pump_ids)} насосов")
                    ss.fsm = PumpAlarmFSM()
                    ss.history, ss.last_tick = {}, {}
                    ss.selected_pump, ss.acked = None, {}
                    ss.playing = True                 # #8: поток стартует сразу
                    ss.events = []                    # чистая история нового прогона
                    ss.pinned, ss.tripped = {}, set()
                    ss.toast_expanded = set()
                    ss.recovering = set()
                    ss.validation = ValidationCollector.from_settings()
                    _store = get_gen_store()
                    with _store["lock"]:
                        _store["jobs"].clear()
                    ss.backend.preproc.reset(None)  # type: ignore
                    for row in ss.player.skip_warmup():    # realtime: пусто (warmup_rows=0)
                        pid = str(row["pump_id"])
                        push_history(pid, row)
                        ss.last_tick[pid] = ss.backend.process_tick(pid, row)
                    st.success(done)
                except Exception as e:
                    st.error(f"Не удалось запустить: {e}")
            
            if ss.validation is not None and st.button("Сохранить графики валидации"):
                log = ss.validation.to_frame()
                if len(log):
                    project_root = os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(
                                os.path.abspath(__file__))))
                    graphs_dir = os.path.join(project_root, 'artifacts', 'graphs')
                    paths = render_all(log, graphs_dir)
                    st.success(f"Метрики: {summarize(log)}")
                else:
                    st.warning("Лог пуст — сначала прогоните поток.")

        _render_progress()        # авто-обновление пройденных минут
        if ss.player is not None:
            b = st.columns(3)
            if b[0].button("▶"): ss.playing = True
            if b[1].button("⏸"): ss.playing = False
            if b[2].button("⏭"): advance_stream(1)
            ss.speed = st.slider("Минут за тик UI", 1, 20, ss.speed)


def render_header():
    st.markdown("<div class='app-title'>Платформа предиктивного обслуживания</div>"
                "<div class='app-sub'>Интерфейс · NAMUR NE 129</div>",
                unsafe_allow_html=True)


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
    k[2].metric("Подавлено",
                fsm.journal.count("suppressed") + ss.get("anomaly_suppressed", 0),
                help="Пуск/простой + сглаженные аномалии датчиков (наивная модель "
                     "выдала бы тревогу). Скрыты, но в архиве (ФЗ-116).")
    recent_tx = sum(1 for e in ss.fsm.journal.events
                    if e.kind == "transition" and ss.sim_ts
                    and minutes_since(getattr(e, "ts", ss.sim_ts), ss.sim_ts) <= 60)
    k[3].metric("Переходов/час", recent_tx,
                help="Скользящее окно 60 мин. Норматив ISA 18.2 — ≤6 тревог/час.")

    cols = st.columns(max(4, len(pumps)))
    for col, pid in zip(cols, pumps):
        s = NE107[status_key_for(pid)]
        inc = fsm.incident(pid)
        if inc is None and pid in ss.tripped:
            inc = ss.pinned.get(pid)
        prep = ss.backend.preproc
        raw_state = ss.history[pid][-1]["state"] if ss.history.get(pid) else 2
        tripped = pid in ss.tripped
        recovering = pid in ss.recovering             # квитировано — идёт восстановление
        warming = (not tripped) and hasattr(prep, "rows_seen") and prep.rows_seen(pid) < 15
        # тип отказа — только при активной аварии/предупреждении, НЕ на прогреве/восстановлении
        show_fault = tripped or (not warming and not recovering
                                 and inc is not None and fsm.state(pid) >= 1)
        extra = (FAULT_LABELS.get(inc.fault_type or "", "")
                 if (show_fault and inc and inc.fault_type) else "")
        if tripped:                                   # авария важнее прогрева
            color, icon, label = s["color"], s["icon"], s["label"]
        elif recovering:                              # стейл fsm.state=2 после ack — не «Отказ»
            color, icon, label = "#5A5F66", "◌", "Перезапуск · восстановление"
        elif warming:                                 # понятный текст вместо «прогрев»
            color, icon = "#5A5F66", "◌"
            label = ("Оборудование отключено" if int(raw_state) == 0
                     else "Пуск · накопление истории" if int(raw_state) == 1
                     else "Накопление истории")
        else:
            color, icon, label = s["color"], s["icon"], s["label"]
        col.markdown(f"<div class='tile' style='background:{color}'>"
                     f"<div class='tid'>{icon} {pid}</div>"
                     f"<div class='tst'>{label}</div>"
                     f"<div class='tx'>{extra}</div></div>", unsafe_allow_html=True)
        if col.button("Открыть", key=f"open_{pid}", use_container_width=True):
            ss.selected_pump = pid
            st.rerun(scope="fragment")

    if ss.selected_pump and ss.selected_pump in pumps:
        pid = ss.selected_pump
        head = st.columns([6, 1.4])
        head[0].markdown(f"#### Агрегат {pid}")
        if head[1].button("← к парку", use_container_width=True):
            ss.selected_pump = None
            st.rerun(scope="fragment")
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


def _engineer_param_snapshot(pid):
    """Текущие значения параметров агрегата (контекст для состояния «Норма»)."""
    h = st.session_state.history.get(pid, [])
    if not h:
        return
    cur = h[-1]
    rows = []
    for p in ("vibration", "temperature", "current", "pressure"):
        thr = THRESHOLDS[p]
        v = cur.get(p)
        rows.append({"Параметр": PARAM_LABELS[p],
                     "Текущее": round(float(v), 2) if v is not None else "—",
                     "Ед.": thr["unit"],
                     "Пред.": thr["warning"], "Авар.": thr["critical"]})
    st.markdown("**Текущие параметры**")
    st.table(pd.DataFrame(rows))


def view_engineer():
    ss = st.session_state
    fsm = ss.fsm
    st.subheader("Инженерная диагностика")

    pumps = sorted(ss.history.keys())
    if not pumps:
        st.info("Поток данных не запущен. Список оборудования и диагностические "
                "данные появляются по мере поступления данных.")
        return

    def _inc(pid):
        return fsm.incident(pid) or ss.pinned.get(pid)

    def _sev(pid):
            if pid in ss.tripped:
                return 2
            if pid in ss.recovering:              # квитировано — стейл fsm.state=2 гасим
                return 0
            prep = getattr(ss.backend, "preproc", None)
            if prep is not None and hasattr(prep, "rows_seen") and prep.rows_seen(pid) < 15:
                return 0                          # прогрев после рестарта — это не авария
            return fsm.state(pid)

    def _recency(pid):
        inc = _inc(pid)
        return inc.incident_id if inc else -1

    order = sorted(pumps, key=lambda p: (_sev(p), _recency(p)), reverse=True)

    def _label(pid):
        sev = _sev(pid)
        inc = _inc(pid)
        ftype = inc.fault_label if (inc and sev >= 1) else "—"
        return f"{pid} · {SEVERITY_LABELS.get(sev, sev)} · {ftype}"

    pid = st.selectbox("Оборудование (отсортировано по тяжести состояния)",
                       order, format_func=_label, key="eng_pump")
    sev = _sev(pid)
    inc = _inc(pid)

    if sev == 0 or inc is None:
        c = st.columns(3)
        c[0].metric("Состояние", SEVERITY_LABELS.get(sev, sev))
        c[1].metric("Тип отказа", "—")
        c[2].metric("Активный инцидент", "нет")
        st.success("Параметры агрегата в пределах нормы. Диагностический разбор "
                   "(SHAP, симптомы, трассировка RAG) формируется при переходе "
                   "в «Предупреждение» или «Авария».")
        _engineer_param_snapshot(pid)
        return

    stage = inc.stage
    # Уверенность типа и признаки на активном предупреждении считаем ЖИВО по текущему
    # окну — они растут по мере развития дефекта. Зафиксированный при эскалации вектор
    # замораживаем ТОЛЬКО при отказе (агрегат остановлен, новых данных нет) — как SHAP.
    sv = inc.symptom_vectors.get(stage) or \
        next((v for v in inc.symptom_vectors.values() if v is not None), None)
    if pid not in ss.tripped:
        live_sv = ss.backend.explain(pid, ss.sim_ts or inc.stage_ts, stage)
        if live_sv is not None:
            sv = live_sv

    c = st.columns(3)
    c[0].metric("Состояние", SEVERITY_LABELS.get(stage, stage))
    c[1].metric("Тип отказа", inc.fault_label)
    c[2].metric("Уверенность типа", f"{getattr(sv, 'fault_confidence', 0):.0f}%")
    if stage == 1:
        st.caption("Стадия «Предупреждение»: абсолютные значения параметров могут "
                   "быть в норме — сигнал в статистической сигнатуре, не в пороге.")
    if 1 in inc.symptom_vectors and 2 in inc.symptom_vectors:
        st.caption("Агрегат прошёл «Предупреждение» → «Авария»; показан текущий "
                   "(старший) этап. Ранние предписания — в истории (левый сайдбар).")

    tabs = st.tabs(["Диагностика (SHAP)", "Симптомы", "Трассировка RAG",
                    "ТОиР: план и история"])
    with tabs[0]:
        use_frozen = inc.pump_id in ss.tripped
        if use_frozen:
            f_sev, f_fault = ss.shap_frozen.get((inc.incident_id, stage)) or (None, None)
        else:
            # живой SHAP с троттлингом ~5 c: иначе медиафайлы плодятся каждые 2 c,
            # старые вытесняются → MediaFileStorageError и картинка «замирает».
            ck = f"_shaplive_{inc.pump_id}"
            cached = ss.get(ck)
            now = time.time()
            if cached is None or now - cached[0] >= 5:
                cached = (now, ss.backend.shap_figures(inc.pump_id))
                ss[ck] = cached
            f_sev, f_fault = cached[1]
        if not (f_sev or f_fault):
            st.caption("SHAP-графики доступны в боевом режиме ядра.")
        gc = st.columns(2)
        if f_fault:
            gc[0].image(f_fault, use_container_width=True,
                        caption="Почему классификатор выбрал этот тип отказа")
        if f_sev:
            gc[1].image(f_sev, use_container_width=True,
                        caption="Вклад признаков по классу «Авария» "
                                "(удалённость от аварии)")
        st.caption("SHAP зафиксирован на момент аварии — не пересчитывается после "
                   "останова." if use_frozen else
                   "SHAP по текущему окну — обновляется каждые ~5 c, пока агрегат в работе.")
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
        trace = inc.retrieval_traces.get(stage) or \
            (ss.backend.retrieval_trace(sv, STAGE_BY_SEVERITY.get(stage, "critical"))
             if sv is not None else [])
        if trace:
            st.dataframe(pd.DataFrame(trace), use_container_width=True, hide_index=True)
            st.caption("Источники по разделам: диагноз — мануал/ГОСТ/вибродиагностика; "
                       "предписание и ТОиР — регламент; плановый ремонт — график ППР.")
        else:
            st.caption("Трассировка формируется вместе с предписанием.")
    with tabs[3]:
        st.markdown("**Плановый ремонт (из графика ППР)**")
        text = inc.prescriptions.get(stage, "")
        m = re.search(r"ПЛАНОВЫЙ РЕМОНТ:\s*(.+)", text, re.S)
        st.write(m.group(1).strip()[:400] if m
                 else ("⏳ формируется…" if stage not in inc.prescriptions
                       else "— нет данных графика —"))
        st.markdown("**История работ по агрегату** _(подключается к системе ТОиР "
                    "предприятия; здесь — демонстрационные данные)_")
        log = ss.toir_log.get(inc.pump_id, [])
        demo = [
            {"Дата": "2026-02-11", "Работа": "ТО-1: замена смазки картера",
             "Тип": "—", "Статус": "выполнено"},
            {"Дата": "2025-11-03", "Работа": "ТО-2: лазерная центровка валов",
             "Тип": "—", "Статус": "выполнено"},
        ]
        st.dataframe(pd.DataFrame(log + demo), use_container_width=True, hide_index=True)


def _engine_tick():
    ss = st.session_state
    now = time.monotonic()
    if ss.playing and ss.player is not None and not ss.player.finished:
        if now - ss.get("_last_advance", 0.0) >= REFRESH * 0.8:   # темп задаёт таймер,
            advance_stream(ss.speed)                              # а не клики
            ss._last_advance = now
    poll_generations()
    poll_renags()


@st.fragment(run_every=REFRESH)
def _operator_live():
    _engine_tick()
    view_operator()


@st.fragment(run_every=REFRESH)
def _render_progress():
    ss = st.session_state
    if ss.player is not None:
        st.progress(ss.player.progress,
                    text=f"Поток: {ss.player.pos}/{len(ss.player)} мин")


@st.fragment(run_every=REFRESH)
def _engineer_live():
    _engine_tick()
    view_engineer()


@st.fragment(run_every=REFRESH)
def _render_history():
    """История из устойчивого журнала ss.events.

    Каждое возникновение (incident_id, stage) — РОВНО одна запись (append-once
    в _log_event). Дубли невозможны по построению, склейка по времени не нужна.
    Статус записи (active/acked/resolved/escalated) выводится здесь; acked и
    resolved фиксируются с временем и больше не меняются."""
    ss = st.session_state
    events = ss.events
    GREY = "#5A5F66"

    # incident_id -> Incident (для текста предписания); и множество ЖИВЫХ инцидентов
    by_id = {}
    for inc in ss.fsm.all_incidents():
        by_id[inc.incident_id] = inc
    for inc in ss.pinned.values():
        by_id.setdefault(inc.incident_id, inc)
    live_ids = {inc.incident_id for inc in ss.pinned.values()}
    for pid in ss.history:
        cur = ss.fsm.incident(pid)
        if cur is not None:
            live_ids.add(cur.incident_id)
    alarm_ids = {e["incident_id"] for e in events if e["stage"] == 2}

    def _kind(e):
        """Текущий вид записи. acked/resolved — терминальные (фиксируются на месте)."""
        if e["status"] == "acked":
            return "acked"
        if e["status"] == "resolved":
            return "resolved"
        # предупреждение, доросшее до аварии того же инцидента — не «норма», а эскалация
        if e["stage"] == 1 and e["incident_id"] in alarm_ids:
            return "escalated"
        if e["incident_id"] not in live_ids:          # инцидент закрыт → возврат в норму
            e["status"] = "resolved"
            e["status_ts"] = ss.sim_ts
            return "resolved"
        return "active"

    rows = sorted(events, key=lambda e: (str(e["ts"]), e["incident_id"], e["stage"]),
                  reverse=True)

    rules = []
    for e in rows:
        kind = _kind(e)
        col = (ACK_COLOR if kind == "acked"
               else GREY if kind == "resolved"
               else SEV_COLOR.get(int(e["stage"]), GREY))
        cls = f"hist_{e['incident_id']}_{e['stage']}"
        rules.append(
            f".st-key-{cls}{{border-left:4px solid {col};border-radius:8px;"
            f"background:{col}14;padding:1px 8px 1px 9px;margin-bottom:7px;}}"
            f".st-key-{cls} [data-testid='stExpander']{{border:none;}}")
    if rules:
        st.markdown("<style>" + "".join(rules) + "</style>", unsafe_allow_html=True)
    if not rows:
        st.caption("Предписаний пока нет.")

    for e in rows:
        kind = _kind(e)
        stage = e["stage"]
        inc = by_id.get(e["incident_id"])
        text = inc.prescriptions.get(stage) if inc is not None else None

        def _fmt(t):
            try: return pd.Timestamp(t).strftime("%d.%m %H:%M")
            except Exception: return str(t)

        if kind == "acked":
            badge = f" · ✓ квитировано {_fmt(e['status_ts'])}"
        elif kind == "resolved":
            badge = f" · ↩ возвращено в норму {_fmt(e['status_ts'])}"
        elif kind == "escalated":
            badge = " · ↑ переросло в аварию"
        else:
            badge = ""
        when = _fmt(e["ts"])
        with st.container(key=f"hist_{e['incident_id']}_{stage}"):
            with st.expander(f"{when} · {e['pump_id']} · "
                             f"{SEVERITY_LABELS.get(stage, stage)}{badge}"):
                if text:
                    st.markdown(sections_html(text), unsafe_allow_html=True)
                else:
                    st.caption("⏳ Формируется предписание…")


def page_operator():
    _operator_live()


def page_engineer():
    _engineer_live()


def main():
    ss = st.session_state
    init_state()
    get_backend()
    op = st.Page(page_operator, title="Оператор", url_path="operator", default=True)
    en = st.Page(page_engineer, title="Инженер", url_path="engineer")
    nav = st.navigation([op, en], position="hidden")
    render_sidebar(op, en, nav)
    nav.run()
    # Поток и переотрисовка живут во фрагментах (run_every). Глобального
    # time.sleep/st.rerun НЕТ — иначе он перебивает фрагменты, съедает клики
    # (тосты «развернуть/квитировать»), кидает экран наверх и двоит тики.


if __name__ == "__main__":
    main()