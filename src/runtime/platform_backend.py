"""
Единственная точка интеграции интерфейса с аналитическим ядром.

Дашборды (app.py) НЕ импортируют ml_pipeline / xai_module / rag_database /
ai_agent напрямую — только этот адаптер. Поэтому:
  * замена источника данных (демо-CSV -> реальная SCADA) не трогает UI;
  * изменение сигнатур внутренних модулей локализовано в одном файле;
  * UI тестируем без поднятых Ollama/ChromaDB (см. ProtoBackend ниже).

РАСПОЛОЖЕНИЕ: src/app/. Project root = три уровня вверх. На sys.path
кладём И корень (для пакетов config / validation / src.*), И src/
(для «голых» импортов data_preprocessor, как внутри самого ml_pipeline).
"""

import os
import sys
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import joblib
import pandas as pd
import tempfile

# разрешение путей (файл лежит в src/app/) 
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import WINDOW_SIZES
from src.data.data_preprocessor import DataPreprocessor
from src.ml.severity_classifier_pipeline import AlarmManager
from src.xai.xai_module import XAIExplainer
from src.rag.rag_database import KnowledgeBaseManager, resolve_stage
from src.agent.ai_agent import DiagnosticAgent
from src.runtime.online_preprocessor import OnlinePreprocessor

_preprocessor = DataPreprocessor(window_sizes=WINDOW_SIZES)
FEATURE_COLS = _preprocessor.FEATURE_COLS

# Пороги для линий на трендах. Источник истины — config/settings;
# значения ниже — fallback, совпадающий с нормативной базой работы.
THRESHOLDS = {
    "vibration":   {"warning": 3.0,  "critical": 8.0,  "unit": "мм/с",
                    "source": "ГОСТ 32601-2013 / паспорт МНХВ"},
    "temperature": {"warning": 82.0, "critical": 93.0, "unit": "°C",
                    "source": "регламент предприятия / API 610"},
    "current":     {"warning": 65.0, "critical": 80.0, "unit": "А",
                    "source": "паспорт МНХВ"},
    "pressure":    {"warning": 1.1,  "critical": 0.8,  "unit": "МПа",
                    "source": "паспорт МНХВ", "inverted": True},
}

PARAM_LABELS = {
    "vibration": "Вибрация", "temperature": "Температура",
    "current": "Ток", "pressure": "Давление",
}

# Человекочитаемые имена документов для трассировки извлечения (инженер).
_DOC_DISPLAY = {
    "tm_regulation": "Регламент ТО (SOP)",
    "tm_schedule": "График ППР",
    "gost_extract": "ГОСТ 32601-2013",
    "mnhv_extract": "Руководство МНХВ",
    "diagnostics_extended": "Вибродиагностика",
}


def _doc_display(source: str) -> str:
    stem = os.path.splitext(os.path.basename(str(source)))[0]
    return _DOC_DISPLAY.get(stem, stem or "неизвестный источник")


@dataclass
class TickResult:
    """Результат обработки одного тика телеметрии."""

    ready: bool                      # False во время прогрева окон
    severity: int = 0                # 0/1/2 (после контекстной фильтрации)
    raw_severity: int = 0            # 0/1/2 (предсказание модели до фильтра)
    suppressed: bool = False         # подавлено state-based фильтрацией
    severity_proba: Optional[list] = None
    fault_type: Optional[str] = None
    fault_proba: Optional[dict] = None


class PlatformBackend:
    """Боевой адаптер: реальные модели, RAG и LLM-агент."""

    def __init__(self):
        self.feature_cols = list(FEATURE_COLS)
        self.preproc = OnlinePreprocessor(self.feature_cols)

        _models_dir = os.path.join(_PROJECT_ROOT, "models")
        _sev_path = os.path.join(_models_dir, 'severity', "severity_xgboost_model.joblib")
        _fault_path = os.path.join(_models_dir, 'fault_type', "fault_xgboost_model.joblib")
        _chroma_dir = os.path.join(_PROJECT_ROOT, 'artifacts', "chroma_db")
        _kb_dir = os.path.join(_PROJECT_ROOT, "knowledge_base")

        # severity-модель грузим отдельно (нужна AlarmManager и для proba);
        # XAIExplainer внутри сам грузит обе модели — fault_model берём у него.
        self.severity_model = joblib.load(_sev_path)
        self.alarm_manager = AlarmManager(self.severity_model)
        self.xai = XAIExplainer(_sev_path, _fault_path)
        self.fault_model = self.xai.fault_model
        self.rag = KnowledgeBaseManager(data_dir=_kb_dir, chroma_dir=_chroma_dir)
        self.agent = DiagnosticAgent()   # model_name=DEFAULT_MODEL по умолчанию

        self._last_features: dict = {}     # pump_id -> последняя строка признаков
        self._last_trace: List[dict] = []  # трасса последнего RAG-извлечения

    def process_tick(self, pump_id: str, raw_row: pd.Series) -> TickResult:
        raw = raw_row.to_dict() if hasattr(raw_row, "to_dict") else dict(raw_row)
        feats = self.preproc.push(pump_id, raw) # type: ignore
        if feats is None:
            return TickResult(ready=False)
        self._last_features[pump_id] = feats

        raw_state = int(raw_row.get("state", 2))

        # контекстная фильтрация (Off/Startup -> 0); raw_pred — до фильтра
        severity = int(self.alarm_manager.predict_with_context(feats, raw_state))
        raw_pred = int(self.severity_model.predict(feats)[0])
        proba = self.severity_model.predict_proba(feats)[0].tolist()
        suppressed = (severity == 0 and raw_pred > 0)

        fault_type, fault_proba = None, None
        if severity > 0 and self.fault_model is not None:
            # классификатор типа активируется только на нештатных режимах
            fp = self.fault_model.predict_proba(feats)[0]
            classes = list(getattr(self.fault_model, "classes_", range(len(fp))))
            idx_to_name = {0: "overheat", 1: "cavitation", 2: "electrical"}
            fault_proba = {idx_to_name.get(int(c), str(c)): float(p)
                           for c, p in zip(classes, fp)}
            fault_type = max(fault_proba, key=fault_proba.get)  # type: ignore

        return TickResult(ready=True, severity=severity, raw_severity=raw_pred,
                          suppressed=suppressed, severity_proba=proba,
                          fault_type=fault_type, fault_proba=fault_proba)

    def explain(self, pump_id: str, ts: str, severity: int):
        """SymptomVector для текущего инцидента (XAI обеих моделей).

        severity не передаётся в XAI — модель тяжести сама вычисляет класс
        по predict_proba; аргумент оставлен в сигнатуре ради единого
        вызова из UI.
        """

        feats = self._last_features[pump_id]
        return self.xai.explain_prediction(feats, pump_id=pump_id, timestamp=ts)

    def _rag_context(self, sv, stage: str):
        """Собирает 4 канала RAG-контекста по вектору симптомов.

        Паттерн идентичен ai_agent.__main__: справочный контекст,
        график ППР, работы ТОиР, действия оператора нужной стадии.
        Здесь же формируется трасса извлечения для инженерной вкладки.
        """

        symptom_dict = {
            "critical_probability": sv.critical_probability,
            "inferred_fault": sv.inferred_fault,
            "top_symptoms": [{"sensor": s.sensor, "value": s.value,
                              "shap_weight": s.shap_weight}
                             for s in sv.top_symptoms],
        }
        op_stage = stage if stage in ("warning", "critical") else "critical"
        rag_results = self.rag.search_by_symptoms(symptom_dict, k=4)
        schedule_results = self.rag.search_maintenance_schedule(sv.pump_id, k=2)
        repair_results = self.rag.search_repair_works(sv.inferred_fault, k=2)
        operator_results = (
            self.rag.search_operator_actions(sv.inferred_fault, stage=op_stage, k=2)
            if stage != "unknown" else [])

        self._last_trace = self._build_trace(
            reference=rag_results, operator=operator_results,
            repair=repair_results, schedule=schedule_results)
        return rag_results, schedule_results, repair_results, operator_results

    @staticmethod
    def _build_trace(reference, operator, repair, schedule) -> List[dict]:
        """Строит трассу извлечения из (Document, distance)-пар всех каналов.

        Не зависит от внутренностей агента: backend сам ретривит контекст,
        поэтому сам и знает, какой фрагмент какой раздел питал.
        """

        rows: List[dict] = []
        channels = (
            ("ДИАГНОЗ (справка)", reference),
            ("ПРЕДПИСАНИЕ (оператор)", operator),
            ("РЕКОМЕНДАЦИИ ТОиР", repair),
            ("ПЛАНОВЫЙ РЕМОНТ", schedule),
        )
        for block, results in channels:
            for item in (results or []):
                doc, dist = (item if isinstance(item, tuple) else (item, None))
                meta = getattr(doc, "metadata", {}) or {}
                rows.append({
                    "block": block,
                    "doc": _doc_display(meta.get("source", "")),
                    "distance": round(float(dist), 3) if dist is not None else None,
                    "fault_type": meta.get("fault_type", "—"),
                    "sop_part": meta.get("sop_part", "—"),
                    "stage": meta.get("stage", "—"),
                })
        return rows

    def prescription_stream(self, symptom_vector, stage: str) -> Iterator[str]:
        """Потоковая генерация предписания (Ollama stream=True).

        stage in {'warning','critical'} — стадийная выборка действий оператора;
        агент дополнительно резолвит стадию через resolve_stage(sv).
        """

        rag, schedule, repair, operator = self._rag_context(symptom_vector, stage)
        return self.agent.generate_prescription_stream(
            symptom_vector, rag, schedule_results=schedule,
            repair_results=repair, operator_results=operator)

    def retrieval_trace(self, symptom_vector, stage: str) -> List[dict]:
        """Трасса последнего извлечения (заполняется в prescription_stream).

        Если стрим ещё не запускался — ретривим контекст здесь, чтобы
        инженерная вкладка не пустовала.
        """

        if not self._last_trace:
            self._rag_context(symptom_vector, stage)
        return self._last_trace

    def shap_figures(self, pump_id: str) -> Tuple[Optional[bytes], Optional[bytes]]:
        """PNG-байты waterfall-графиков (модель тяжести, модель типа) В ПАМЯТИ.

        Рисуем во временный файл, читаем байты, файл сразу удаляем — на диск
        ничего не оседает (artifacts/graphs не засоряется). UI показывает байты
        через st.image. None — если график не построился.
        """
        feats = self._last_features.get(pump_id)
        if feats is None:
            return None, None
        try:
            from src.visualisation.xai_visualisation import (plot_severity_waterfall,
                                                             plot_fault_waterfall)
        except Exception:
            return None, None

        def _to_bytes(plot_fn):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                plot_fn(self.xai, feats, pump_id=pump_id, save_path=tmp_path)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    with open(tmp_path, "rb") as fh:
                        return fh.read()
                return None
            except Exception:
                return None
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        return _to_bytes(plot_severity_waterfall), _to_bytes(plot_fault_waterfall)


class ProtoBackend(PlatformBackend):
    """Облегчённый backend для отладки UI без Ollama/ChromaDB/моделей.

    Тяжесть и тип берутся из меток датасета (state, fault_type),
    предписание имитируется. Позволяет верстать и прогонять демо-сценарий
    на любой машине. НЕ для защиты — только для разработки/вёрстки.
    """

    def __init__(self):  # намеренно не зовём super().__init__
        from src.runtime.online_preprocessor import (OnlinePreprocessor,
                                                     PARAMS, WINDOWS)
        cols = []
        for p in PARAMS:
            for w in WINDOWS:
                cols += [f"{p}_mean_{w}", f"{p}_std_{w}", f"{p}_max_{w}"]
            cols.append(f"{p}_diff_30")
        self.feature_cols = cols
        self.preproc = OnlinePreprocessor(cols)
        self._last_features = {}
        self._last_label = {}
        self._last_trace = []

    def process_tick(self, pump_id: str, raw_row: pd.Series) -> TickResult:
        feats = self.preproc.push(pump_id, raw_row.to_dict())   # type: ignore
        if feats is None:
            return TickResult(ready=False)
        self._last_features[pump_id] = feats

        raw_state = int(raw_row.get("state", 2))
        label = {2: 0, 3: 1, 4: 2}.get(raw_state, 0)
        suppressed = raw_state in (0, 1) and label > 0
        ft = raw_row.get("fault_type")
        ft = None if (not isinstance(ft, str) or ft == "none") else ft
        self._last_label[pump_id] = (label, ft)
        proba = [[0.97, 0.025, 0.005], [0.10, 0.85, 0.05], [0.02, 0.08, 0.90]][label]
        fp = None
        if label > 0 and ft:
            fp = {k: (0.93 if k == ft else 0.035)
                  for k in ("overheat", "cavitation", "electrical")}
        return TickResult(ready=True, severity=0 if suppressed else label,
                          raw_severity=label, suppressed=suppressed,
                          severity_proba=proba, fault_type=ft, fault_proba=fp)

    def explain(self, pump_id, ts, severity):
        # имитация SymptomVector ровно с теми полями, что у боевого (xai_module):
        # UI читает probabilities[1] (drill-down предупреждения) — без него падает.
        from types import SimpleNamespace
        label, ft = self._last_label.get(pump_id, (0, None))
        proba = [[0.97, 0.025, 0.005],
                 [0.10, 0.85, 0.05],
                 [0.02, 0.08, 0.90]][label]
        sym = SimpleNamespace(
            pump_id=pump_id, timestamp=ts, predicted_class=label,
            probabilities=proba,                       # [P(Норма),P(Пред),P(Авария)]
            critical_probability=proba[2] * 100,
            shap_base_value=0.0,
            inferred_fault=ft or "unknown",
            true_fault=ft or "unknown",
            top_symptoms=[SimpleNamespace(feature="temperature_mean_60",
                                          sensor="temperature", value=94.1,
                                          shap_weight=2.31, window="mean_60")],
            fault_top_symptoms=[SimpleNamespace(feature="current_mean_60",
                                                sensor="current", value=77.0,
                                                shap_weight=1.9, window="mean_60")],
            fault_confidence=93.0 if ft else 0.0,
            fault_probabilities=[0.93, 0.04, 0.03] if ft else [],
        )
        return sym

    def prescription_stream(self, sv, stage):
        import time
        self._last_trace = self.retrieval_trace(sv, stage)
        text = (
            "ДИАГНОЗ И ОБОСНОВАНИЕ: [имитация — ProtoBackend]\n\n"
            "ПРЕДПИСАНИЕ ОПЕРАТОРУ:\n1. Шаг 1…\n2. Шаг 2…\n\n"
            "РЕКОМЕНДАЦИИ ТОиР: …\n\nПЛАНОВЫЙ РЕМОНТ: …\n\n"
            "ИСТОЧНИКИ: Регламент ТО; ГОСТ 32601-2013."
        )
        for tok in text.split(" "):
            time.sleep(0.02)
            yield tok + " "

    def retrieval_trace(self, sv, stage):
        return [{"block": "ПРЕДПИСАНИЕ (оператор)",
                 "doc": "Регламент ТО (имитация)", "distance": 0.24,
                 "fault_type": getattr(sv, "inferred_fault", "—"),
                 "sop_part": "operator", "stage": stage}]

    def shap_figures(self, pump_id):
        return None, None