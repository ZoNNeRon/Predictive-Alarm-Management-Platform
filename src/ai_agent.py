"""
Модуль интеллектуального LLM-агента (AI Agent)
===============================================
Назначение: интеграция вектора симптомов (XAI) и контекста нормативной базы (RAG)
для генерации прескриптивного предписания оператору. Переводит систему от
Predictive Maintenance (прогноз) к Prescriptive Maintenance (руководство к действию).

Ответственность модуля (намеренно ограничена):
  - Принимает SymptomVector (из xai_module) и контекст (из rag_database)
  - Строит системный и пользовательский промпт
  - Вызывает локальную LLM через Ollama
  - Возвращает структурированный AgentResponse

Модуль НЕ вычисляет SHAP и НЕ ищет в векторной базе — это делают xai_module и
rag_database. Здесь только оркестрация промпта и вызов LLM (принцип SRP).

Выбор модели:
  По умолчанию qwen3.5:9b — лучшая поддержка русского языка и
  структурированного вывода среди локальных моделей (Qwen Team).
  Альтернативы переключаются одним параметром: phi3:14b, phi4.
  Это позволяет провести сравнение моделей для диплома.
"""

import sys
import os
import ollama
import time
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config.settings import LLM_MODELS, DEFAULT_LLM_MODEL

# Импорт типов из XAI-модуля — единый контракт данных
from xai_module import SymptomVector, SymptomContribution

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    # Fallback на старый импорт, если установлена прежняя версия langchain
    from langchain_community.llms import Ollama as OllamaLLM


# Структура ответа агента 

@dataclass
class AgentResponse:
    pump_id:            str
    raw_text:           str
    model_name:         str
    used_context:       bool
    sources:            List[str] = field(default_factory=list)
    latency_sec:        Optional[float] = None   # настенное время (вкл. загрузку)
    gen_time_sec:       Optional[float] = None    # чистая генерация (eval_duration)
    eval_count:         Optional[int]   = None    # токенов на выводе
    prompt_eval_count:  Optional[int]   = None    # токенов на входе
    tokens_per_sec:     Optional[float] = None    # скорость генерации
    format_ok:          Optional[bool]  = None    # соблюдён ли формат (4 раздела)
    error:              Optional[str]   = None    # текст ошибки, если была


# Системный промпт — загружается из config/prompts/diagnostic_agent.md

def _load_system_prompt() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, 'config', 'prompts', 'diagnostic_agent.md')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл с промптом не найден: {path}.")
    with open(path, encoding='utf-8') as f:
        return f.read().strip()

SYSTEM_PROMPT = _load_system_prompt()


# Основной класс агента 

class DiagnosticAgent:
    """
    LLM-агент для генерации прескриптивных предписаний.

    Пример использования (из app.py):
        agent = DiagnosticAgent(model_name="qwen3.5:9b")
        response = agent.generate_prescription(symptom_vector, rag_results)
        print(response.raw_text)
    """

    # Модель по умолчанию: лучший баланс русский язык + структурированный вывод
    DEFAULT_MODEL = DEFAULT_LLM_MODEL

    def __init__(self, model_name: str = None,  # type: ignore
                 temperature: float = 0.1,
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Args:
            model_name:     Имя модели Ollama. None → DEFAULT_MODEL.
                            Варианты: 'qwen3.5:9b', 'phi3:14b', 'phi4', 'qwen3.5:9b'.
            temperature:    Низкая (0.1) — для детерминированных, точных предписаний.
                            Высокая температура недопустима в промышленной диагностике.
            ollama_base_url: Адрес локального сервера Ollama.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.temperature = temperature

        # num_predict ограничивает длину вывода — предписание должно быть компактным.
        # Для Qwen 3.5 (гибридная reasoning-модель) reasoning отключается через
        # options или префикс /no_think в промпте — иначе модель "думает вслух",
        # ломая формат и тратя минуты на генерацию.
        # Нативный клиент Ollama: позволяет явно управлять reasoning через think=False.
        # Это надёжнее текстовой директивы /no_think, которая ведёт себя
        # по-разному в разных сборках модели.
        self.client = ollama.Client(host=ollama_base_url)
        self.options = {
            'temperature': temperature,
            'top_p': 0.9,
            'num_predict': 800,
            'num_ctx': 4096,
        }

    # Форматирование входных данных 

    def _format_symptoms(self, sv: SymptomVector) -> str:
        """
        Преобразует SymptomVector в человекочитаемый блок СИМПТОМЫ.
        Транслирует точные численные показания и пороги ГОСТ — это критично:
        агент не должен "размывать" вклад математической модели (XGBoost/SHAP).
        """
        class_names = ['НОРМА', 'ПРЕДУПРЕЖДЕНИЕ (Warning)', 'АВАРИЯ (Critical)']
        lines = [
            f"Агрегат: {sv.pump_id}",
            f"Время: {sv.timestamp or 'не указано'}",
            f"Прогноз модели XGBoost: {class_names[sv.predicted_class]}",
            f"Вероятность аварии: {sv.critical_probability}%",
            "",
            "Ключевые признаки (по убыванию важности SHAP):",
        ]

        sensor_ru = {'vibration': 'Вибрация', 'temperature': 'Температура',
                     'current': 'Ток', 'pressure': 'Давление'}

        for i, s in enumerate(sv.top_symptoms, 1):
            sensor_name = sensor_ru.get(s.sensor, s.sensor)
            direction = "повышает риск аварии" if s.shap_weight > 0 else "снижает риск аварии"

            # Сравнение с порогом ГОСТ, если он применим
            threshold_note = ""
            if s.critical_threshold is not None:
                if s.value >= s.critical_threshold:
                    threshold_note = (f" — ПРЕВЫШЕН критический порог "
                                      f"{s.critical_threshold} (норматив)")
                elif s.warning_threshold is not None and s.value >= s.warning_threshold:
                    threshold_note = (f" — превышен порог предупреждения "
                                      f"{s.warning_threshold} (норматив)")

            unit = {'vibration': 'мм/с', 'temperature': '°C',
                    'current': 'А', 'pressure': 'МПа'}.get(s.sensor, '')

            lines.append(
                f"  {i}. {sensor_name} ({s.feature}) = {s.value} {unit}"
                f"{threshold_note}. Вклад SHAP: {s.shap_weight:+.3f} ({direction})."
            )

        return "\n".join(lines)

    def _format_context(self, rag_results: list) -> tuple:
        """
        Преобразует результаты RAG в блок КОНТЕКСТ.

        Args:
            rag_results: список (Document, distance) из rag_database.search_by_symptoms()

        Returns:
            (context_text, sources) — текст контекста и список источников для атрибуции.
        """
        if not rag_results:
            return "", []

        blocks = []
        sources = []
        for i, item in enumerate(rag_results, 1):
            # Поддержка как (doc, score), так и просто doc
            doc = item[0] if isinstance(item, tuple) else item
            text = doc.page_content.replace('passage: ', '', 1)
            source = doc.metadata.get('source', 'неизвестный источник')
            doc_type = doc.metadata.get('doc_type', '')

            blocks.append(f"[Фрагмент {i}] (источник: {source})\n{text}")
            if source not in sources:
                sources.append(source)

        return "\n\n".join(blocks), sources

    # Главный метод 

    def generate_prescription(self, symptom_vector: SymptomVector,
                              rag_results: list) -> AgentResponse:
        """
        Генерирует прескриптивное предписание на основе симптомов и контекста.

        Args:
            symptom_vector: SymptomVector из xai_module.XAIExplainer.explain()
            rag_results:    результаты rag_database.search_by_symptoms()

        Returns:
            AgentResponse со структурированным предписанием.
        """

        symptoms_block = self._format_symptoms(symptom_vector)
        context_block, sources = self._format_context(rag_results)

        if context_block:
            context_section = f"КОНТЕКСТ (нормативная документация):\n{context_block}"
        else:
            context_section = ("КОНТЕКСТ: отсутствует. "
                               "Релевантной нормативной документации не найдено.")

        # Полный промпт: системная роль + симптомы + контекст
        user_prompt = (
            f"СИМПТОМЫ (данные от аналитической модели):\n{symptoms_block}\n\n"
            f"{context_section}\n\n"
            f"Сформируй предписание строго по заданному формату."
        )

        full_prompt = f"{SYSTEM_PROMPT}\n\n{'='*60}\n\n{user_prompt}"

        # Условно формируем аргументы chat: think только для reasoning-моделей
        chat_kwargs = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': full_prompt}],
            'options': self.options,
        }
        if 'qwen3' in self.model_name.lower():
            chat_kwargs['think'] = False   # отключаем chain-of-thought нативно

        start = time.time()
        try:
            resp = self.client.chat(**chat_kwargs)
        except Exception as e:
            # Не роняем UI: возвращаем AgentResponse с ошибкой
            return AgentResponse(
                pump_id=symptom_vector.pump_id,
                raw_text="", model_name=self.model_name,
                used_context=bool(context_block), sources=sources,
                error=f"Ошибка вызова Ollama: {e}",
            )
        latency = round(time.time() - start, 2)

        raw_text = resp['message']['content'].strip()
        eval_count = resp.get('eval_count', 0)
        prompt_eval_count = resp.get('prompt_eval_count', 0)
        eval_duration = resp.get('eval_duration', 0) / 1e9   # нс → с
        gen_time = round(eval_duration, 2)
        tps = round(eval_count / eval_duration, 1) if eval_duration > 0 else 0

        # Автопроверка формата: присутствуют ли все 4 обязательных раздела
        required = ['СТАТУС', 'ДИАГНОЗ', 'ПРЕДПИСАНИЕ', 'ТОиР']
        format_ok = all(any(r in line for line in raw_text.split('\n')) for r in required)

        return AgentResponse(
            pump_id=symptom_vector.pump_id,
            raw_text=raw_text, model_name=self.model_name,
            used_context=bool(context_block), sources=sources,
            latency_sec=latency, gen_time_sec=gen_time,
            eval_count=eval_count, prompt_eval_count=prompt_eval_count,
            tokens_per_sec=tps, format_ok=format_ok,
        )


# Точка входа (тестирование модуля) 

if __name__ == "__main__":
    import pandas as pd
    from xai_module import XAIExplainer
    from data_preprocessor import DataPreprocessor
    from rag_database import KnowledgeBaseManager

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path   = os.path.join(project_root, 'models', 'xgboost_pump_model.joblib')
    data_path    = os.path.join(project_root, 'data', 'processed', 'processed_features.csv')
    chroma_dir   = os.path.join(project_root, 'chroma_db')
    kb_dir       = os.path.join(project_root, 'knowledge_base')

    print("="*60)
    print("ТЕСТ AI-АГЕНТА: полная цепочка XAI → RAG → LLM")
    print("="*60)

    # 1. XAI: получаем SymptomVector для реальной аварии
    print("\n[1] XAI: вычисление вектора симптомов...")
    xai = XAIExplainer(model_path)
    df  = pd.read_csv(data_path)
    preprocessor = DataPreprocessor(window_sizes=[15, 30, 60])
    feature_cols = preprocessor.FEATURE_COLS

    critical_cases = df[df['target'] == 2]
    sample_row  = critical_cases.iloc[-1:][feature_cols]
    pump_id_val = critical_cases.iloc[-1]['pump_id']
    ts_val      = str(critical_cases.iloc[-1].get('timestamp', 'N/A'))

    sv = xai.explain(sample_row, pump_id=pump_id_val, timestamp=ts_val, top_k=5)
    print(f"  Прогноз: класс {sv.predicted_class}, P(Авария)={sv.critical_probability}%")

    # 2. RAG: ищем релевантный контекст по симптомам
    print("\n[2] RAG: поиск нормативного контекста...")
    kb = KnowledgeBaseManager(data_dir=kb_dir, chroma_dir=chroma_dir)
    symptom_dict = {
        'critical_probability': sv.critical_probability,
        'top_symptoms': [
            {'sensor': s.sensor, 'value': s.value, 'shap_weight': s.shap_weight}
            for s in sv.top_symptoms
        ]
    }
    rag_results = kb.search_by_symptoms(symptom_dict, k=4)
    print(f"  Найдено фрагментов: {len(rag_results)}")

    # 3. Agent: генерируем предписание для каждой модели
    models_to_test = LLM_MODELS

    for n, model_name in enumerate(models_to_test, 1):
        print(f"\n[3.{n}/{len(models_to_test)}] LLM-агент: {model_name}...")

        agent = DiagnosticAgent(model_name=model_name)
        response = agent.generate_prescription(sv, rag_results)

        print("="*60)
        print(f"ОТВЕТ АГЕНТА {n} (модель: {response.model_name}, "
              f"время: {response.latency_sec} с)")
        print("="*60)
        print(response.raw_text)
        print("="*60)
        print(f"Источники: {', '.join(response.sources)}")
        print(f"Контекст использован: {'да' if response.used_context else 'нет'}")