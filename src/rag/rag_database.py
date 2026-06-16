"""
Модуль формирования RAG-базы знаний (Knowledge Base Manager)
=============================================================
Стек: LangChain + ChromaDB + multilingual-e5-large (HuggingFace)
Платформа: macOS M2, локальный запуск без GPU (MPS-ускорение через PyTorch)
"""

import sys
import os
import shutil
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config.settings import (DOC_TYPES, CHUNK_CONFIG, RELEVANCE_THRESHOLD,
                              EMBED_MODEL, DOC_TYPE_MAP, FAULT_TYPES, 
                              FAULT_CONFIDENCE_THRESHOLD)
from src.visualisation.rag_visualisation import plot_all_rag

import pdfplumber
import pymupdf4llm
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

_FAULT_KEYS = set(FAULT_TYPES)
_FAULT_RU = {'overheat': 'перегрев', 'cavitation': 'кавитация',
             'electrical': 'электрическая неисправность'}
# Явный тег в заголовке сценария — самый надёжный путь.
_FAULT_TAG_RE = re.compile(r'\[fault_type:\s*([a-z_]+)\s*\]', re.I)
# Строка-заголовок сценария (для carry-forward по чанкам).
_SCENARIO_HEADER_RE = re.compile(r'(?:^|\n)#{2,3}\s+Сценари[йяе][^\n]*', re.I)
# Фолбэк-определение типа по русским словам в заголовке (если тега нет).
_HEADER_FAULT_KEYWORDS = (
    ('кавитац', 'cavitation'),
    ('перегрев', 'overheat'),
    ('электрическ', 'electrical'),
    ('электрик', 'electrical'),
)

# Подразделы сценария. Операторный заголовок дополнительно несёт СТАДИЮ.
_OPERATOR_RE = re.compile(
    r'Действия\s+оператора(?:\s*\(\s*стади[яи]\s*[«"]?\s*(Предупреждени\w*|Авари\w*))?', 
    re.I)
_REPAIR_RE = re.compile(r'(?:Связанные\s+)?работы\s+ТОиР', re.I)
_REF_RES = (
    re.compile(r'Симптоматика', re.I),
    re.compile(r'вероятные\s+причины', re.I),
    re.compile(r'Рекомендаци\w*\s+по\s+графику', re.I)
    )


def _stage_from(word):
    """Слово из заголовка → метка стадии. Без явной метки = 'critical'."""

    if not word:    # Вывод по умолчанию
        return 'critical'
    
    w = word.lower()
    if w.startswith('предупрежд'):
        return 'warning'
    
    if w.startswith('авари'):
        return 'critical'
    
    return 'critical'   # Вывод при отсутствии тегов


def resolve_stage(symptom_vector, conf_threshold: float = FAULT_CONFIDENCE_THRESHOLD) -> str:
    """
    Единый резолвер стадии для агента и бенчмарка. Возвращает:
      'warning'  — класс 1 и тип уверенно определён;
      'critical' — класс 2 и тип уверенно определён;
      'unknown'  — класс вне {1,2} ЛИБО уверенность типа ниже порога.
    """

    pc = getattr(symptom_vector, 'predicted_class', None)
    if pc not in (1, 2):
        return 'unknown'
    
    fc = getattr(symptom_vector, 'fault_confidence', 1.0)
    fc = 1.0 if fc is None else (fc if fc <= 1.0 else fc / 100.0)  # доля или проценты

    if fc < conf_threshold:
        return 'unknown'
    
    return 'warning' if pc == 1 else 'critical'


def _header_fault(line: str):
    """Тип отказа из заголовка сценария: тег [fault_type:X] или ключевое слово."""

    tag = _FAULT_TAG_RE.search(line)
    if tag and tag.group(1).lower() in _FAULT_KEYS:
        return tag.group(1).lower()
    
    low = line.lower()
    for kw, key in _HEADER_FAULT_KEYWORDS:
        if kw in low:
            return key
    return None


def _subsection_segments(scenario_text) -> List:
    """
    Режет сценарий на подразделы. Для operator-блоков извлекает СТАДИЮ из
    заголовка; для repair/reference стадия = None (стадийно-независимы).
    """

    marks = []  # (pos, sop_part, stage)
    for m in _OPERATOR_RE.finditer(scenario_text):
        marks.append((m.start(), 'operator', _stage_from(m.group(1))))
    for m in _REPAIR_RE.finditer(scenario_text):
        marks.append((m.start(), 'repair', None))
    for rgx in _REF_RES:
        for m in rgx.finditer(scenario_text):
            marks.append((m.start(), 'reference', None))
    marks.sort(key=lambda x: x[0])
 
    if not marks:
        return [('reference', None, scenario_text)]
    
    segs = []
    if marks[0][0] > 0: # заголовок сценария → reference
        segs.append(('reference', None, scenario_text[:marks[0][0]]))
    for i, (pos, part, stage) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(scenario_text)
        segs.append((part, stage, scenario_text[pos:end]))

    return segs

# Загрузчик текстовых файлов (.md, .txt)

class TextKnowledgeLoader:
    """
    Загружает .md и .txt файлы из директорий базы знаний.

    Используется для вручную подготовленных выжимок из больших PDF.
    """

    SUPPORTED_EXTENSIONS = {'.md', '.txt'}

    def load_file(self, file_path: str, doc_type: str = 'manual') -> List[Document]:
        
        path = Path(file_path)
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return []
        try:
            text = path.read_text(encoding='utf-8').strip()
        except Exception as e:
            print(f"  [WARN] Не удалось прочитать {path.name}: {e}")
            return []
        if len(text) < 50:
            return []
        return [Document(
            page_content=text,
            metadata={
                'source':   path.name,
                'doc_type': doc_type,
                'loader':   'text_loader',
                'full_path': str(path.resolve()),
            }
        )]

    def load_directory(self, data_dir: str,
                       doc_type_map: Dict[str, str] = None) -> List[Document]:  # type: ignore
        
        doc_type_map = doc_type_map or {}
        all_documents: List[Document] = []

        text_names = {name for name in doc_type_map
                      if Path(name).suffix.lower() in self.SUPPORTED_EXTENSIONS}
        text_files = sorted(
            f for ext in self.SUPPORTED_EXTENSIONS
            for f in Path(data_dir).glob(f'**/*{ext}')
            if f.name in text_names
        )
        for file_path in text_files:
            doc_type = doc_type_map.get(file_path.name, 'manual')
            docs = self.load_file(str(file_path), doc_type=doc_type)
            if docs:
                print(f"  Загрузка [текст/{doc_type}]: {file_path.name} → {len(docs[0].page_content)} символов")
            all_documents.extend(docs)
        return all_documents


# Загрузчик PDF с сохранением структуры
# Оставлен в качестве расширения возможностей, в текущей версии не используется

class StructuredPDFLoader:
    """
    Загрузчик PDF на базе pymupdf4llm.
    Конвертирует PDF в Markdown, сохраняя:
      - заголовки разделов (## Раздел 4.2)
      - таблицы (таблица неисправностей — ключевой источник для RAG)
      - нумерованные списки (пошаговые инструкции)

    Fallback: если pymupdf4llm не извлёк текст (сканированный PDF) —
    используется pdfplumber с постраничным разбиением.
    """

    def load(self, pdf_path: str, doc_type: str = 'manual') -> List[Document]:
        """
        Загружает один PDF-файл и возвращает список Document с метаданными.

        Args:
            pdf_path: Абсолютный путь к PDF.
            doc_type: Тип документа (ключ из DOC_TYPES).

        Returns:
            Список Document; каждый содержит page_content и metadata.
        """

        source_name = Path(pdf_path).name
        documents = []

        try:
            # pymupdf4llm: PDF → Markdown (сохраняет таблицы и заголовки)
            md_text = pymupdf4llm.to_markdown(pdf_path)

            if len(md_text.strip()) > 100:
                documents.append(Document(
                    page_content=md_text,
                    metadata={
                        'source':      source_name,
                        'doc_type':    doc_type,
                        'loader':      'pymupdf4llm',
                        'full_path':   pdf_path,
                    }
                ))
                return documents

        except Exception:
            pass  # Fallback ниже

        # Fallback: pdfplumber — постраничный текст
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ''
                    if len(text.strip()) > 50:
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                'source':    source_name,
                                'doc_type':  doc_type,
                                'page':      page_num,
                                'loader':    'pdfplumber',
                                'full_path': pdf_path,
                            }
                        ))
        except Exception as e:
            print(f"  [WARN] Не удалось загрузить {source_name}: {e}")

        return documents

    def load_directory(self, data_dir: str,
                       doc_type_map: Dict[str, str] = None) -> List[Document]:  # type: ignore
        """
        Загружает PDF-файлы, явно перечисленные в doc_type_map.

        Args:
            data_dir:     Путь к директории с PDF.
            doc_type_map: Словарь {имя_файла: doc_type} — загружаются только эти файлы.

        Returns:
            Список всех Document из указанных файлов.
        """

        doc_type_map = doc_type_map or {}
        all_documents: List[Document] = []

        pdf_names = {name for name in doc_type_map if name.lower().endswith('.pdf')}
        pdf_files = sorted(
            f for f in Path(data_dir).glob('**/*.pdf')
            if f.name in pdf_names
        )

        if not pdf_files:
            print("  [INFO] PDF к загрузке нет (база использует .md); "
                "StructuredPDFLoader с починкой лигатур доступен для будущих PDF.")
            return all_documents

        for pdf_path in pdf_files:
            fname = pdf_path.name
            doc_type = doc_type_map.get(fname, 'manual')
            print(f"  Загрузка [{DOC_TYPES.get(doc_type, doc_type)}]: {fname}")
            docs = self.load(str(pdf_path), doc_type=doc_type)
            all_documents.extend(docs)
            print(f"    → {len(docs)} блок(ов) текста")

        return all_documents


# Основной класс 

class KnowledgeBaseManager:
    """
    Управляет полным пайплайном RAG-базы знаний:
        PDF → Markdown → чанки → эмбеддинги → ChromaDB

    Модель эмбеддингов: intfloat/multilingual-e5-large
        Причина выбора: значительно лучше понимает технический русский
        по сравнению с MiniLM.
    """

    EMBED_MODEL = EMBED_MODEL

    def __init__(self, data_dir: str, chroma_dir: str,
                 doc_type_map: Dict[str, str] = None):  # type: ignore
        """
        Args:
            data_dir:     Директория с документами (PDF + MD/TXT).
            chroma_dir:   Директория для хранения ChromaDB.
            doc_type_map: {имя_файла: doc_type} — загружаются только перечисленные файлы.
        """

        self.data_dir = data_dir
        self.chroma_dir = chroma_dir
        self.doc_type_map = doc_type_map or {}
        self.loader = StructuredPDFLoader()
        self.text_loader = TextKnowledgeLoader()

        print(f"Загрузка модели эмбеддингов: {self.EMBED_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.EMBED_MODEL,
            model_kwargs={'device': 'mps'},   # M2 Apple Silicon
            encode_kwargs={'normalize_embeddings': True},  # Требуется для e5
        )

    # Построение базы 

    def build_database(self, reset: bool = False) -> Optional[Chroma]:
        """
        Полный пайплайн: PDF → чанки → ChromaDB.

        Args:
            reset: True — удалить существующую базу и пересобрать.

        Returns:
            Объект Chroma (vectorstore).
        """

        if reset and os.path.exists(self.chroma_dir):
            print(f"Удаление старой базы: {self.chroma_dir}")
            shutil.rmtree(self.chroma_dir)

        print(f"\n{'─'*55}")
        print("Шаг 1: Загрузка документов")
        print(f"{'─'*55}")

        # Текстовые файлы (.md, .txt) загружаются первыми — они содержат
        # вручную подготовленные выжимки из больших PDF (gost_extract.md и т.п.)
        text_docs = self.text_loader.load_directory(
            self.data_dir,
            doc_type_map=self.doc_type_map,
        )
        if text_docs:
            print(f"  Текстовых документов загружено: {len(text_docs)}")

        pdf_docs = self.loader.load_directory(self.data_dir, self.doc_type_map)
        documents = text_docs + pdf_docs

        if not documents:
            print("[ERROR] Документы не найдены. Добавьте PDF в папку knowledge_base/")
            return None

        print(f"\n{'─'*55}")
        print("Шаг 2: Разбиение на чанки")
        print(f"{'─'*55}")
        chunks = self._split_documents(documents)
        print(f"Итого чанков: {len(chunks)}")

        print(f"\n{'─'*55}")
        print("Шаг 3: Генерация эмбеддингов и сохранение в ChromaDB")
        print(f"{'─'*55}")
        os.makedirs(self.chroma_dir, exist_ok=True)

        # Батчевое добавление: ChromaDB может зависнуть на большом объёме за раз
        db = self._build_chroma_batched(chunks)

        print(f"\nБаза знаний готова: {self.chroma_dir}")
        print(f"Всего документов в ChromaDB: {db._collection.count()}")
        return db
    
    # Хелпер: определение типа отказа чанка SOP (метод класса) 
    def _detect_chunk_fault(self, raw_text: str, current):
        """
        Определяет fault_type для чанка SOP по заголовкам сценариев в нём.
        carry-forward: если в чанке нет заголовка — наследуется тип предыдущего
        (под-чанки длинного сценария теряют заголовок при сплите, но не привязку).
        Берёт последний заголовок в чанке — корректно для границы сценариев.
        """

        found = current
        for m in _SCENARIO_HEADER_RE.finditer(raw_text):
            line = m.group(0)
            tag = _FAULT_TAG_RE.search(line)
            if tag and tag.group(1).lower() in _FAULT_KEYS:
                found = tag.group(1).lower()
                continue
            low = line.lower()
            for kw, key in _HEADER_FAULT_KEYWORDS:
                if kw in low:
                    found = key
                    break
        return found

    def _split_documents(self, documents):
        """
        Разбивает документы на чанки с параметрами под каждый тип документа.
        Добавляет метаданные секции (первые 80 символов чанка как заголовок).

        Для doc_type='sop' навешивает fault_type (carry-forward),
        чтобы сценарные поиски можно было детерминированно фильтровать по типу.
        """

        all_chunks = []
        by_type = {}
        for doc in documents:
            dt = doc.metadata.get('doc_type', 'manual')
            by_type.setdefault(dt, []).append(doc)

        for doc_type, docs in by_type.items():
            cfg = CHUNK_CONFIG.get(doc_type, CHUNK_CONFIG['manual'])
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg['chunk_size'], chunk_overlap=cfg['chunk_overlap'],
                length_function=len,
                separators=['\n## ', '\n### ', '\n\n', '\n', ' ', ''])

            if doc_type == 'sop':
                chunks = []
                for d in docs:
                    chunks.extend(self._chunk_sop_document(d, splitter))
            else:
                chunks = splitter.split_documents(docs)   # путь прочих типов не меняется

            tagged = 0
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                chunk.metadata['section'] = chunk.page_content[:80].replace('\n', ' ')
                chunk.page_content = f"passage: {chunk.page_content}"
                if doc_type == 'sop' and chunk.metadata.get('fault_type'):
                    tagged += 1

            all_chunks.extend(chunks)
            extra = (f", сценарных чанков с fault_type: {tagged}, sop_part размечен"
                    if doc_type == 'sop' else "")
            print(f"  [{doc_type:8s}] {len(docs)} блок(ов) → {len(chunks)} чанков "
                f"(size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}{extra})")
            if doc_type == 'sop' and tagged == 0:
                print("  [WARN] Сценарии SOP не размечены fault_type — проверьте заголовки "
                    "'### Сценарий ... [fault_type: ...]'.")
        return all_chunks
    
    def _chunk_sop_document(self, doc, splitter):
        """
        Режет регламент на регионы → подразделы → чанки с метаданными 
        fault_type / sop_part / stage.
        Возвращает list[Document] (без passage-префикса — он добавится в _split_documents).
        """
        
        text = doc.page_content
        base = dict(doc.metadata)

        # 1) Регионы: '### Сценарий ...' → сценарий с типом; '## ...' → общий (fault=None).
        regions, cur_fault, buf = [], None, []

        def _flush():
            if buf and any(l.strip() for l in buf):
                regions.append((cur_fault, '\n'.join(buf)))

        for line in text.split('\n'):
            s = line.strip()
            if s.startswith('### ') and 'Сценари' in s:
                _flush()
                buf = [line]
                cur_fault = _header_fault(s)
            elif s.startswith('## ') and not s.startswith('### '):
                _flush()
                buf = [line]
                cur_fault = None
            else:
                buf.append(line)
        _flush()

        # 2) Подразделы (для сценариев) / целиком (для общих секций) → чанки с метаданными.
        out = []
        for fault, region in regions:
            segs = _subsection_segments(region) if fault else [('reference', None, region)]
            for sop_part, stage, seg in segs:
                for ch in splitter.split_text(seg):
                    if not ch.strip():
                        continue
                    meta = dict(base)
                    meta['sop_part'] = sop_part
                    if stage: 
                        meta['stage'] = stage   # тег стадии — только на operator-чанках
                    if fault:
                        meta['fault_type'] = fault
                    out.append(Document(page_content=ch, metadata=meta))

        return out

    def _build_chroma_batched(self, chunks: List[Document],
                               batch_size: int = 500) -> Chroma:
        """
        Добавляет чанки в ChromaDB батчами.
        ChromaDB может зависнуть при одновременной записи >1000 документов.
        """

        db = None
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            if db is None:
                db = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=self.chroma_dir,
                )
            else:
                db.add_documents(batch)
            print(f"  Записано {min(i + batch_size, len(chunks))}/{len(chunks)} чанков...")

        return db # type: ignore

    # Поиск 

    def search(self, query, k=4, doc_type_filter=None,
           metadata_filter=None, apply_threshold=True):
        """
        Семантический поиск. Добавлены:
        metadata_filter: доп. условия по метаданным (например {'fault_type': 'cavitation'});
        apply_threshold: отключаемая отсечка по RELEVANCE_THRESHOLD — нужна для
                        сценарного фолбэка (тип уже гарантирован метаданными,
                        дистанция вторична).
        Обратная совместимость: старые вызовы search(q, k, doc_type_filter) не меняются.
        """

        db = Chroma(persist_directory=self.chroma_dir,
                    embedding_function=self.embeddings)
        prefixed_query = f"query: {query}"           # e5: prefix для запроса

        conds = []
        if doc_type_filter:
            conds.append({'doc_type': doc_type_filter})
        if metadata_filter:
            for kk, vv in metadata_filter.items():
                conds.append({kk: vv})

        search_kwargs = {'k': k}
        if len(conds) == 1:
            search_kwargs['filter'] = conds[0]
        elif len(conds) > 1:
            search_kwargs['filter'] = {'$and': conds} # type: ignore

        results = db.similarity_search_with_score(prefixed_query, **search_kwargs) # type: ignore
        
        if not apply_threshold:
            return results
        relevant = [(doc, score) for doc, score in results
                    if score <= RELEVANCE_THRESHOLD]
        if len(relevant) < len(results):
            print(f"  [INFO] Отфильтровано {len(results) - len(relevant)} нерелевантных "
                f"(distance > {RELEVANCE_THRESHOLD})")
        return relevant

    def search_by_symptoms(self, symptom_vector_dict: dict, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Поиск по симптомам из XAI-модуля через multi-query retrieval.

        Делает ДВА запроса:
        1. Описательный — находит пороги/нормативы (что превышено).
        2. Прескриптивный — находит причины и действия (что делать).
        Это устраняет проблему, когда поиск находит только уставки,
        но не находит раздел "Действия оператора".
        """

        symptoms = symptom_vector_dict.get('top_symptoms', [])
        prob = symptom_vector_dict.get('critical_probability', 0)
        sensor_map = {'vibration': 'вибрация', 'temperature': 'температура',
                    'current': 'ток', 'pressure': 'давление'}
        inferred_fault = symptom_vector_dict.get('inferred_fault')
        fault_ru = {'overheat': 'перегрев', 'cavitation': 'кавитация',
                    'electrical': 'электрическая неисправность'}.get(inferred_fault, '') # type: ignore

        # Запрос 1: описание состояния (пороги, нормативы)
        parts = [f"Вероятность аварии: {prob}%."]
        for s in symptoms:
            sensor_ru = sensor_map.get(s.get('sensor', ''), s.get('sensor', ''))
            direction = "повышена" if s.get('shap_weight', 0) > 0 else "понижена"
            parts.append(f"{sensor_ru} {direction}: значение {s.get('value', '')}")
        query_descriptive = " ".join(parts)

        # Запрос 2: действия и причины (прескриптивный)
        symptom_words = " ".join(
            sensor_map.get(s.get('sensor', ''), '') for s in symptoms
        )
        query_prescriptive = (
            f"причина и устранение неисправности: {symptom_words} {fault_ru}. "
            f"Действия оператора, диагностика отказа, связанные работы ТОиР, "
            f"капитальный ремонт, рекомендации."
        )

        print(f"  Запрос 1 (состояние): {query_descriptive[:90]}...")
        print(f"  Запрос 2 (действия):  {query_prescriptive[:90]}...")

        # Объединяем результаты с дедупликацией по содержимому
        seen = set()
        combined = []
        for q in (query_descriptive, query_prescriptive):
            for doc, score in self.search(q, k=k):
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    combined.append((doc, score))

        # Справочный контекст — ТОЛЬКО для обоснования диагноза: убираем из него
        # подразделы «действия оператора» и «работы ТОиР» любого сценария, чтобы
        # модель не перенесла их в ПРЕДПИСАНИЕ/ТОиР (замечание №5). Фильтр стоит
        # ИМЕННО здесь, а не в общем search(), иначе ломаются operator/repair-поиски.
        combined = [(d, s) for (d, s) in combined
                    if d.metadata.get('sop_part') not in ('operator', 'repair')]
        combined.sort(key=lambda x: x[1])

        return combined[:k]
    
    def search_maintenance_schedule(self, pump_id: str, k: int = 2):
        """
        Прямой поиск графика ТО по агрегату (отдельно от поиска по симптомам).
        График привязан к pump_id, а не к симптомам датчиков, поэтому
        поиск по симптомам его не находит — нужен прямой запрос.
        """

        SCHEDULE_THRESHOLD = 1.5 # Мягче основного threshold
        query = f"график технического обслуживания капитальный ремонт {pump_id}"
        db = Chroma(persist_directory=self.chroma_dir, embedding_function=self.embeddings)
        results = db.similarity_search_with_score(
            f"query: {query}", k=k,
            filter={'doc_type': 'schedule'}   # фильтр только по графику ТО
        )
        return [(doc, score) for doc, score in results if score <= SCHEDULE_THRESHOLD]
    
    def search_repair_works(self, fault_type, k=2):
        """«Связанные работы ТОиР» — ТОЛЬКО из сценария верного типа отказа."""

        fault_ru = _FAULT_RU.get(fault_type, '')
        if not fault_ru:
            return []
        query = (f"связанные работы ТОиР при отказе типа {fault_ru}: "
                f"дефектоскопия, замена, центровка, балансировка, ТО-1, ТО-2")
        exact = {'fault_type': fault_type, 'sop_part': 'repair'}

        res = self.search(query, k=k, doc_type_filter='sop', metadata_filter=exact)
        if res:
            return res
        res = self.search(query, k=k, doc_type_filter='sop', metadata_filter=exact,
                        apply_threshold=False)
        if res:
            return res
        res = self.search(query, k=k, doc_type_filter='sop',
                        metadata_filter={'fault_type': fault_type})
        if res:
            return res
        return self.search(query, k=k, doc_type_filter='sop')
    
    def search_operator_actions(self, fault_type, stage='critical', k=2):
        """
        «Действия оператора» строго из нужного сценария И нужной СТАДИИ.
        stage: 'warning' (Предупреждение) | 'critical' (Авария).

        Лестница фолбэков ослабляет фильтр постепенно: стадия → любой operator-блок
        сценария → любой чанк сценария → весь SOP.
        """

        if str(stage).lower() == 'unknown':
            return [] 

        fault_ru = _FAULT_RU.get(fault_type, '')
        if not fault_ru:
            return []
        stage = ('warning' if str(stage).lower() in ('warning', 'предупреждение', '1') 
                 else 'critical')
        tone = ('упреждающие действия оператора' if stage == 'warning'
                else 'действия оператора при аварии')
        query = f"{tone} при отказе типа {fault_ru}"
    
        exact = {'fault_type': fault_type, 'sop_part': 'operator', 'stage': stage}
        res = self.search(query, k=k, doc_type_filter='sop', metadata_filter=exact)

        # Постепенно ослабляет соответствие контексту для однозначной выдачи результата
        if res:
            return res
        res = self.search(query, k=k, doc_type_filter='sop', metadata_filter=exact,
                        apply_threshold=False)
        if res:
            return res
        res = self.search(query, k=k, doc_type_filter='sop',
                        metadata_filter={'fault_type': fault_type, 'sop_part': 'operator'})
        if res:
            return res
        res = self.search(query, k=k, doc_type_filter='sop',
                        metadata_filter={'fault_type': fault_type})
        if res:
            return res
        return self.search(query, k=k, doc_type_filter='sop')


# Точка входа 

if __name__ == "__main__":
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))))
    knowledge_dir = os.path.join(project_root, 'knowledge_base')
    chroma_db_dir = os.path.join(project_root, 'artifacts', 'chroma_db')
    plots_dir = os.path.join(project_root, 'artifacts', 'graphs')

    os.makedirs(knowledge_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    kb = KnowledgeBaseManager(
        data_dir=knowledge_dir,
        chroma_dir=chroma_db_dir,
        doc_type_map=DOC_TYPE_MAP,
    )

    # Шаг 1: Построить базу
    db = kb.build_database(reset=True)

    if db is not None:
        # Шаг 2: Визуализация базы знаний
        test_queries = [
            {'query': 'температура подшипника выше 93 градусов, перегрев',
             'label': 'Темп. >93°C'},
            {'query': 'падение давления нагнетания',
             'label': 'Давление↓'},
            {'query': 'износ торцевого уплотнения утечка рабочей жидкости',
             'label': 'Уплотнение'},
        ]

        plot_all_rag(kb, test_queries, plots_dir)

        # Шаг 3: Демонстрация поиска в консоли
        print(f"\n{'─'*55}")
        print("Тестовый поиск (сценарий: высокая вибрация + температура):")
        print(f"{'─'*55}")
        symptom_vec = {
            'critical_probability': 87.3,
            'inferred_fault': 'overheat',
            'top_symptoms': [
                {'sensor': 'temperature', 'value': 94.1,'shap_weight': 3.4},
                {'sensor': 'pressure', 'value': 1.3, 'shap_weight': 2.2},
            ]
        }
        results_symptoms = kb.search_by_symptoms(symptom_vec, k=3)

        for i, (doc, score) in enumerate(results_symptoms, 1):
            print(f"\n[{i}] Distance: {score:.4f} | "
                  f"Тип: {doc.metadata.get('doc_type')} | "
                  f"Источник: {doc.metadata.get('source')}")
            # Убираем prefix перед выводом
            clean_text = doc.page_content.replace('passage: ', '', 1)
            print(f"    {clean_text[:250]}...")

        # Шаг 4: Проверка поиска в графике обслуживания
        scheduled_maintenance = 'MNHV_005'

        print(f"\n{'─'*55}")
        print(f"Тестовый поиск (плановое обслуживание насоса {scheduled_maintenance}):")
        print(f"{'─'*55}")

        results_schedule = kb.search_maintenance_schedule(scheduled_maintenance)

        for i, (doc, score) in enumerate(results_schedule, 1):
            print(f"\n[{i}] Distance: {score:.4f} | "
                  f"Тип: {doc.metadata.get('doc_type')} | "
                  f"Источник: {doc.metadata.get('source')}")
            # Убираем prefix перед выводом
            clean_text = doc.page_content.replace('passage: ', '', 1)
            print(f"    {clean_text[:250]}...")