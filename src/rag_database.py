"""
Модуль формирования RAG-базы знаний (Knowledge Base Manager)
=============================================================
Стек: LangChain + ChromaDB + multilingual-e5-large (HuggingFace)
Платформа: macOS M2, локальный запуск без GPU (MPS-ускорение через PyTorch)
"""

import os
import shutil
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pdfplumber
import pymupdf4llm
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import pandas as pd


# Константы

# Типы документов базы знаний
DOC_TYPES = {
    'manual':   'Технический мануал (руководство по эксплуатации)',
    'gost':     'ГОСТ / Нормативный документ',
    'sop':      'Регламент технического обслуживания (SOP)',
    'schedule': 'График технического обслуживания',
    'diagnostics': 'Расширенная вибродиагностика (аналитическое дополнение)',
}

# Параметры чанкинга по типу документа
CHUNK_CONFIG = {
    'manual':   {'chunk_size': 600,  'chunk_overlap': 120},
    'gost':     {'chunk_size': 700,  'chunk_overlap': 150},   # короткие пункты ГОСТ
    'sop':      {'chunk_size': 500,  'chunk_overlap': 100},
    'schedule': {'chunk_size': 350,  'chunk_overlap': 50},
    'diagnostics': {'chunk_size': 500, 'chunk_overlap': 100},  # новый тип
}

# Порог релевантности: результаты с distance > threshold отсекаются
RELEVANCE_THRESHOLD = 1.2   # L2-расстояние; > 1.2 = нерелевантно для e5-large


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
            print(f"  [WARN] PDF-файлы не найдены в {data_dir}")
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

    EMBED_MODEL = "intfloat/multilingual-e5-large"

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

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Разбивает документы на чанки с параметрами под каждый тип документа.
        Добавляет метаданные секции (первые 80 символов чанка как заголовок).
        """
        all_chunks: List[Document] = []

        # Группируем по doc_type для применения разных настроек
        by_type: Dict[str, List[Document]] = {}
        for doc in documents:
            dt = doc.metadata.get('doc_type', 'manual')
            by_type.setdefault(dt, []).append(doc)

        for doc_type, docs in by_type.items():
            cfg = CHUNK_CONFIG.get(doc_type, CHUNK_CONFIG['manual'])
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg['chunk_size'],
                chunk_overlap=cfg['chunk_overlap'],
                length_function=len,
                separators=['\n## ', '\n### ', '\n\n', '\n', ' ', ''],
            )
            chunks = splitter.split_documents(docs)

            # Добавляем метаданные: порядковый номер и первые слова как section
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                chunk.metadata['section'] = chunk.page_content[:80].replace('\n', ' ')
                # e5 требует prefix для passage (при поиске — prefix 'query:')
                chunk.page_content = f"passage: {chunk.page_content}"

            all_chunks.extend(chunks)
            print(f"  [{doc_type:8s}] {len(docs)} блок(ов) → {len(chunks)} чанков "
                  f"(size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']})")

        return all_chunks

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

    def search(self, query: str, k: int = 4,
               doc_type_filter: str = None) -> List[Tuple[Document, float]]: # type: ignore
        """
        Семантический поиск по базе знаний.

        Args:
            query:           Поисковый запрос (симптомы от XAI-модуля).
            k:               Число результатов.
            doc_type_filter: Фильтр по типу документа (например 'manual').

        Returns:
            Список (Document, distance); отфильтровано по RELEVANCE_THRESHOLD.
        """
        db = Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=self.embeddings
        )

        # e5 требует prefix 'query:' для поисковых запросов
        prefixed_query = f"query: {query}"

        search_kwargs = {'k': k}
        if doc_type_filter:
            search_kwargs['filter'] = {'doc_type': doc_type_filter} # type: ignore

        results = db.similarity_search_with_score(prefixed_query, **search_kwargs) # type: ignore

        # Фильтрация нерелевантных результатов
        relevant = [(doc, score) for doc, score in results
                    if score <= RELEVANCE_THRESHOLD]

        if len(relevant) < len(results):
            print(f"  [INFO] Отфильтровано {len(results) - len(relevant)} нерелевантных результатов "
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
            f"причина и устранение неисправности: {symptom_words}. "
            f"Действия оператора, диагностика отказа, рекомендации по ремонту."
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

        # Сортируем по релевантности и возвращаем top-k
        combined.sort(key=lambda x: x[1])
        return combined[:k]

    # Визуализация для диплома 

    def plot_knowledge_base_stats(self, save_dir: str):
        """
        График 1: Состав базы знаний.
        Доля чанков по типам документов — демонстрирует покрытие базы знаний.
        """
        db = Chroma(persist_directory=self.chroma_dir,
                    embedding_function=self.embeddings)
        all_meta = db._collection.get()['metadatas']

        if not all_meta:
            print("[WARN] База пуста — нечего визуализировать.")
            return

        # Подсчёт по типам
        type_counts: Dict[str, int] = {}
        for m in all_meta:
            dt = m.get('doc_type', 'unknown')
            type_counts[dt] = type_counts.get(dt, 0) + 1 # type: ignore

        all_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
        sorted_items = sorted(
            zip(type_counts.keys(), type_counts.values()),
            key=lambda x: x[1], reverse=True
        )
        labels = [DOC_TYPES.get(k, k) for k, _ in sorted_items]
        sizes  = [v for _, v in sorted_items]
        colors = all_colors[:len(labels)]

        fig, ax = plt.subplots(figsize=(8, 7))
        wedges, texts, autotexts = ax.pie( # type: ignore
            sizes, labels=None, autopct='%1.1f%%',
            colors=colors, startangle=90, counterclock=False,
            pctdistance=0.82, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight('bold')
        ax.legend(wedges, [f"{l}\n({s} чанков)" for l, s in zip(labels, sizes)],
                  loc='lower center', bbox_to_anchor=(0.5, -0.22),
                  fontsize=9, framealpha=0.8)
        ax.set_title(
            f'Состав базы знаний RAG-системы\n(всего чанков: {sum(sizes)})',
            fontsize=13, fontweight='bold', pad=6
        )
        path = os.path.join(save_dir, 'rag_plot1_kb_composition.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"График 1 сохранён: {path}")

    def plot_chunk_length_distribution(self, save_dir: str):
        """
        График 2: Распределение длин чанков по типам документов.
        Валидирует, что chunk_size выбран правильно — чанки не слишком короткие
        (теряется контекст) и не слишком длинные (размываются по теме).
        """
        db = Chroma(persist_directory=self.chroma_dir,
                    embedding_function=self.embeddings)
        collection = db._collection.get()

        if not collection['documents']:
            print("[WARN] База пуста.")
            return

        docs_by_type: Dict[str, List[int]] = {}
        for doc_text, meta in zip(collection['documents'], collection['metadatas']): # type: ignore
            # Убираем prefix 'passage: ' для подсчёта реальной длины
            clean = doc_text.replace('passage: ', '', 1)
            dt    = meta.get('doc_type', 'unknown')
            docs_by_type.setdefault(dt, []).append(len(clean))

        fig, ax = plt.subplots(figsize=(11, 6))
        # 6 цветов — с запасом на все типы документов (не 4!)
        palette = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']

        types_sorted = list(docs_by_type.keys())
        data   = [docs_by_type[dt] for dt in types_sorted]
        labels = [f"{DOC_TYPES.get(dt, dt)}\n(n={len(docs_by_type[dt])}, μ={np.mean(docs_by_type[dt]):.0f})"
                  for dt in types_sorted]

        bp = ax.boxplot(data, vert=False, patch_artist=True,
                        tick_labels=labels,
                        medianprops={'color': 'black', 'lw': 1.5})
        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Накладываем точки (strip plot) — при малом N важно видеть каждый чанк
        for i, dt in enumerate(types_sorted, 1):
            y = np.random.normal(i, 0.04, size=len(docs_by_type[dt]))
            ax.scatter(docs_by_type[dt], y, color=palette[(i-1) % len(palette)],
                       edgecolor='black', s=22, zorder=3, alpha=0.8)

        # Целевые chunk_size — вертикальные линии
        for dt in types_sorted:
            cfg = CHUNK_CONFIG.get(dt)
            if cfg:
                ax.axvline(cfg['chunk_size'], color='red', lw=1.0, ls='--', alpha=0.4)

        ax.set_xlabel('Длина чанка (символов)', fontsize=11)
        ax.set_title('Распределение длин текстовых фрагментов (чанков)\n'
                     'по типам документов базы знаний (N=59)',
                     fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.35, ls='--')
        plt.tight_layout()
        path = os.path.join(save_dir, 'rag_plot2_chunk_distribution.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"График 2 сохранён: {path}")

    def plot_retrieval_quality(self, test_queries: List[Dict],
                               save_dir: str):
        """
        График 3: Качество поиска — расстояния для набора тестовых запросов.
        Показывает, насколько уверенно RAG-система находит релевантный контекст
        для типичных симптомов из XAI-модуля.

        Args:
            test_queries: Список {'query': str, 'label': str} — тестовые сценарии.
        """
        results_data = []
        for tq in test_queries:
            results = self.search(tq['query'], k=3)
            for rank, (doc, score) in enumerate(results, 1):
                results_data.append({
                    'query_label': tq['label'],
                    'rank':        rank,
                    'distance':    score,
                    'doc_type':    doc.metadata.get('doc_type', 'unknown'),
                })

        if not results_data:
            print("[WARN] Нет результатов для визуализации качества.")
            return

        df = pd.DataFrame(results_data)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Левый: boxplot расстояний по запросам
        query_labels = df['query_label'].unique()
        data_per_query = [df[df['query_label'] == ql]['distance'].values
                          for ql in query_labels]
        bp = axes[0].boxplot(data_per_query, tick_labels=query_labels, patch_artist=True,
                             medianprops={'color': 'red', 'lw': 2})
        colors_box = ['#4C72B0', '#55A868', '#C44E52', '#8172B2',
                      '#CCB974', '#64B5CD'][:len(query_labels)]
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0].axhline(RELEVANCE_THRESHOLD, color='orange', ls='--', lw=1.5,
                        label=f'Порог релевантности ({RELEVANCE_THRESHOLD})')
        axes[0].set_ylabel('L2-расстояние (меньше = лучше)', fontsize=10)
        axes[0].set_title('Расстояния поиска по тестовым запросам\n'
                          '(ниже порога = релевантный результат)', fontsize=11, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].tick_params(axis='x', rotation=25)
        axes[0].grid(axis='y', alpha=0.35, ls='--')

        # Правый: доля результатов выше/ниже порога (порядок — как на левом графике)
        df['relevant'] = df['distance'] <= RELEVANCE_THRESHOLD
        summary = df.groupby('query_label')['relevant'].mean() * 100
        summary = summary.reindex(query_labels)

        bar_colors = colors_box[:len(summary)]
        bars = axes[1].bar(summary.index, summary.values, color=bar_colors, alpha=0.85,
                           edgecolor='white')
        axes[1].axhline(100, color='green', ls='--', lw=1.2, alpha=0.5, label='100% релевантность')
        axes[1].set_ylim(0, 115)
        axes[1].set_ylabel('% релевантных результатов', fontsize=10)
        axes[1].set_title('Доля релевантных результатов\nпо каждому тестовому сценарию',
                          fontsize=11, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=25)
        axes[1].grid(axis='y', alpha=0.35, ls='--')
        for bar, val in zip(bars, summary.values):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 2,
                         f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')

        plt.suptitle('Оценка качества семантического поиска RAG-системы\n'
                     f'Модель эмбеддингов: {self.EMBED_MODEL}',
                     fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(save_dir, 'rag_plot3_retrieval_quality.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"График 3 сохранён: {path}")


# Точка входа 

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    knowledge_dir = os.path.join(project_root, 'knowledge_base')
    chroma_db_dir = os.path.join(project_root, 'chroma_db')
    plots_dir = os.path.join(project_root, 'data', 'graphs')

    os.makedirs(knowledge_dir, exist_ok=True)
    os.makedirs(plots_dir,     exist_ok=True)

    # Карта типов документов: загружаются только перечисленные здесь файлы.
    # gost_32601_2013.pdf и mnhv_manual.pdf заменены .md-выжимками и не включены.
    doc_type_map = {
        'tm_regulation.pdf': 'sop',
        'tm_schedule.pdf':   'schedule',
        'gost_extract.md':   'gost',
        'mnhv_extract.md':   'manual',
        'diagnostics_extended.md': 'diagnostics',
    }

    kb = KnowledgeBaseManager(
        data_dir=knowledge_dir,
        chroma_dir=chroma_db_dir,
        doc_type_map=doc_type_map,
    )

    # Шаг 1: Построить базу
    db = kb.build_database(reset=True)

    if db is not None:
        # Шаг 2: Визуализация состава базы
        print("\nГенерация графиков для диплома...")
        kb.plot_knowledge_base_stats(plots_dir)
        kb.plot_chunk_length_distribution(plots_dir)

        # Шаг 3: Тестовые запросы (симулируем вывод XAI-модуля)
        test_queries = [
            {'query': 'вибрация подшипника превышает 8 мм/с, нарастающий тренд',
             'label': 'Вибрация >8'},
            {'query': 'температура подшипника выше 93 градусов, перегрев',
             'label': 'Темп. >93°C'},
            {'query': 'падение давления нагнетания при росте тока двигателя',
             'label': 'Давление↓ / Ток↑'},
            {'query': 'износ торцевого уплотнения утечка рабочей жидкости',
             'label': 'Уплотнение'},
        ]

        kb.plot_retrieval_quality(test_queries, plots_dir)

        # Шаг 4: Демонстрация поиска в консоли
        print(f"\n{'─'*55}")
        print("Тестовый поиск (сценарий: высокая вибрация + температура):")
        print(f"{'─'*55}")
        symptom_vec = {
            'critical_probability': 87.3,
            'top_symptoms': [
                {'sensor': 'vibration',   'value': 8.7, 'shap_weight':  0.42},
                {'sensor': 'temperature', 'value': 94.1,'shap_weight':  0.31},
                {'sensor': 'pressure',    'value': 1.1, 'shap_weight': -0.18},
            ]
        }
        results = kb.search_by_symptoms(symptom_vec, k=3)

        for i, (doc, score) in enumerate(results, 1):
            print(f"\n[{i}] Distance: {score:.4f} | "
                  f"Тип: {doc.metadata.get('doc_type')} | "
                  f"Источник: {doc.metadata.get('source')}")
            # Убираем prefix перед выводом
            clean_text = doc.page_content.replace('passage: ', '', 1)
            print(f"    {clean_text[:250]}...")