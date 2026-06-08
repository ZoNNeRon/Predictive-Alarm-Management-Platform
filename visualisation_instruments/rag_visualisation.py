"""
Блок визуализации RAG-базы знаний
=================================
Три графика для диплома, вынесенные из KnowledgeBaseManager (SRP):
  1. Состав базы знаний — доля чанков по типам документов (pie chart)
  2. Распределение длин чанков — boxplot + strip plot по типам
  3. Качество семантического поиска — L2-расстояния и доля релевантных результатов

Вызов из rag_database.py:
    from rag_visualisation import plot_all_rag
    plot_all_rag(kb.chroma_dir, kb.embeddings, kb.EMBED_MODEL, test_queries, plots_dir)

Автономный запуск:
    python rag_visualisation.py
"""

import os
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_chroma import Chroma

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config.settings import DOC_TYPES, CHUNK_CONFIG, RELEVANCE_THRESHOLD, EMBED_MODEL


# Вспомогательная функция поиска

def _search(chroma_dir: str, embeddings, query: str, k: int = 3) -> list:
    """Семантический поиск с фильтрацией нерелевантных результатов."""
    db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    results = db.similarity_search_with_score(f"query: {query}", k=k)
    return [(doc, score) for doc, score in results if score <= RELEVANCE_THRESHOLD]


# Три графика

def plot_knowledge_base_stats(chroma_dir: str, embeddings, save_dir: str):
    """
    График 1: Состав базы знаний.
    Доля чанков по типам документов — демонстрирует покрытие базы знаний.
    """

    db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    all_meta = db._collection.get()['metadatas']

    if not all_meta:
        print("[WARN] База пуста — нечего визуализировать.")
        return

    type_counts: Dict[str, int] = {}
    for m in all_meta:
        dt = m.get('doc_type', 'unknown')
        type_counts[dt] = type_counts.get(dt, 0) + 1  # type: ignore

    all_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
    sorted_items = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [DOC_TYPES.get(k, k) for k, _ in sorted_items]
    sizes = [v for _, v in sorted_items]
    colors = all_colors[:len(labels)]

    fig, ax = plt.subplots(figsize=(8, 7))
    wedges, texts, autotexts = ax.pie(  # type: ignore
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
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'rag_plot1_kb_composition.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"График 1 сохранён: {path}")


def plot_chunk_length_distribution(chroma_dir: str, embeddings, save_dir: str):
    """
    График 2: Распределение длин чанков по типам документов.
    Boxplot + strip plot валидирует корректность выбора chunk_size.
    """

    db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    collection = db._collection.get()

    if not collection['documents']:
        print("[WARN] База пуста.")
        return

    docs_by_type: Dict[str, List[int]] = {}
    for doc_text, meta in zip(collection['documents'], collection['metadatas']):  # type: ignore
        clean = doc_text.replace('passage: ', '', 1)
        dt = meta.get('doc_type', 'unknown')
        docs_by_type.setdefault(dt, []).append(len(clean))

    fig, ax = plt.subplots(figsize=(11, 6))
    palette = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']

    types_sorted = list(docs_by_type.keys())
    data = [docs_by_type[dt] for dt in types_sorted]
    labels = [
        f"{DOC_TYPES.get(dt, dt)}\n(n={len(docs_by_type[dt])}, μ={np.mean(docs_by_type[dt]):.0f})"
        for dt in types_sorted
    ]

    bp = ax.boxplot(data, vert=False, patch_artist=True,
                    tick_labels=labels,
                    medianprops={'color': 'black', 'lw': 1.5})
    for patch, color in zip(bp['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, dt in enumerate(types_sorted, 1):
        y = np.random.normal(i, 0.04, size=len(docs_by_type[dt]))
        ax.scatter(docs_by_type[dt], y,
                   color=palette[(i - 1) % len(palette)],
                   edgecolor='black', s=22, zorder=3, alpha=0.8)

    for dt in types_sorted:
        cfg = CHUNK_CONFIG.get(dt)
        if cfg:
            ax.axvline(cfg['chunk_size'], color='red', lw=1.0, ls='--', alpha=0.4)

    ax.set_xlabel('Длина чанка (символов)', fontsize=11)
    ax.set_title('Распределение длин текстовых фрагментов (чанков)\n'
                 'по типам документов базы знаний',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.35, ls='--')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'rag_plot2_chunk_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"График 2 сохранён: {path}")


def plot_retrieval_quality(chroma_dir: str, embeddings,
                           embed_model_name: str,
                           test_queries: List[Dict],
                           save_dir: str):
    """
    График 3: Качество семантического поиска.
    Левая панель: L2-расстояния по тестовым запросам (boxplot).
    Правая панель: доля релевантных результатов (bar chart).
    """

    results_data = []
    for tq in test_queries:
        for rank, (doc, score) in enumerate(_search(chroma_dir, embeddings, tq['query']), 1):
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

    query_labels = df['query_label'].unique()
    data_per_query = [df[df['query_label'] == ql]['distance'].values for ql in query_labels]
    colors_box = ['#4C72B0', '#55A868', '#C44E52', '#8172B2',
                  '#CCB974', '#64B5CD'][:len(query_labels)]

    bp = axes[0].boxplot(data_per_query, tick_labels=query_labels, patch_artist=True,
                         medianprops={'color': 'red', 'lw': 2})
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

    df['relevant'] = df['distance'] <= RELEVANCE_THRESHOLD
    summary = df.groupby('query_label')['relevant'].mean() * 100
    summary = summary.reindex(query_labels)

    bars = axes[1].bar(summary.index, summary.values,
                       color=colors_box[:len(summary)], alpha=0.85, edgecolor='white')
    axes[1].axhline(100, color='green', ls='--', lw=1.2, 
                    alpha=0.5, label='100% релевантность')
    axes[1].set_ylim(0, 115)
    axes[1].set_ylabel('% релевантных результатов', fontsize=10)
    axes[1].set_title('Доля релевантных результатов\nпо каждому тестовому сценарию',
                      fontsize=11, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=25)
    axes[1].grid(axis='y', alpha=0.35, ls='--')
    for bar, val in zip(bars, summary.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle(f'Оценка качества семантического поиска RAG-системы\n'
                 f'Модель эмбеддингов: {embed_model_name}',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'rag_plot3_retrieval_quality.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"График 3 сохранён: {path}")


def plot_fault_coverage_heatmap(chroma_dir: str, embeddings, save_dir: str):
    """
    Матрица: строки = типы отказов, столбцы = типы документов.
    Значение = кол-во попаданий в топ-3 по 3 запросам на тип отказа.
    """

    fault_queries = {
        'Перегрев': [
            'температура подшипника выше 93 градусов',
            'перегрев двигателя нарастающий тренд',
            'действия оператора при превышении температуры',
        ],
        'Кавитация': [
            'кавитация насоса падение давления',
            'кавитационный износ рабочего колеса',
            'устранение кавитации регламент',
        ],
        'Электрика': [
            'скачок тока двигателя при нормальной вибрации',
            'электрическая неисправность привода насоса',
            'проверка обмоток двигателя ТОиР',
        ],
    }
    
    # 1. Подключение к локальной векторной базе
    db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    
    # Словарь для сбора статистики
    coverage_data = {fault: {} for fault in fault_queries.keys()}
    all_found_doc_types = set()

    # 2. Выполнение запросов и сбор метаданных (doc_type)
    for fault, queries in fault_queries.items():
        for q in queries:
            # Извлечение топ-3 фрагмента для каждого запроса
            docs = db.similarity_search(f"query: {q}", k=3)
            for doc in docs:
                # Если метаданных нет, 'Unknown'
                doc_type = doc.metadata.get('doc_type', 'Unknown')
                all_found_doc_types.add(doc_type)
                coverage_data[fault][doc_type] = coverage_data[fault].get(doc_type, 0) + 1

    # 3. Формирование DataFrame для тепловой карты
    df_heatmap = pd.DataFrame(coverage_data).T
    # Заполняем пустоты нулями и приводим к целым числам (счетчик документов)
    df_heatmap = df_heatmap.fillna(0).astype(int)

    # 4. Визуализация (светлая академическая тема)
    LIGHT_BG = '#FFFFFF'
    TEXT_CLR = '#222222'

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    # Отрисовка тепловой карты
    sns.heatmap(
        df_heatmap,
        annot=True,
        fmt='d',
        cmap='Blues',
        linewidths=1,
        linecolor='#DDDDDD',
        cbar_kws={'label': 'Количество извлеченных чанков в Топ-3'},
        ax=ax,
        annot_kws={'size': 12, 'weight': 'bold'}
    )

    # Стилизация текста и заголовков
    ax.set_title('Матрица покрытия базы знаний (Fault Type × Doc Type)\n'
                 'Сбалансированность извлечения релевантного контекста (RAG)',
                 color=TEXT_CLR, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Категория отказа (Симптомы)', color=TEXT_CLR, 
                  fontsize=12, fontweight='bold')
    ax.set_xlabel('Тип нормативного документа (doc_type)', color=TEXT_CLR, 
                  fontsize=12, fontweight='bold')

    ax.tick_params(colors=TEXT_CLR, labelsize=11)

    plt.tight_layout()

    # 5. Сохранение графика
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'rag_plot4_fault_coverage_heatmap.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    print(f"График 4 сохранён: {plot_path}")
    plt.close(fig)


# Точка входа (все три графика за один вызов)

def plot_all_rag(chroma_dir: str, embeddings,
                 embed_model_name: str,
                 test_queries: List[Dict],
                 save_dir: str):
    """
    Строит все три графика RAG-визуализации.
    Аналог analyze_fault_recall в fault_recall_analysis.py.
    """
    print("\nГенерация графиков RAG-системы...")
    plot_knowledge_base_stats(chroma_dir, embeddings, save_dir)
    plot_chunk_length_distribution(chroma_dir, embeddings, save_dir)
    plot_retrieval_quality(chroma_dir, embeddings, embed_model_name, test_queries, save_dir)
    plot_fault_coverage_heatmap(chroma_dir, embeddings, save_dir)