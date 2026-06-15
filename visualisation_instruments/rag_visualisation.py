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


def plot_fault_coverage_heatmap(kb, save_dir: str):
    """
    Матрица: строки = типы отказов, столбцы = типы документов (doc_type).
    Значение = число чанков данного doc_type, извлечённых РЕАЛЬНЫМ пайплайном
    (справочный контекст + действия оператора + работы ТОиР + график ТО) для
    репрезентативного входа по каждому типу отказа.
 
    В отличие от прежней версии (3 обобщённых запроса), здесь heatmap отражает
    фактическое покрытие всех четырёх частей ответа агента.
    """
    # Репрезентативные входы по типам отказа (сигнатуры из tm_regulation.md).
    FAULT_PROBES = {
        'Перегрев': {
            'fault': 'overheat',
            'symptoms': {
                'critical_probability': 95, 'inferred_fault': 'overheat',
                'top_symptoms': [
                    {'sensor': 'temperature', 'value': 94.5, 'shap_weight':  0.42},
                    {'sensor': 'current',     'value': 95.0, 'shap_weight':  0.30},
                ],
            },
        },
        'Кавитация': {
            'fault': 'cavitation',
            'symptoms': {
                'critical_probability': 100, 'inferred_fault': 'cavitation',
                'top_symptoms': [
                    {'sensor': 'vibration', 'value': 9.11, 'shap_weight':  0.50},
                    {'sensor': 'pressure',  'value': 0.71, 'shap_weight': -0.28},
                ],
            },
        },
        'Электрика': {
            'fault': 'electrical',
            'symptoms': {
                'critical_probability': 90, 'inferred_fault': 'electrical',
                'top_symptoms': [
                    {'sensor': 'current',   'value': 93.0, 'shap_weight':  0.48},
                    {'sensor': 'vibration', 'value': 2.41, 'shap_weight':  0.10},
                ],
            },
        },
    }
    # График ТО извлекается lookup'ом по pump_id и не зависит от типа отказа —
    # берём представительный агрегат с ближайшим предиктивным риском.
    PROBE_PUMP_ID = 'MNHV_005'
 
    # Фиксированный порядок столбцов: gost и schedule присутствуют всегда.
    _PREFERRED = ['manual', 'gost', 'diagnostics', 'sop', 'schedule']
    doc_type_order = ([d for d in _PREFERRED if d in DOC_TYPES] +
                      [d for d in DOC_TYPES if d not in _PREFERRED])
 
    coverage = {fault: {dt: 0 for dt in doc_type_order} for fault in FAULT_PROBES}
 
    for fault_label, probe in FAULT_PROBES.items():
        f = probe['fault']
        # Реальные 4 канала пайплайна (как в ai_agent_benchmark.build_scenarios).
        retrieved = []
        retrieved += kb.search_by_symptoms(probe['symptoms'], k=4)        # справочный
        retrieved += kb.search_operator_actions(f, k=2)                   # предписание
        retrieved += kb.search_repair_works(f, k=2)                       # ТОиР
        retrieved += kb.search_maintenance_schedule(PROBE_PUMP_ID, k=2)   # график
 
        for doc, _score in retrieved:
            dt = doc.metadata.get('doc_type', 'unknown')
            coverage[fault_label][dt] = coverage[fault_label].get(dt, 0) + 1
 
    df_heatmap = (pd.DataFrame(coverage).T
                  .reindex(columns=doc_type_order)
                  .fillna(0).astype(int))
 
    # Визуализация (светлая академическая тема, как в исходнике).
    LIGHT_BG, TEXT_CLR = '#FFFFFF', '#222222'
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)
 
    sns.heatmap(
        df_heatmap, annot=True, fmt='d', cmap='Blues',
        linewidths=1, linecolor='#DDDDDD',
        cbar_kws={'label': 'Кол-во чанков, извлечённых пайплайном'},
        ax=ax, annot_kws={'size': 12, 'weight': 'bold'},
    )
    ax.set_title('Матрица покрытия базы знаний (Fault Type × Doc Type)\n'
                 'Реальный 4-канальный пайплайн: контекст + предписание + ТОиР + график',
                 color=TEXT_CLR, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Категория отказа', color=TEXT_CLR, fontsize=12, fontweight='bold')
    ax.set_xlabel('Тип нормативного документа (doc_type)',
                  color=TEXT_CLR, fontsize=12, fontweight='bold')
    ax.tick_params(colors=TEXT_CLR, labelsize=11)
    plt.yticks(rotation=0)
    plt.tight_layout()
 
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'rag_plot4_fault_coverage_heatmap.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"График 4 сохранён: {plot_path}")

def plot_fault_section_sourcing(kb, save_dir: str):
    """
    Строки = типы отказов; столбцы = 4 раздела ответа агента.
    В ячейке — какой(ие) doc_type извлечены реальным каналом пайплайна и сколько
    чанков. Цвет — число чанков. Демонстрирует разделение источников по разделам.
    """

    FAULT_PROBES = {
        'Перегрев': {
            'fault': 'overheat',
            'symptoms': {'critical_probability': 95, 'inferred_fault': 'overheat',
                         'top_symptoms': [
                             {'sensor': 'temperature', 'value': 94.5, 'shap_weight': 0.42},
                             {'sensor': 'current',     'value': 95.0, 'shap_weight': 0.30}]},
        },
        'Кавитация': {
            'fault': 'cavitation',
            'symptoms': {'critical_probability': 100, 'inferred_fault': 'cavitation',
                         'top_symptoms': [
                             {'sensor': 'vibration', 'value': 9.11, 'shap_weight':  0.50},
                             {'sensor': 'pressure',  'value': 0.71, 'shap_weight': -0.28}]},
        },
        'Электрика': {
            'fault': 'electrical',
            'symptoms': {'critical_probability': 90, 'inferred_fault': 'electrical',
                         'top_symptoms': [
                             {'sensor': 'current',   'value': 93.0, 'shap_weight': 0.48},
                             {'sensor': 'vibration', 'value': 2.41, 'shap_weight': 0.10}]},
        },
    }
    PROBE_PUMP_ID = 'MNHV_005'   # для канала «График ТО» (lookup по pump_id)
    SECTIONS = ['Справочный\nконтекст', 'Действия\nоператора',
                'Работы\nТОиР', 'График\nТО']
 
    counts = pd.DataFrame(0, index=list(FAULT_PROBES), columns=SECTIONS, dtype=int)
    annot = pd.DataFrame('', index=list(FAULT_PROBES), columns=SECTIONS, dtype=object)
 
    for fault, probe in FAULT_PROBES.items():
        f = probe['fault']
        channels = {
            'Справочный\nконтекст': kb.search_by_symptoms(probe['symptoms'], k=4),
            'Действия\nоператора':  kb.search_operator_actions(f, k=2),
            'Работы\nТОиР':         kb.search_repair_works(f, k=2),
            'График\nТО':           kb.search_maintenance_schedule(PROBE_PUMP_ID, k=2),
        }
        for sec, res in channels.items():
            dts = [d.metadata.get('doc_type', 'unknown') for d, _ in res]
            counts.loc[fault, sec] = len(dts)
            uniq = sorted(set(dts))
            # Каждый doc_type — на своей строке (чтобы подпись не выходила за ячейку).
            annot.loc[fault, sec] = ('\n'.join(uniq) + f"\n({len(dts)} чанк.)"
                                     if dts else '—')
 
    LIGHT_BG, TEXT_CLR = '#FFFFFF', '#222222'
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)
    sns.heatmap(counts, annot=annot.values, fmt='', cmap='Blues', vmin=0,
                linewidths=1, linecolor='#DDDDDD',
                cbar_kws={'label': 'Число извлечённых чанков'},
                annot_kws={'size': 10, 'weight': 'bold'}, ax=ax)
    ax.set_title('Источники по разделам предписания (Fault Type x Раздел ответа)\n'
                 'Разделение источников: предписание/ТОиР — регламент, график — расписание',
                 color=TEXT_CLR, fontsize=13, fontweight='bold', pad=14)
    ax.set_ylabel('Категория отказа', color=TEXT_CLR, fontsize=12, fontweight='bold')
    ax.set_xlabel('Раздел ответа агента', color=TEXT_CLR, fontsize=12, fontweight='bold')
    ax.tick_params(colors=TEXT_CLR, labelsize=11)
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    p = os.path.join(save_dir, 'rag_plot5_section_sourcing.png')
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close(fig)
    print(f"График 5 сохранён: {p}")


# Точка входа (все три графика за один вызов)

def plot_all_rag(kb, test_queries, save_dir: str):
    """Строит все пять графиков RAG-визуализации. Принимает kb (KnowledgeBaseManager)."""
    print("\nГенерация графиков RAG-системы...")
    plot_knowledge_base_stats(kb.chroma_dir, kb.embeddings, save_dir)
    plot_chunk_length_distribution(kb.chroma_dir, kb.embeddings, save_dir)
    plot_retrieval_quality(kb.chroma_dir, kb.embeddings, kb.EMBED_MODEL,
                           test_queries, save_dir)
    plot_fault_coverage_heatmap(kb, save_dir)
    plot_fault_section_sourcing(kb, save_dir)     