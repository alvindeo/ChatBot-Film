import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import sys
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from math import log2
from tabulate import tabulate

# Import komponen yang diperlukan dari search_engine
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.search_engine import RAGChatbot, SearchEngine

# Baca dataset
questions, answers, file_names = [], [], []
for filepath in glob.glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset/*.txt')):
    file_name = os.path.basename(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if '->' in line:
                q, a = line.split('->')
                questions.append(q.strip("- ").lower().strip())
                answers.append(a.strip())
                file_names.append(file_name)

# Buat instance chatbot untuk evaluasi
chatbot = RAGChatbot(questions, answers, file_names)

# Daftar query untuk evaluasi dan groundtruth relevansi dokumen (doc index)
queries = [
    "Film aksi terbaik 2025",
    "Siapa sutradara film Titanic?",
    "Film Indonesia rating tinggi",
    "Film animasi terbaru viral",
    "Film apa yang menang Oscar 2023?"
]

# Tentukan groundtruth relevansi (gold label, pakai index dari jawaban di dataset)
groundtruth_relevant_docs = [
    [52, 50, 48],      # Film aksi terbaik 2025 (dokumen yang berisi info film aksi 2025)
    [121, 23, 22],     # Siapa sutradara Titanic (dokumen tentang James Cameron/Titanic)
    [61, 58, 60],      # Film Indonesia rating tinggi
    [50, 104, 49],     # Film animasi trending/viral
    [84, 85, 83]       # Film peraih Oscar 2023
]

# Top-k retrieval
k = 3
all_precisions, all_recalls, all_f1s = [], [], []

for i, q in enumerate(queries):
    results = chatbot.engine.search(q, model='vsm', k=k)
    retrieved = [idx for idx, score in results]
    relevant = set(groundtruth_relevant_docs[i])

    # Binary relevance label
    y_true = [1 if doc in relevant else 0 for doc in retrieved]
    y_pred = [1]*len(retrieved)  # Retrieval: selalu prediksi 1 di posisi hasil

    if len(y_true) > 0 and sum(y_true) > 0:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        precision = recall = f1 = 0.0

    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1s.append(f1)
    print(f"Query: {q}")
    print(f"  Retrieved doc: {retrieved}")
    print(f"  Groundtruth relevant: {groundtruth_relevant_docs[i]}")
    print(f"  Precision@{k}: {precision:.2f}, Recall@{k}: {recall:.2f}, F1@{k}: {f1:.2f}\n")

print("Rata-rata evaluasi semua query:")
print(f"  Precision@{k}: {np.mean(all_precisions):.2f}")
print(f"  Recall@{k}: {np.mean(all_recalls):.2f}")
print(f"  F1@{k}: {np.mean(all_f1s):.2f}")


print("\n--- Detail Hasil MAP@K dan nDCG@K ---")
def average_precision(relevant, retrieved):
    hits = 0
    sum_precisions = 0
    for n, doc in enumerate(retrieved, 1):
        if doc in relevant:
            hits += 1
            sum_precisions += hits / n
    return sum_precisions / max(1, len(relevant))

mapk = []
for i, q in enumerate(queries):
    results = chatbot.engine.search(q, model='vsm', k=k)
    retrieved = [idx for idx, score in results]
    relevant = set(groundtruth_relevant_docs[i])
    ap = average_precision(relevant, retrieved)
    mapk.append(ap)
    print(f"Query: {q} AP@{k}: {ap:.2f}")
print(f"MAP@{k}: {np.mean(mapk):.2f}")


# Fungsi untuk menghitung nDCG
def ndcg_at_k(relevant_docs, retrieved_docs, k):
    dcg = 0
    idcg = 0
    
    # Hitung DCG
    for i, doc_id in enumerate(retrieved_docs[:k]):
        if doc_id in relevant_docs:
            rel = 1
            dcg += rel / log2(i + 2)  # i + 2 karena i mulai dari 0
            
    # Hitung IDCG
    for i in range(min(k, len(relevant_docs))):
        idcg += 1 / log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0

# Visualisasi hasil evaluasi
plt.figure(figsize=(15, 10))

# Plot 1: Metrik per Query
plt.subplot(2, 2, 1)
x = range(len(queries))
width = 0.25
plt.bar([i - width for i in x], all_precisions, width, label='Precision')
plt.bar(x, all_recalls, width, label='Recall')
plt.bar([i + width for i in x], all_f1s, width, label='F1')
plt.xlabel('Query Index')
plt.ylabel('Score')
plt.title(f'Evaluasi Metrik per Query @{k}')
plt.legend()
plt.xticks(x)

# Plot 2: Rata-rata Metrik
plt.subplot(2, 2, 2)
avg_metrics = [np.mean(all_precisions), np.mean(all_recalls), np.mean(all_f1s)]
plt.bar(['Precision', 'Recall', 'F1'], avg_metrics)
plt.ylabel('Score')
plt.title('Rata-rata Metrik')

# Plot 3: nDCG per Query
ndcg_scores = []
for i, q in enumerate(queries):
    results = chatbot.engine.search(q, model='vsm', k=k)
    retrieved = [idx for idx, score in results]
    ndcg = ndcg_at_k(set(groundtruth_relevant_docs[i]), retrieved, k)
    ndcg_scores.append(ndcg)

plt.subplot(2, 2, 3)
plt.bar(range(len(queries)), ndcg_scores)
plt.xlabel('Query Index')
plt.ylabel('nDCG Score')
plt.title(f'nDCG@{k} per Query')

# Plot 4: Heatmap Relevansi
plt.subplot(2, 2, 4)
relevance_matrix = np.zeros((len(queries), k))
for i, q in enumerate(queries):
    results = chatbot.engine.search(q, model='vsm', k=k)
    retrieved = [idx for idx, score in results]
    for j, doc_id in enumerate(retrieved):
        if doc_id in groundtruth_relevant_docs[i]:
            relevance_matrix[i, j] = 1

sns.heatmap(relevance_matrix, annot=True, cmap='YlOrRd', 
            xticklabels=[f'Rank {i+1}' for i in range(k)],
            yticklabels=[f'Q{i+1}' for i in range(len(queries))])
plt.title('Relevansi Hasil (1=Relevan, 0=Tidak Relevan)')

plt.tight_layout()
plt.savefig('evaluasi_visual.png')
plt.close()

# Tampilkan hasil evaluasi dalam bentuk tabel
print("\n=== Evaluasi Detail Per Query ===")
eval_data = []
for i, q in enumerate(queries):
    results = chatbot.engine.search(q, model='vsm', k=k)
    retrieved = [idx for idx, score in results]
    ndcg = ndcg_at_k(set(groundtruth_relevant_docs[i]), retrieved, k)
    # Hitung AP untuk query ini (jika belum dihitung sebelumnya)
    ap = average_precision(set(groundtruth_relevant_docs[i]), retrieved)

    eval_data.append([
        q,
        f"{all_precisions[i]:.2f}",
        f"{all_recalls[i]:.2f}",
        f"{all_f1s[i]:.2f}",
        f"{ap:.4f}",
        f"{ndcg:.4f}"
    ])

print(tabulate(eval_data, 
              headers=['Query', 'Precision', 'Recall', 'F1', 'AP', 'nDCG'],
              tablefmt='grid'))

# Cetak MAP@k dan rata-rata nDCG@k dengan format jelas
if 'mapk' in globals():
    print(f"\nMAP@{k}: {np.mean(mapk):.4f}")
else:
    # jika mapk belum dihitung, hitung sekarang
    ap_list = [average_precision(set(groundtruth_relevant_docs[i]), [idx for idx, _ in chatbot.engine.search(q, model='vsm', k=k)]) for i, q in enumerate(queries)]
    print(f"\nMAP@{k}: {np.mean(ap_list):.4f}")

print(f"Mean nDCG@{k}: {np.mean(ndcg_scores):.4f}")

print("\n=== Rata-rata Metrik ===")
avg_data = [
    ['Rata-rata', 
     f"{np.mean(all_precisions):.2f}",
     f"{np.mean(all_recalls):.2f}",
     f"{np.mean(all_f1s):.2f}",
     f"{np.mean(ndcg_scores):.2f}"
    ]
]
print(tabulate(avg_data, 
              headers=['Metrik', 'Precision', 'Recall', 'F1', 'nDCG'],
              tablefmt='grid'))

print("\n=== Detail Hasil vs Groundtruth ===")
for i, q in enumerate(queries):
    results = chatbot.engine.search(q, model='vsm', k=k)
    print(f"\nQuery: {q}")
    print("Hasil Pencarian:")
    for idx, score in results:
        print(f"  • {answers[idx][:100]}... (score: {score:.2f})")
    print("Groundtruth:")
    for j in groundtruth_relevant_docs[i]:
        print(f"  • {answers[j][:100]}...")

print("\nHasil visualisasi telah disimpan dalam file 'evaluasi_visual.png'")
