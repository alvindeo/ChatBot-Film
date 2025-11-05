# Sistem Information Retrieval - Chatbot Film

Sistem chatbot untuk pencarian informasi film menggunakan metode Boolean IR dan VSM (Vector Space Model).

## Struktur Proyek

```
├── app/
│   └── chatbot.py           # Entry point untuk menjalankan chatbot
├── dataset/                 # Dataset film dalam format txt
├── src/
│   ├── search_engine.py    # Implementasi Boolean IR dan VSM
│   └── evaluasi.py         # Script untuk evaluasi sistem
└── requirements.txt        # Dependensi Python yang dibutuhkan
```

## Cara Instalasi

1. Install Python 3.x
2. Install dependensi:

```bash
pip install -r requirements.txt
```

## Cara Menjalankan Sistem

### 1. Menjalankan Chatbot

Untuk menggunakan chatbot:

```bash
python app/chatbot.py
```

Buka browser dan akses URL yang muncul (biasanya http://127.0.0.1:7860)

### 2. Menjalankan Evaluasi

Untuk melihat performa sistem:

```bash
python src/evaluasi.py
```

Hasil evaluasi akan ditampilkan di terminal dan visualisasi akan disimpan dalam file `evaluasi_visual.png`

## Fitur Sistem

1. **Pencarian Informasi Film**

   - Query dalam bahasa natural
   - Mendukung pertanyaan tentang film, sutradara, aktor, dll.
   - Menampilkan hasil dengan skor relevansi

2. **Metode Pencarian**

   - Boolean IR: Pencarian berdasarkan kecocokan kata kunci
   - VSM: Pencarian dengan pembobotan TF-IDF

3. **Evaluasi Sistem**
   - Precision, Recall, F1-Score
   - Mean Average Precision (MAP)
   - Normalized Discounted Cumulative Gain (nDCG)
   - Visualisasi hasil evaluasi

## Dataset

Dataset berisi informasi film dalam berbagai kategori:

- Film terbaru
- Sutradara
- Aktor/Aktris
- Genre film
- Rating dan fakta menarik
- Dan kategori lainnya

## Contoh Penggunaan

Setelah menjalankan chatbot, Anda dapat menanyakan berbagai hal seperti:

- "Film aksi terbaik 2025?"
- "Siapa sutradara film Titanic?"
- "Film Indonesia dengan rating tinggi?"
- "Film animasi terbaru yang viral?"
- "Film apa yang menang Oscar 2023?"
