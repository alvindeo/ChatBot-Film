# 🎬 Chatbot Film - Enhanced Edition

Chatbot film berbasis Information Retrieval dengan fitur **Intent Classification**, **Multi-LLM Integration**, dan **Movie Dataset** dari sumber publik.

## ✨ Fitur Utama

### 1. 🎯 Intent Classification (KNN & Naive Bayes)
- Mendeteksi maksud user dari query (rekomendasi, info film, sutradara, aktor, dll)
- Mendukung 10 kategori intent
- Menggunakan algoritma K-Nearest Neighbors (KNN) dan Naive Bayes
- Auto-training dengan dataset yang sudah disiapkan

### 2. 🤖 Multi-LLM Integration
- **Google Gemini**: AI model dari Google (GRATIS)
- **OpenAI GPT**: GPT-3.5/GPT-4 dari OpenAI
- **Groq**: Ultra-fast inference dengan Llama 3 (GRATIS & CEPAT!)
- Fallback mechanism jika LLM tidak tersedia
- Response enhancement untuk jawaban yang lebih natural

### 3. 📽️ Movie Dataset Integration
- Built-in dataset 15+ film populer (Inception, The Dark Knight, Parasite, dll)
- Support TMDb API untuk dataset lebih lengkap (100+ film)
- Auto-conversion ke format Q&A
- Informasi lengkap: sutradara, aktor, genre, rating, sinopsis

### 4. 🔍 Advanced Search Engine
- Vector Space Model (TF-IDF + Cosine Similarity)
- Boolean Information Retrieval
- Top-K ranking results
- Score threshold validation

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup API Keys (Optional)

Copy `.env.example` ke `.env`:

```bash
copy .env.example .env
```

Edit `.env` dan tambahkan API keys Anda:

```bash
# Pilih salah satu (atau semua):

# Groq (RECOMMENDED - Gratis & Cepat!)
GROQ_API_KEY=your_groq_api_key_here
DEFAULT_LLM=groq

# Atau Gemini (Gratis dari Google)
GEMINI_API_KEY=your_gemini_api_key_here
DEFAULT_LLM=gemini

# Atau OpenAI (Berbayar)
OPENAI_API_KEY=your_openai_api_key_here
DEFAULT_LLM=openai

# TMDb untuk dataset lebih lengkap (Optional)
TMDB_API_KEY=your_tmdb_api_key_here
```

**Cara Mendapatkan API Keys (GRATIS):**

1. **Groq** (RECOMMENDED): https://console.groq.com/keys
   - Daftar dengan email
   - Langsung dapat API key
   - Ultra-fast & unlimited (untuk development)

2. **Google Gemini**: https://makersuite.google.com/app/apikey
   - Login dengan Google account
   - Klik "Create API Key"
   - Gratis dengan quota harian

3. **TMDb**: https://www.themoviedb.org/settings/api
   - Daftar akun TMDb
   - Request API key (gratis)
   - Dapat 1000+ requests per hari

### 3. Run Chatbot

```bash
python app/chatbot.py
```

Atau langsung dari src:

```bash
python src/search_engine.py
```

Buka browser di: `http://localhost:7860`

## 📁 Struktur Project

```
UAS/
├── app/
│   └── chatbot.py              # Entry point aplikasi
├── src/
│   ├── search_engine.py        # Main search engine & UI
│   ├── intent_classifier.py    # Intent classification (KNN/NB)
│   ├── llm_integration.py      # Multi-LLM integration
│   ├── movie_dataset.py        # Movie dataset loader
│   ├── preprocess.py           # Text preprocessing
│   ├── vsm_ir.py              # Vector Space Model
│   └── boolean_ir.py          # Boolean IR
├── dataset/
│   ├── *.txt                   # Dataset film existing
│   └── cache/                  # Cached movie datasets
├── models/                     # Trained ML models
│   ├── intent_knn.pkl         # KNN classifier
│   └── intent_nb.pkl          # Naive Bayes classifier
├── .env.example               # Template environment variables
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎮 Cara Menggunakan

### Contoh Query:

1. **Rekomendasi Film:**
   - "rekomendasikan film action terbaik"
   - "film bagus untuk ditonton"
   - "film terbaik 2024"

2. **Info Sutradara:**
   - "siapa sutradara film Inception?"
   - "sutradara The Dark Knight"

3. **Info Aktor:**
   - "siapa pemeran film Titanic?"
   - "aktor di film Interstellar"

4. **Info Film:**
   - "ceritakan tentang film Parasite"
   - "sinopsis film The Matrix"

5. **Genre:**
   - "film horor terbaik"
   - "genre film Inception"

## 🧪 Testing Modules

### Test Intent Classifier:

```bash
python src/intent_classifier.py
```

### Test LLM Integration:

```bash
python src/llm_integration.py
```

### Test Movie Dataset:

```bash
python src/movie_dataset.py
```

## 📊 Fitur Intent Classification

Sistem dapat mendeteksi 10 kategori intent:

1. **rekomendasi** - User ingin rekomendasi film
2. **info_film** - User ingin info detail film
3. **sutradara** - User bertanya tentang sutradara
4. **aktor** - User bertanya tentang aktor/aktris
5. **genre** - User bertanya tentang genre
6. **tahun** - User bertanya tentang tahun rilis
7. **penghargaan** - User bertanya tentang awards
8. **trivia** - User ingin fakta menarik
9. **soundtrack** - User bertanya tentang musik
10. **teknologi** - User bertanya tentang efek visual

## 🤖 LLM Enhancement

Ketika LLM aktif, sistem akan:
1. Melakukan pencarian normal dengan TF-IDF
2. Mendeteksi intent user
3. Mengirim hasil ke LLM untuk enhancement
4. Mengembalikan response yang lebih natural dan informatif

**Keuntungan:**
- Jawaban lebih natural dan conversational
- Konteks lebih baik
- Bisa menjelaskan dengan lebih detail
- Tetap grounded pada data (tidak halusinasi)

## 📈 Performance

- **Search Speed**: < 100ms untuk 500+ dokumen
- **Intent Classification**: ~50ms per query
- **LLM Enhancement**: 1-3 detik (tergantung provider)
- **Dataset Size**: 500+ Q&A pairs (bisa lebih dengan TMDb)

## 🛠️ Troubleshooting

### Error: Module not found

```bash
pip install -r requirements.txt
```

### Error: API Key invalid

Pastikan API key sudah benar di file `.env` dan provider sudah dipilih:

```bash
DEFAULT_LLM=groq  # atau gemini, openai
```

### LLM tidak aktif

Sistem tetap berjalan tanpa LLM! Hanya response tidak akan di-enhance oleh AI.

### Intent Classifier error

Model akan auto-train saat pertama kali dijalankan. Tunggu beberapa detik.

## 📝 TODO / Future Improvements

- [ ] Add more movie datasets (IMDb, Rotten Tomatoes)
- [ ] Implement caching untuk LLM responses
- [ ] Add conversation history
- [ ] Multi-language support
- [ ] Add image generation untuk poster film
- [ ] Implement user feedback system
- [ ] Add A/B testing untuk algoritma

## 🤝 Contributing

Silakan fork dan submit pull request untuk improvement!

## 📄 License

MIT License - Feel free to use for academic purposes

## 👨‍💻 Author

**Alvin Deo**
- GitHub: [@alvindeo](https://github.com/alvindeo)
- Project: ChatBot Film - Information Retrieval

---

**Dibuat untuk UAS Information Retrieval (STKI) - UDINUS Semester 5**
