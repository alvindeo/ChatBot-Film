# ChatBot Film - UAS Information Retrieval

Proyek chatbot film dengan fitur klasifikasi intent, analisis sentimen, dan integrasi multiple LLM APIs.

## 📋 Daftar Isi
- [Fitur](#fitur)
- [Teknologi](#teknologi)
- [Instalasi](#instalasi)
- [Konfigurasi](#konfigurasi)
- [Cara Menjalankan](#cara-menjalankan)
- [Struktur Proyek](#struktur-proyek)

## ✨ Fitur

- 🤖 **Intent Classification**: Menggunakan KNN dan Naive Bayes
- 💬 **Multiple LLM Integration**: Google Gemini, OpenAI GPT, Groq
- 🎬 **Movie Dataset**: Dataset film publik untuk pencarian
- 📊 **Sentiment Analysis**: Analisis sentimen review film
- 🎨 **Gradio UI**: Interface web yang user-friendly

## 🛠️ Teknologi

- Python 3.8+
- NLTK untuk NLP
- Scikit-learn untuk Machine Learning
- Gradio untuk UI
- Google Gemini, OpenAI, Groq APIs

## 📦 Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/alvindeo/ChatBot-Film.git
cd UAS_ALVIN
```

### 2. Buat Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## ⚙️ Konfigurasi

### 1. Copy file `.env.example` menjadi `.env`

```bash
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac
```

### 2. Edit file `.env` dan isi API keys Anda:

```env
# Google Gemini API
GOOGLE_API_KEY=your_google_api_key_here

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Groq API
GROQ_API_KEY=your_groq_api_key_here
```

### Cara Mendapatkan API Keys:

- **Google Gemini**: https://makersuite.google.com/app/apikey
- **OpenAI**: https://platform.openai.com/api-keys
- **Groq**: https://console.groq.com/keys

## 🚀 Cara Menjalankan

### Menjalankan Chatbot Utama

```bash
python app/chatbot.py
```

### Menjalankan Demo Lainnya

```bash
# Klasifikasi Intent
python classify.py

# Analisis Sentimen
python sentiment_demo.py

# Search Plus
python search_plus.py
```

## 📁 Struktur Proyek

```
UAS_ALVIN/
├── .venv/                  # Virtual environment (TIDAK DIUPLOAD)
├── app/                    # Aplikasi utama
│   └── chatbot.py
├── src/                    # Source code
│   ├── intent_classifier.py
│   ├── sentiment_analyzer.py
│   └── ...
├── dataset/                # Dataset film
├── models/                 # Model ML yang sudah ditraining
├── notebooks/              # Jupyter notebooks
├── reports/                # Laporan hasil
├── .env                    # Environment variables (TIDAK DIUPLOAD)
├── .env.example            # Template untuk .env
├── .gitignore              # File yang diabaikan Git
├── requirements.txt        # Daftar dependencies
└── README.md               # Dokumentasi ini
```

## 📝 Catatan Penting

### ⚠️ File yang TIDAK Perlu Diupload ke GitHub:

1. **`.venv/`** - Virtual environment (600MB+)
   - Orang lain bisa buat sendiri dengan `python -m venv .venv`
   
2. **`nltk_data/`** - NLTK data (bisa ratusan MB)
   - Bisa didownload ulang dengan perintah NLTK
   
3. **`.env`** - File berisi API keys (RAHASIA!)
   - Gunakan `.env.example` sebagai template

### ✅ File yang Perlu Diupload:

1. **`requirements.txt`** - Daftar library yang dibutuhkan
2. **Source code** (`.py` files)
3. **Dataset** (jika tidak terlalu besar, < 10MB)
4. **`.gitignore`** - Agar file besar tidak terupload
5. **`README.md`** - Dokumentasi

## 🎓 Untuk Pengumpulan Tugas

Jika Anda perlu mengumpulkan proyek ini:

### Opsi 1: GitHub (Recommended)
```bash
git add .
git commit -m "UAS Information Retrieval - ChatBot Film"
git push origin main
```
Kemudian share link GitHub repository.

### Opsi 2: ZIP File (Tanpa .venv)
1. Pastikan `.gitignore` sudah ada
2. Compress folder TANPA `.venv/`:
   ```bash
   # Exclude .venv saat membuat ZIP
   # Ukuran akan < 10MB
   ```
3. Upload ZIP file

### Opsi 3: Google Drive/OneDrive
Upload ke cloud storage dan share link.

## 👥 Kontributor

- Alvin Deo

## 📄 Lisensi

MIT License

---

**Dibuat untuk UAS Information Retrieval - UDINUS Semester 5**
