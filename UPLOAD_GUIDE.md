# 📤 Panduan Upload Proyek (Maks 10MB)

## ❌ MASALAH: Proyek Terlalu Besar (600MB+)

Folder `.venv/site-packages/` berisi semua library Python yang sangat besar.

## ✅ SOLUSI: Jangan Upload `.venv`!

### Langkah-langkah:

## 1️⃣ Pastikan `.gitignore` Sudah Ada

File `.gitignore` sudah dibuat dan akan mengabaikan:
- ✅ `.venv/` (600MB+)
- ✅ `__pycache__/`
- ✅ `.env` (berisi API keys rahasia)
- ✅ `nltk_data/`
- ✅ `.gradio/`

## 2️⃣ Cek Status Git

```bash
git status
```

Pastikan `.venv/` TIDAK muncul di daftar file yang akan diupload.

## 3️⃣ Upload ke GitHub

```bash
# Add semua file (kecuali yang di .gitignore)
git add .

# Commit
git commit -m "UAS Information Retrieval - ChatBot Film"

# Push ke GitHub
git push origin main
```

## 4️⃣ Ukuran Proyek Setelah Exclude `.venv`

Seharusnya hanya **< 5MB** yang berisi:
- ✅ Source code (`.py` files)
- ✅ Dataset (jika kecil)
- ✅ `requirements.txt`
- ✅ `README.md`
- ✅ `.env.example`
- ✅ `.gitignore`

## 📊 Perbandingan Ukuran

| Item | Dengan `.venv` | Tanpa `.venv` |
|------|----------------|---------------|
| `.venv/` | 600 MB | 0 MB ❌ |
| Source code | 1 MB | 1 MB ✅ |
| Dataset | 2 MB | 2 MB ✅ |
| Models | 1 MB | 1 MB ✅ |
| **TOTAL** | **604 MB** ❌ | **4 MB** ✅ |

## 🎯 Cara Orang Lain Menjalankan Proyek Anda

Mereka cukup:

```bash
# 1. Clone repository
git clone https://github.com/alvindeo/ChatBot-Film.git
cd UAS_ALVIN

# 2. Buat virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies dari requirements.txt
pip install -r requirements.txt

# 4. Copy dan edit .env
copy .env.example .env
# Edit .env dengan API keys mereka

# 5. Jalankan!
python app/chatbot.py
```

## 🚨 PENTING: Jangan Upload File Ini

1. **`.venv/`** - Terlalu besar (600MB+)
2. **`.env`** - Berisi API keys rahasia
3. **`__pycache__/`** - File cache Python
4. **`nltk_data/`** - Bisa didownload ulang
5. **`.gradio/`** - File temporary

## ✅ File yang HARUS Diupload

1. **`requirements.txt`** - Daftar library
2. **`.env.example`** - Template untuk API keys
3. **`.gitignore`** - Agar file besar tidak terupload
4. **`README.md`** - Dokumentasi
5. **Source code** - Semua file `.py`
6. **Dataset** - Jika tidak terlalu besar

## 🔍 Verifikasi Sebelum Upload

```bash
# Cek ukuran folder (tanpa .venv)
# Windows PowerShell:
Get-ChildItem -Recurse -File | Where-Object { $_.FullName -notmatch '\\.venv' } | Measure-Object -Property Length -Sum

# Atau cek di File Explorer:
# Klik kanan folder > Properties (pastikan .venv tidak termasuk)
```

## 📦 Alternatif: Upload ZIP Manual

Jika tidak menggunakan Git:

1. **Hapus folder `.venv` sementara** (atau jangan include saat ZIP)
2. **Compress folder** menjadi ZIP
3. **Upload ZIP** (seharusnya < 10MB)
4. **Restore `.venv`** di komputer Anda (jika dihapus)

```bash
# Backup .venv (optional)
move .venv .venv_backup

# Buat ZIP (tanpa .venv)
# Gunakan WinRAR/7-Zip dan exclude .venv

# Restore .venv
move .venv_backup .venv
```

## 💡 Tips Tambahan

### Jika Dataset Terlalu Besar
Jika file dataset > 5MB, upload ke Google Drive dan tambahkan link di README:

```markdown
## Dataset
Dataset terlalu besar untuk GitHub. Download dari:
https://drive.google.com/file/d/xxxxx
```

### Jika Model ML Terlalu Besar
Sama seperti dataset, upload model ke cloud storage.

---

**Kesimpulan**: Dengan `.gitignore` yang benar, proyek Anda akan **< 10MB** dan mudah diupload! 🎉
