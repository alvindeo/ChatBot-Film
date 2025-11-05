import os
from google.colab import files
import glob

# Upload file
uploaded = files.upload()

# Pindahkan file ke folder data
os.makedirs('data', exist_ok=True)
for filename in uploaded.keys():
    os.rename(filename, os.path.join('data', filename))

print("File berhasil diupload dan dipindahkan ke folder /content/data")

# Baca file .txt dan parsing
questions, answers = [], []
for filepath in glob.glob('data/*.txt'):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if '->' in line:
                q, a = line.split('->')
                questions.append(q.strip("- ").lower().strip())
                answers.append(a.strip())

print(f"Berhasil membaca {len(questions)} pasangan pertanyaan-jawaban dari dataset.")