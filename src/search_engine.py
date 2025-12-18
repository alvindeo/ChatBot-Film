import os
import glob
import re
import nltk
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# --- Preprocessing ---
def cleantext(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenizetext(text):
    return text.split()

def removestopwordstokens(tokens):
    return [t for t in tokens if t not in stop_words]

def stemtokens(tokens):
    ps = PorterStemmer()
    return [ps.stem(t) for t in tokens]

def preprocess(text):
    text = cleantext(text)
    tokens = tokenizetext(text)
    tokens = removestopwordstokens(tokens)
    tokens = stemtokens(tokens)
    return tokens


# --- Boolean IR ---
class BooleanIR:
    def __init__(self):
        self.inverted_index = defaultdict(set)

    def build_index(self, documents):
        for doc_id, doc_tokens in enumerate(documents):
            for token in doc_tokens:
                self.inverted_index[token].add(doc_id)

    def query(self, q_tokens):
        result = None
        for token in q_tokens:
            docs = self.inverted_index.get(token, set())
            if result is None:
                result = docs
            else:
                result = result.intersection(docs)
        return result if result else set()


# --- VSM IR (TF-IDF) ---
class VSMIR:
    def __init__(self, docs_questions):
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(docs_questions)

    def rank(self, query, top_k=3):
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.doc_vectors).flatten()
        # Ambil top_k indices dengan score tertinggi (descending order)
        top_indices = scores.argsort()[-top_k:][::-1]
        return [(i, scores[i]) for i in top_indices if scores[i] > 0]


# --- Search Engine Kombinasi ---
class SearchEngine:
    def __init__(self, docs, questions):
        self.docs = docs
        self.questions = questions
        self.preprocessed_docs = [preprocess(doc) for doc in docs]
        self.boolean_ir = BooleanIR()
        self.boolean_ir.build_index(self.preprocessed_docs)
        self.vsm_ir = VSMIR(questions)

    def search(self, query, model='vsm', k=4):
        q_tokens = preprocess(query)
        if model == 'boolean':
            doc_ids = self.boolean_ir.query(q_tokens)
            return [(did, 1.0) for did in doc_ids]
        elif model == 'vsm':
            return self.vsm_ir.rank(query, top_k=k)


# --- Chatbot RAG sederhana ---
class RAGChatbot:
    def __init__(self, questions, documents, file_names):
        self.questions = questions
        self.documents = documents
        self.file_names = file_names
        self.engine = SearchEngine(documents, questions)

    def get_best_answer(self, query, top_k=3):
        """Mengembalikan jawaban dengan score tertinggi"""
        results = self.engine.search(query, model='vsm', k=top_k)
        if not results:
            return "Maaf, tidak ada informasi yang sesuai."
        
        # Ambil hasil dengan score tertinggi (index 0 karena sudah terurut descending)
        doc_id, score = results[0]
        
        # Validasi score threshold
        if score < 0.8:
            return "### ⚠️ Pencarian anda kurang tepat\n\nSilakan coba dengan kata kunci yang lebih spesifik atau relevan dengan topik film."
        
        answer = self.documents[doc_id]
        file_name = self.file_names[doc_id]
        
        return f"### 🏆 Jawaban Terbaik (Score: {score:.2f})\n\n📄 **Sumber: {file_name}**\n\n{answer}"
    
    def generate_answer(self, query, top_k=3):
        """Mengembalikan semua top-k hasil pencarian"""
        results = self.engine.search(query, model='vsm', k=top_k)
        if not results:
            return "Maaf, tidak ada informasi yang sesuai."
        answers = []
        for idx, (doc_id, score) in enumerate(results, 1):
            snippet = self.documents[doc_id][:200]
            file_name = self.file_names[doc_id]
            answers.append(f"**{idx}. {file_name}** — (score: {score:.2f})\n{snippet}...")
        return "### 📊 Top 3 Hasil Pencarian:\n\n" + "\n\n".join(answers)


# --- Membaca dataset dari folder /dataset ---
questions, answers, file_names = [], [], []

for filepath in glob.glob('dataset/*.txt'):
    file_name = os.path.basename(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if '->' in line:
                q, a = line.split('->')
                questions.append(q.strip("- ").lower().strip())
                answers.append(a.strip())
                file_names.append(file_name)

print(f"Loaded {len(questions)} questions and answers from dataset.")

# --- Buat chatbot instance ---
chatbot = RAGChatbot(questions, answers, file_names)


# --- Gradio UI ---
def chatbot_response(user_input):
    best = chatbot.get_best_answer(user_input)
    all_results = chatbot.generate_answer(user_input)
    return best, all_results

with gr.Blocks(title="Chatbot Film") as demo:
    gr.Markdown("# 🎬 Chatbot Tentang Film\nTanyakan apa saja seputar film, genre, sutradara, dan aktor.")
    
    with gr.Row():
        question_input = gr.Textbox(label="Pertanyaan", placeholder="Contoh: Siapa sutradara film Titanic?", scale=4)
        ask_button = gr.Button("Tanyakan", scale=1)
    
    with gr.Row():
        with gr.Column(scale=1):
            best_answer_output = gr.Markdown(label="Jawaban Terbaik")
        with gr.Column(scale=1):
            all_results_output = gr.Markdown(label="Semua Hasil Pencarian")
    
    reset_button = gr.Button("🔄 Reset")
    
    ask_button.click(chatbot_response, inputs=question_input, outputs=[best_answer_output, all_results_output])
    reset_button.click(lambda: ("", "", ""), None, [question_input, best_answer_output, all_results_output])

demo.launch(share=True)


# --- Pengujian manual ---
try:
    test_query = "Film aksi terbaik 2025"
    response = chatbot.generate_answer(test_query, top_k=3)
    print(response)

    test_query_2 = "Siapa sutradara film Titanic?"
    response_2 = chatbot.generate_answer(test_query_2, top_k=3)
    print("\n" + response_2)

except Exception as e:
    print(f"An error occurred: {e}")
