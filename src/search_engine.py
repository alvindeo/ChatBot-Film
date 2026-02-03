# -*- coding: utf-8 -*-
import sys
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
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Import modul baru
from intent_classifier import IntentClassifier
from llm_integration import LLMManager
from movie_dataset import MovieDataset

# Load environment variables
load_dotenv()

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
        
        print("🔍 Building search engine...")
        
        # Preprocessing documents dengan progress bar
        self.preprocessed_docs = []
        for doc in tqdm(docs, desc="📝 Preprocessing docs", unit="doc"):
            self.preprocessed_docs.append(preprocess(doc))
        
        # Build Boolean IR index
        with tqdm(total=100, desc="🗂️ Building index", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            self.boolean_ir = BooleanIR()
            pbar.update(30)
            self.boolean_ir.build_index(self.preprocessed_docs)
            pbar.update(40)
            self.vsm_ir = VSMIR(questions)
            pbar.update(30)

    def search(self, query, model='vsm', k=4):
        q_tokens = preprocess(query)
        if model == 'boolean':
            doc_ids = self.boolean_ir.query(q_tokens)
            return [(did, 1.0) for did in doc_ids]
        elif model == 'vsm':
            return self.vsm_ir.rank(query, top_k=k)


# --- Chatbot RAG dengan Intent Classification & LLM ---
class RAGChatbot:
    def __init__(self, questions, documents, file_names, use_intent=True, use_llm=True):
        self.questions = questions
        self.documents = documents
        self.file_names = file_names
        self.engine = SearchEngine(documents, questions)
        
        # Initialize Intent Classifier
        self.use_intent = use_intent
        self.intent_classifier = None
        if use_intent:
            try:
                print("🎯 Initializing Intent Classifier...")
                # Coba load model yang sudah di-train
                if os.path.exists('models/intent_knn.pkl'):
                    with tqdm(total=100, desc="📥 Loading intent model", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                        self.intent_classifier = IntentClassifier.load('models/intent_knn.pkl')
                        pbar.update(100)
                    print("✅ Intent Classifier loaded from cache")
                else:
                    # Train baru jika belum ada
                    print("🔨 Training new intent classifier...")
                    with tqdm(total=100, desc="🧠 Training KNN model", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                        self.intent_classifier = IntentClassifier(algorithm='knn', k=3)
                        pbar.update(30)
                        self.intent_classifier.train()
                        pbar.update(50)
                        self.intent_classifier.save('models/intent_knn.pkl')
                        pbar.update(20)
            except Exception as e:
                print(f"⚠️ Intent Classifier disabled: {e}")
                self.use_intent = False
        
        # Initialize LLM Manager
        self.use_llm = use_llm
        self.llm_manager = None
        if use_llm:
            try:
                print("🤖 Initializing LLM Manager...")
                default_provider = os.getenv('DEFAULT_LLM', 'none')
                
                with tqdm(total=100, desc="🔌 Connecting to LLM", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                    self.llm_manager = LLMManager(default_provider=default_provider)
                    pbar.update(70)
                    
                    if self.llm_manager.is_active():
                        pbar.update(30)
                        print(f"✅ LLM Active: {self.llm_manager.get_provider_info()['provider']}")
                    else:
                        pbar.update(30)
                        print("ℹ️ LLM Mode: Disabled (set DEFAULT_LLM in .env to enable)")
            except Exception as e:
                print(f"⚠️ LLM disabled: {e}")
                self.use_llm = False

    def get_best_answer(self, query, top_k=3):
        """Mengembalikan jawaban dengan score tertinggi"""
        # Detect intent
        intent_info = None
        if self.use_intent and self.intent_classifier:
            try:
                intent_info = self.intent_classifier.predict_with_details(query)
                print(f"🎯 Detected Intent: {intent_info['intent']} (confidence: {intent_info['confidence']:.2%})")
            except Exception as e:
                print(f"⚠️ Intent detection error: {e}")
        
        # Search
        results = self.engine.search(query, model='vsm', k=top_k)
        if not results:
            return "Maaf, tidak ada informasi yang sesuai."
        
        # Ambil hasil dengan score tertinggi
        doc_id, score = results[0]
        
        # Validasi score threshold
        if score < 0.6:
            # Score rendah: HANYA gunakan LLM (jika aktif), JANGAN tampilkan hasil database
            if self.use_llm and self.llm_manager and self.llm_manager.is_active():
                try:
                    intent_str = intent_info['intent'] if intent_info else None
                    # Generate response menggunakan LLM tanpa menampilkan hasil pencarian
                    enhanced = self.llm_manager.generate_enhanced_response(
                        query=query,
                        search_results="",  # Kosongkan search results
                        intent=intent_str,
                        max_tokens=400,
                        temperature=0.7
                    )
                    return f"🤖 **AI Response** (Score terlalu rendah: {score:.2f})\n\n{enhanced}"
                except Exception as e:
                    print(f"⚠️ LLM enhancement error: {e}")
                    # Fallback jika LLM error
                    base_response = "⚠️ Pencarian anda kurang tepat\n\nSilakan coba dengan kata kunci yang lebih spesifik atau relevan dengan topik film."
                    if intent_info:
                        base_response += f"\n\n💡 Terdeteksi bahwa Anda ingin: {intent_info['description']}"
                    return base_response
            else:
                # LLM tidak aktif, tampilkan pesan error
                base_response = "⚠️ Pencarian anda kurang tepat\n\nSilakan coba dengan kata kunci yang lebih spesifik atau relevan dengan topik film."
                if intent_info:
                    base_response += f"\n\n💡 Terdeteksi bahwa Anda ingin: {intent_info['description']}"
                return base_response
        
        # Score tinggi (>= 0.7): Tampilkan hasil database TANPA LLM enhancement
        answer = self.documents[doc_id]
        file_name = self.file_names[doc_id]
        
        base_response = f"🏆 Jawaban Terbaik (Score: {score:.2f})\n\nSumber: {file_name}\n\n{answer}"
        
        # TIDAK menggunakan LLM untuk score tinggi
        return base_response
    
    def generate_answer(self, query, top_k=3):
        """Mengembalikan semua top-k hasil pencarian"""
        # Detect intent
        intent_info = None
        if self.use_intent and self.intent_classifier:
            try:
                intent_info = self.intent_classifier.predict_with_details(query)
            except:
                pass
        
        results = self.engine.search(query, model='vsm', k=top_k)
        if not results:
            return "Maaf, tidak ada informasi yang sesuai."
        
        # Cek score tertinggi untuk threshold
        best_score = results[0][1]
        
        # Jika score terbaik < 0.7, JANGAN tampilkan hasil database
        if best_score < 0.7:
            header = ""
            if intent_info:
                header = f"🎯 Intent: {intent_info['intent']} ({intent_info['confidence']:.0%})\n\n"
            
            # Tampilkan pesan bahwa hasil tidak ditampilkan karena score rendah
            return header + f"⚠️ **Hasil pencarian tidak ditampilkan** (Score terlalu rendah: {best_score:.2f})\n\nSilakan gunakan kata kunci yang lebih spesifik atau relevan dengan topik film."
        
        # Score >= 0.7: Tampilkan hasil database
        answers = []
        for idx, (doc_id, score) in enumerate(results, 1):
            snippet = self.documents[doc_id][:200]
            file_name = self.file_names[doc_id]
            answers.append(f"{idx}. {file_name} — (score: {score:.2f})\n{snippet}...")
        
        header = "📊 Top 3 Hasil Pencarian:\n\n"
        if intent_info:
            header = f"🎯 Intent: {intent_info['intent']} ({intent_info['confidence']:.0%})\n\n" + header
        
        return header + "\n\n".join(answers)


# --- Membaca dataset dari folder /dataset dan movie dataset ---
print("\n" + "="*60)
print("🎬 LOADING MOVIE CHATBOT DATASETS")
print("="*60 + "\n")

questions, answers, file_names = [], [], []

# 1. Load dataset dari file .txt (existing)
print("📂 Loading existing text datasets...")
txt_files = glob.glob('dataset/*.txt')
for filepath in tqdm(txt_files, desc="📄 Processing text files", unit="file"):
    file_name = os.path.basename(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if '->' in line:
                q, a = line.split('->')
                questions.append(q.strip("- ").lower().strip())
                answers.append(a.strip())
                file_names.append(file_name)

print(f"✅ Loaded {len(questions)} Q&A from text files")

# 2. Load movie dataset (built-in sample)
print("\n📽️ Loading movie dataset...")
try:
    movie_dataset = MovieDataset()
    
    # Tampilkan progress saat loading
    print("⏳ Initializing movie dataset loader...")
    time.sleep(0.3)  # Small delay untuk visual feedback
    
    movie_questions, movie_answers, movie_files = movie_dataset.load_all(
        use_tmdb=False  # Set True jika punya TMDb API key
    )
    
    # Progress bar untuk menambahkan ke dataset utama
    print("🔄 Integrating movie data...")
    for i in tqdm(range(len(movie_questions)), desc="📊 Adding movie Q&A", unit="pair"):
        questions.append(movie_questions[i])
        answers.append(movie_answers[i])
        file_names.append(movie_files[i])
    
    print(f"✅ Added {len(movie_questions)} Q&A from movie dataset")
except Exception as e:
    print(f"⚠️ Movie dataset loading failed: {e}")

print(f"\n📊 Total Dataset: {len(questions)} Q&A pairs")
print("="*60 + "\n")

# --- Buat chatbot instance ---
print("🚀 Initializing RAG Chatbot System...")
print("="*60)

# Overall progress tracker
overall_steps = [
    "Loading datasets",
    "Building search engine", 
    "Initializing intent classifier",
    "Connecting LLM",
    "Finalizing setup"
]

with tqdm(total=len(overall_steps), desc="🎬 Overall Progress", unit="step", position=0) as overall_pbar:
    overall_pbar.set_postfix_str("Loading datasets")
    overall_pbar.update(1)
    
    overall_pbar.set_postfix_str("Building search engine")
    chatbot = RAGChatbot(questions, answers, file_names)
    overall_pbar.update(4)

print("\n" + "="*60)
print("✅ CHATBOT READY!")
print("="*60)
print(f"📊 Statistics:")
print(f"   • Total Q&A pairs: {len(questions)}")
print(f"   • Unique sources: {len(set(file_names))}")
print(f"   • Intent Classification: {'✅ Active' if chatbot.use_intent else '❌ Disabled'}")
print(f"   • LLM Enhancement: {'✅ Active' if chatbot.use_llm and chatbot.llm_manager.is_active() else '❌ Disabled'}")
print("="*60 + "\n")


# --- Gradio UI ---
def chatbot_response(user_input):
    best = chatbot.get_best_answer(user_input)
    all_results = chatbot.generate_answer(user_input)
    return best, all_results

# Get system info
llm_info = chatbot.llm_manager.get_provider_info() if chatbot.llm_manager else {'status': 'inactive'}
intent_status = "✅ Active (KNN)" if chatbot.use_intent else "❌ Disabled"
llm_status = f"✅ Active ({llm_info['provider']})" if llm_info['status'] == 'active' else "❌ Disabled"

with gr.Blocks(title="Chatbot Film - Enhanced", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎬 Chatbot Film - Enhanced Edition
    
    ### Fitur Baru:
    - 🎯 **Intent Classification** (KNN): {intent_status}
    - 🤖 **AI Enhancement** (LLM): {llm_status}
    - 📽️ **Movie Dataset**: 15+ film populer terintegrasi
    
    Tanyakan apa saja seputar film, genre, sutradara, aktor, dan rekomendasi!
    """.format(intent_status=intent_status, llm_status=llm_status))
    
    with gr.Row():
        question_input = gr.Textbox(
            label="Pertanyaan", 
            placeholder="Contoh: Rekomendasikan film action terbaik / Siapa sutradara Inception?", 
            scale=4
        )
        ask_button = gr.Button("🔍 Tanyakan", scale=1, variant="primary")
    
    with gr.Row():
        with gr.Column(scale=1):
            best_answer_output = gr.Textbox(
                label="🏆 Jawaban Terbaik", 
                lines=15, 
                max_lines=30, 
                interactive=False
            )
        with gr.Column(scale=1):
            all_results_output = gr.Textbox(
                label="📊 Semua Hasil Pencarian", 
                lines=15, 
                max_lines=30, 
                interactive=False
            )
    
    with gr.Row():
        reset_button = gr.Button("🔄 Reset")
        gr.Markdown("""
        **Tips:**
        - Gunakan kata kunci spesifik untuk hasil lebih akurat
        - Coba tanya: "film terbaik", "sutradara Nolan", "aktor Inception"
        """)
    
    ask_button.click(chatbot_response, inputs=question_input, outputs=[best_answer_output, all_results_output])
    question_input.submit(chatbot_response, inputs=question_input, outputs=[best_answer_output, all_results_output])
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
