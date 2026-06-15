import requests
import os

diagrams = {
    "rag_architecture": """graph TD
    User([User]) -->|Input Pertanyaan| Preprocess(Preprocessing<br>Cleantext, Tokenize, Stem)
    Preprocess --> Intent{Intent Classifier<br>KNN}
    Intent -->|Deteksi Intent| VSM(Vector Space Model<br>TF-IDF)
    Preprocess --> VSM
    
    VSM -->|Cosine Similarity| Eval{Evaluasi Score}
    
    Eval -->|Score >= 0.7| ExactMatch[Hasil Pencarian Database<br>Top K Docs]
    Eval -->|Score < 0.6| LLM{LLM Enhancement<br>Gemini / Groq / OpenAI}
    
    LLM -->|Prompt + Intent Context| Generative[Respon AI Generatif]
    ExactMatch --> Output([Final Response])
    Generative --> Output
    Output --> User
    
    classDef process fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px;
    classDef ai fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px;
    classDef decision fill:#fff3e0,stroke:#ff9800,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;
    
    class Preprocess,VSM process;
    class Intent,LLM,Generative ai;
    class Eval decision;
    class ExactMatch,Output output;
""",
    "system_workflow": """sequenceDiagram
    participant U as User
    participant G as Gradio UI
    participant C as RAG Chatbot
    participant E as Search Engine
    participant M as Intent Model (KNN)
    participant L as LLM Manager
    
    Note over C, L: Initialization Phase
    C->>E: Load Data (.txt & MovieDataset)
    E-->>C: Data Loaded
    C->>E: Preprocess & Build Index (Boolean + VSM)
    C->>M: Train/Load Intent KNN Model (.pkl)
    C->>L: Initialize LLM Provider 
    
    Note over U, L: Chat Phase (Online)
    U->>G: Kirim Pertanyaan
    G->>C: get_best_answer(query)
    
    C->>M: predict(query)
    M-->>C: Intent & Confidence Score
    
    C->>E: search(query, model='vsm')
    E-->>C: Top-K Ranked Docs & Cosine Score
    
    alt Score >= 0.7 (High Confidence)
        C-->>G: Return Base Response (Hasil Data Asli DB)
    else Score < 0.6 (Low Confidence)
        C->>L: generate_enhanced_response(prompt, intent)
        L-->>C: AI Generated Answer (Fallback)
        C-->>G: Return Enhanced Response
    end
    
    G-->>U: Tampilkan Jawaban ke Layar
""",
    "retrieval_pipeline": """graph LR
    subgraph Offline Phase: Indexing
        A[Dataset .txt & TMDb] --> B(Text Cleaning)
        B --> C(Tokenization)
        C --> D(Stopword Removal)
        D --> E(Stemming)
        E --> F[(TF-IDF Vectorizer)]
    end
    
    subgraph Online Phase: Querying
        G[User Query] --> H(Text Cleaning)
        H --> I(Tokenization)
        I --> J(Stopword Removal)
        J --> K(Stemming)
        K --> L(Vectorize Query)
    end
    
    F -.->|Fit & Transform| M{Cosine Similarity}
    L --> M
    M --> N[Ranked Results Top-K]
    
    style F fill:#eceff1,stroke:#607d8b,stroke-width:2px;
    style N fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;
"""
}

output_dir = "images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Mulai menghasilkan gambar diagram...")

for name, content in diagrams.items():
    print(f"Mengunduh {name}.png...")
    response = requests.post("https://kroki.io/mermaid/png", data=content.encode("utf-8"))
    if response.status_code == 200:
        filepath = os.path.join(output_dir, f"{name}.png")
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Berhasil menyimpan {filepath}")
    else:
        print(f"Gagal mengunduh {name}. Status code: {response.status_code}")
        print(response.text)

print("Selesai.")
