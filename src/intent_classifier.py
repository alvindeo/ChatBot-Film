# -*- coding: utf-8 -*-
"""
Intent Classifier menggunakan KNN dan Naive Bayes
untuk mendeteksi intent user sebelum pencarian
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


class IntentClassifier:
    """
    Classifier untuk mendeteksi intent user dari query
    Mendukung 2 algoritma: KNN dan Naive Bayes
    """
    
    # Definisi intent categories
    INTENTS = {
        'rekomendasi': 'User ingin rekomendasi film',
        'info_film': 'User ingin info detail tentang film tertentu',
        'sutradara': 'User bertanya tentang sutradara',
        'aktor': 'User bertanya tentang aktor/aktris',
        'genre': 'User bertanya tentang genre film',
        'tahun': 'User bertanya tentang tahun rilis',
        'penghargaan': 'User bertanya tentang penghargaan/awards',
        'trivia': 'User ingin fakta menarik/trivia',
        'soundtrack': 'User bertanya tentang musik/soundtrack',
        'teknologi': 'User bertanya tentang efek visual/teknologi'
    }
    
    def __init__(self, algorithm='knn', k=5):
        """
        Initialize classifier
        
        Args:
            algorithm: 'knn' atau 'naive_bayes'
            k: jumlah neighbors untuk KNN (default: 5)
        """
        self.algorithm = algorithm
        self.k = k
        self.vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        
        if algorithm == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        elif algorithm == 'naive_bayes':
            self.model = MultinomialNB(alpha=1.0)
        else:
            raise ValueError("Algorithm harus 'knn' atau 'naive_bayes'")
        
        self.is_trained = False
        
    def _create_training_data(self):
        """
        Membuat training data untuk intent classification
        """
        training_data = {
            'rekomendasi': [
                'rekomendasikan film bagus',
                'film apa yang bagus',
                'saran film untuk ditonton',
                'film terbaik tahun ini',
                'apa film yang recommended',
                'film apa yang worth it',
                'kasih rekomendasi film dong',
                'film bagus apa ya',
                'mau nonton film apa ya',
                'film yang wajib ditonton',
                'film paling recommended',
                'film terbaik sepanjang masa',
                'film yang harus ditonton',
                'rekomendasi film netflix',
                'film bagus di netflix'
            ],
            'info_film': [
                'ceritakan tentang film',
                'sinopsis film',
                'alur cerita film',
                'tentang film',
                'film ini tentang apa',
                'apa isi cerita film',
                'ringkasan film',
                'plot film',
                'tema film',
                'cerita film',
                'informasi film',
                'detail film',
                'deskripsi film'
            ],
            'sutradara': [
                'siapa sutradara film',
                'siapa yang menyutradarai',
                'sutradara film ini siapa',
                'director film',
                'siapa pembuat film',
                'film karya sutradara',
                'sutradara terkenal',
                'sutradara terbaik',
                'siapa yang mengarahkan film',
                'film disutradarai oleh',
                'karya sutradara'
            ],
            'aktor': [
                'siapa pemeran film',
                'aktor dalam film',
                'aktris film',
                'siapa yang main di film',
                'cast film',
                'pemain film',
                'siapa bintang film',
                'aktor terkenal',
                'aktris cantik',
                'pemeran utama',
                'siapa yang berperan',
                'film dibintangi oleh'
            ],
            'genre': [
                'genre film apa',
                'jenis film',
                'kategori film',
                'film bergenre',
                'film action',
                'film horor',
                'film komedi',
                'film drama',
                'film romance',
                'film thriller',
                'film sci-fi',
                'film animasi',
                'film dokumenter'
            ],
            'tahun': [
                'kapan film dirilis',
                'tahun rilis film',
                'film tahun berapa',
                'kapan film keluar',
                'film produksi tahun',
                'film rilis kapan',
                'film keluaran tahun',
                'film terbaru',
                'film lama',
                'film klasik',
                'film modern'
            ],
            'penghargaan': [
                'penghargaan film',
                'film pemenang oscar',
                'film dapat award',
                'nominasi film',
                'piala film',
                'film menang penghargaan',
                'film peraih oscar',
                'film golden globe',
                'film festival',
                'awards film'
            ],
            'trivia': [
                'fakta menarik film',
                'trivia film',
                'hal unik film',
                'fakta tersembunyi',
                'behind the scenes',
                'fakta produksi film',
                'cerita di balik film',
                'fakta unik',
                'hal menarik tentang film',
                'easter egg film'
            ],
            'soundtrack': [
                'musik film',
                'soundtrack film',
                'lagu tema film',
                'score film',
                'musik latar film',
                'lagu di film',
                'composer film',
                'musik pengiring film',
                'ost film'
            ],
            'teknologi': [
                'efek visual film',
                'cgi film',
                'teknologi film',
                'visual effects',
                'efek khusus',
                'animasi film',
                'teknologi produksi',
                'sinematografi',
                'teknik pengambilan gambar',
                'kamera film'
            ]
        }
        
        queries = []
        labels = []
        
        for intent, examples in training_data.items():
            queries.extend(examples)
            labels.extend([intent] * len(examples))
        
        return queries, labels
    
    def train(self, custom_queries=None, custom_labels=None):
        """
        Train the classifier
        
        Args:
            custom_queries: Optional list of custom training queries
            custom_labels: Optional list of custom training labels
        """
        if custom_queries and custom_labels:
            queries = custom_queries
            labels = custom_labels
        else:
            queries, labels = self._create_training_data()
        
        # Vectorize
        X = self.vectorizer.fit_transform(queries)
        y = np.array(labels)
        
        # Train
        self.model.fit(X, y)
        self.is_trained = True
        
        # Evaluate (optional - split for validation)
        if len(queries) > 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"✅ Intent Classifier ({self.algorithm.upper()}) trained!")
            print(f"📊 Validation Accuracy: {accuracy:.2%}")
        else:
            print(f"✅ Intent Classifier ({self.algorithm.upper()}) trained!")
        
        return self
    
    def predict(self, query):
        """
        Predict intent dari user query
        
        Args:
            query: User's query string
            
        Returns:
            tuple: (predicted_intent, confidence_score)
        """
        if not self.is_trained:
            raise ValueError("Model belum di-train! Panggil .train() terlebih dahulu.")
        
        X = self.vectorizer.transform([query])
        
        # Predict
        intent = self.model.predict(X)[0]
        
        # Get confidence
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = max(proba)
        else:
            # Untuk KNN, gunakan distance-based confidence
            distances, indices = self.model.kneighbors(X)
            # Inverse distance sebagai confidence (semakin dekat = semakin confident)
            avg_distance = np.mean(distances[0])
            confidence = 1 / (1 + avg_distance)  # Normalize to 0-1
        
        return intent, confidence
    
    def predict_with_details(self, query):
        """
        Predict dengan detail lengkap
        
        Returns:
            dict: {
                'intent': predicted intent,
                'confidence': confidence score,
                'description': intent description,
                'all_probabilities': dict of all intent probabilities (if available)
            }
        """
        intent, confidence = self.predict(query)
        
        result = {
            'intent': intent,
            'confidence': confidence,
            'description': self.INTENTS.get(intent, 'Unknown intent')
        }
        
        # Add all probabilities if available
        if hasattr(self.model, 'predict_proba'):
            X = self.vectorizer.transform([query])
            proba = self.model.predict_proba(X)[0]
            classes = self.model.classes_
            result['all_probabilities'] = dict(zip(classes, proba))
        
        return result
    
    def save(self, filepath='models/intent_classifier.pkl'):
        """Save trained model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'algorithm': self.algorithm,
                'k': self.k,
                'vectorizer': self.vectorizer,
                'model': self.model,
                'is_trained': self.is_trained
            }, f)
        print(f"💾 Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath='models/intent_classifier.pkl'):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        classifier = cls(algorithm=data['algorithm'], k=data.get('k', 5))
        classifier.vectorizer = data['vectorizer']
        classifier.model = data['model']
        classifier.is_trained = data['is_trained']
        
        print(f"📂 Model loaded from {filepath}")
        return classifier


# === Testing & Demo ===
if __name__ == "__main__":
    # Fix encoding untuk Windows
    import sys
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("🧪 Testing Intent Classifier\n")
    
    # Test KNN
    print("=" * 50)
    print("Testing K-Nearest Neighbors (KNN)")
    print("=" * 50)
    knn_classifier = IntentClassifier(algorithm='knn', k=3)
    knn_classifier.train()
    
    test_queries = [
        "rekomendasikan film action terbaik",
        "siapa sutradara film Inception?",
        "film apa yang bagus untuk ditonton?",
        "aktor di film Titanic siapa?",
        "fakta menarik tentang film Interstellar"
    ]
    
    print("\n📝 Test Queries (KNN):")
    for query in test_queries:
        result = knn_classifier.predict_with_details(query)
        print(f"\nQuery: '{query}'")
        print(f"  → Intent: {result['intent']}")
        print(f"  → Confidence: {result['confidence']:.2%}")
        print(f"  → Description: {result['description']}")
    
    # Test Naive Bayes
    print("\n" + "=" * 50)
    print("Testing Naive Bayes")
    print("=" * 50)
    nb_classifier = IntentClassifier(algorithm='naive_bayes')
    nb_classifier.train()
    
    print("\n📝 Test Queries (Naive Bayes):")
    for query in test_queries:
        result = nb_classifier.predict_with_details(query)
        print(f"\nQuery: '{query}'")
        print(f"  → Intent: {result['intent']}")
        print(f"  → Confidence: {result['confidence']:.2%}")
        print(f"  → Description: {result['description']}")
    
    # Save models
    knn_classifier.save('models/intent_knn.pkl')
    nb_classifier.save('models/intent_nb.pkl')
