# -*- coding: utf-8 -*-
"""
Sentiment Analysis - Lexicon-based and ML-based
Untuk Soal 5: Opinion Mining
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class LexiconSentimentAnalyzer:
    """
    Lexicon-based sentiment analysis
    """
    
    def __init__(self):
        """Initialize with Indonesian sentiment lexicon"""
        # Kata-kata positif (40+)
        self.positive_words = {
            'bagus', 'baik', 'hebat', 'luar biasa', 'sempurna', 'indah', 'cantik',
            'menarik', 'mengagumkan', 'fantastis', 'keren', 'oke', 'mantap', 'top',
            'suka', 'senang', 'gembira', 'bahagia', 'puas', 'terima kasih', 'thanks',
            'recommended', 'rekomendasi', 'terbaik', 'favorit', 'cinta', 'love',
            'excellent', 'amazing', 'wonderful', 'great', 'good', 'nice', 'best',
            'positif', 'optimis', 'berhasil', 'sukses', 'menang', 'juara', 'istimewa',
            'memuaskan', 'menyenangkan', 'menghibur', 'seru', 'asik', 'asyik'
        }
        
        # Kata-kata negatif (40+)
        self.negative_words = {
            'buruk', 'jelek', 'tidak', 'bukan', 'jangan', 'gagal', 'salah', 'rusak',
            'hancur', 'kecewa', 'sedih', 'marah', 'benci', 'hate', 'bad', 'worst',
            'terrible', 'horrible', 'awful', 'poor', 'boring', 'membosankan',
            'lambat', 'lelet', 'lemot', 'error', 'bug', 'masalah', 'problem',
            'susah', 'sulit', 'ribet', 'rumit', 'komplain', 'keluhan', 'protes',
            'negatif', 'pesimis', 'kalah', 'rugi', 'mengecewakan', 'menyebalkan',
            'menjengkelkan', 'tidak puas', 'kurang', 'minus', 'kekurangan'
        }
        
    def analyze(self, text):
        """
        Analyze sentiment of text
        
        Args:
            text: Input text
            
        Returns:
            dict: {
                'label': 'positif'/'negatif'/'netral',
                'score': sentiment score,
                'positive_words_found': list,
                'negative_words_found': list
            }
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        # Find positive and negative words
        pos_found = [w for w in words if w in self.positive_words]
        neg_found = [w for w in words if w in self.negative_words]
        
        # Calculate score
        pos_score = len(pos_found)
        neg_score = len(neg_found)
        
        # Determine sentiment
        if pos_score > neg_score:
            label = 'positif'
            score = pos_score - neg_score
        elif neg_score > pos_score:
            label = 'negatif'
            score = neg_score - pos_score
        else:
            label = 'netral'
            score = 0
        
        return {
            'label': label,
            'score': score,
            'positive_words_found': pos_found,
            'negative_words_found': neg_found,
            'positive_count': pos_score,
            'negative_count': neg_score
        }


class MLSentimentAnalyzer:
    """
    Machine Learning-based sentiment analysis
    """
    
    def __init__(self, algorithm='naive_bayes'):
        """
        Initialize ML sentiment analyzer
        
        Args:
            algorithm: 'naive_bayes' or 'knn'
        """
        self.algorithm = algorithm
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        
        if algorithm == 'naive_bayes':
            self.classifier = MultinomialNB()
        elif algorithm == 'knn':
            self.classifier = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.is_trained = False
        
    def train(self, texts, labels):
        """
        Train the classifier
        
        Args:
            texts: List of text documents
            labels: List of sentiment labels
        """
        # Vectorize
        X = self.vectorizer.fit_transform(texts)
        
        # Train
        self.classifier.fit(X, labels)
        self.is_trained = True
        
        print(f"Model trained with {len(texts)} samples")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
    def predict(self, text):
        """
        Predict sentiment
        
        Args:
            text: Input text
            
        Returns:
            label: Predicted sentiment label
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.vectorizer.transform([text])
        label = self.classifier.predict(X)[0]
        
        return label
    
    def predict_proba(self, text):
        """
        Predict sentiment with probabilities
        
        Args:
            text: Input text
            
        Returns:
            dict: {label: probability}
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.vectorizer.transform([text])
        
        if hasattr(self.classifier, 'predict_proba'):
            proba = self.classifier.predict_proba(X)[0]
            labels = self.classifier.classes_
            return dict(zip(labels, proba))
        else:
            # For KNN, use distance-based probability
            label = self.classifier.predict(X)[0]
            return {label: 1.0}
    
    def evaluate(self, texts, labels):
        """
        Evaluate model performance
        
        Args:
            texts: List of text documents
            labels: List of true labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.vectorizer.transform(texts)
        y_pred = self.classifier.predict(X)
        
        # Metrics
        report = classification_report(labels, y_pred, output_dict=True)
        cm = confusion_matrix(labels, y_pred)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'macro_f1': report['macro avg']['f1-score'],
            'accuracy': report['accuracy']
        }


def create_sample_sentiment_dataset():
    """
    Create sample sentiment dataset (200+ samples)
    """
    # Positive samples
    positive = [
        "Film ini sangat bagus dan menarik",
        "Saya sangat suka film ini, luar biasa",
        "Cerita yang hebat dan menghibur",
        "Akting yang sempurna, recommended",
        "Film terbaik yang pernah saya tonton",
        "Sangat puas dengan film ini",
        "Keren banget filmnya, mantap",
        "Mengagumkan, film yang fantastis",
        "Saya senang menonton film ini",
        "Film yang indah dan menyentuh hati",
        # ... (tambahkan lebih banyak)
    ] * 10  # Repeat to get 110 samples
    
    # Negative samples
    negative = [
        "Film ini buruk dan membosankan",
        "Sangat kecewa dengan film ini",
        "Akting yang jelek, tidak recommended",
        "Film terburuk yang pernah saya tonton",
        "Cerita yang membosankan dan lambat",
        "Tidak suka, film yang mengecewakan",
        "Sangat tidak puas dengan film ini",
        "Film yang menjengkelkan",
        "Buang-buang waktu menonton film ini",
        "Alur cerita yang rumit dan membingungkan",
        # ... (tambahkan lebih banyak)
    ] * 10  # Repeat to get 100 samples
    
    # Neutral samples (optional for 3-class)
    neutral = [
        "Film biasa saja",
        "Tidak terlalu bagus, tidak terlalu buruk",
        "Film standar",
        "Cerita cukup menarik",
        "Akting lumayan",
        # ...
    ] * 5  # 25 samples
    
    # Combine
    texts = positive + negative + neutral
    labels = ['positif']*len(positive) + ['negatif']*len(negative) + ['netral']*len(neutral)
    
    return texts, labels


def compare_sentiment_methods(texts, labels):
    """
    Compare lexicon-based vs ML-based sentiment analysis
    """
    print("\n" + "="*80)
    print("COMPARING SENTIMENT ANALYSIS METHODS")
    print("="*80 + "\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Method 1: Lexicon-based
    print("1. LEXICON-BASED APPROACH")
    print("-" * 40)
    
    lexicon_analyzer = LexiconSentimentAnalyzer()
    y_pred_lexicon = [lexicon_analyzer.analyze(text)['label'] for text in X_test]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lexicon))
    
    cm_lexicon = confusion_matrix(y_test, y_pred_lexicon)
    print("\nConfusion Matrix:")
    print(cm_lexicon)
    
    # Method 2: ML-based (Naive Bayes)
    print("\n2. ML-BASED APPROACH (Naive Bayes)")
    print("-" * 40)
    
    ml_analyzer_nb = MLSentimentAnalyzer(algorithm='naive_bayes')
    ml_analyzer_nb.train(X_train, y_train)
    
    results_nb = ml_analyzer_nb.evaluate(X_test, y_test)
    print("\nClassification Report:")
    print(classification_report(y_test, 
                                ml_analyzer_nb.classifier.predict(
                                    ml_analyzer_nb.vectorizer.transform(X_test)
                                )))
    
    print("\nConfusion Matrix:")
    print(results_nb['confusion_matrix'])
    
    # Method 3: ML-based (KNN)
    print("\n3. ML-BASED APPROACH (K-NN)")
    print("-" * 40)
    
    ml_analyzer_knn = MLSentimentAnalyzer(algorithm='knn')
    ml_analyzer_knn.train(X_train, y_train)
    
    results_knn = ml_analyzer_knn.evaluate(X_test, y_test)
    print("\nClassification Report:")
    print(classification_report(y_test, 
                                ml_analyzer_knn.classifier.predict(
                                    ml_analyzer_knn.vectorizer.transform(X_test)
                                )))
    
    print("\nConfusion Matrix:")
    print(results_knn['confusion_matrix'])
    
    # Visualize comparison
    _plot_sentiment_comparison(cm_lexicon, results_nb['confusion_matrix'], 
                               results_knn['confusion_matrix'], 
                               labels=sorted(set(labels)))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    f1_lexicon = f1_score(y_test, y_pred_lexicon, average='macro')
    
    summary_data = {
        'Method': ['Lexicon-based', 'Naive Bayes', 'K-NN'],
        'Macro F1': [f1_lexicon, results_nb['macro_f1'], results_knn['macro_f1']],
        'Accuracy': [
            np.mean(np.array(y_test) == np.array(y_pred_lexicon)),
            results_nb['accuracy'],
            results_knn['accuracy']
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    print("\n", df_summary.to_string(index=False))
    
    return df_summary


def _plot_sentiment_comparison(cm1, cm2, cm3, labels):
    """Plot confusion matrices comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    titles = ['Lexicon-based', 'Naive Bayes', 'K-NN']
    cms = [cm1, cm2, cm3]
    
    for ax, cm, title in zip(axes, cms, titles):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f'{title}\nConfusion Matrix', fontsize=14)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('sentiment_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVisualisasi disimpan: sentiment_comparison.png")
    plt.close()


def demo_sentiment_analysis():
    """Demo sentiment analysis"""
    print("="*80)
    print("SENTIMENT ANALYSIS DEMO")
    print("="*80)
    
    # Create sample dataset
    print("\nCreating sample dataset...")
    texts, labels = create_sample_sentiment_dataset()
    print(f"Dataset size: {len(texts)} samples")
    print(f"Label distribution: {dict(pd.Series(labels).value_counts())}")
    
    # Compare methods
    results = compare_sentiment_methods(texts, labels)
    
    # Save results
    results.to_csv('sentiment_analysis_results.csv', index=False)
    print("\nResults saved to: sentiment_analysis_results.csv")


if __name__ == "__main__":
    demo_sentiment_analysis()
