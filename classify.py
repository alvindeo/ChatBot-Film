#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI Demo untuk Klasifikasi Dokumen
Untuk Soal 2: Demo classify.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from intent_classifier import IntentClassifier
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def load_classifier():
    """Load trained intent classifier"""
    model_path = 'models/intent_knn.pkl'
    
    if os.path.exists(model_path):
        print("Loading trained classifier...")
        classifier = IntentClassifier.load(model_path)
        print("✓ Classifier loaded successfully\n")
    else:
        print("Training new classifier...")
        classifier = IntentClassifier(algorithm='knn', k=5)
        classifier.train()
        classifier.save(model_path)
        print("✓ Classifier trained and saved\n")
    
    return classifier


def classify_text(text, classifier, show_neighbors=True, k=3):
    """
    Classify text and show top-k neighbors
    
    Args:
        text: Input text to classify
        classifier: Trained classifier
        show_neighbors: Whether to show nearest neighbors
        k: Number of neighbors to show
    """
    print("="*80)
    print("CLASSIFICATION RESULT")
    print("="*80)
    
    # Predict
    result = classifier.predict_with_details(text)
    
    print(f"\nInput Text: {text}")
    print(f"\nPredicted Label: {result['intent']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Description: {result['description']}")
    
    if 'all_probabilities' in result and result['all_probabilities']:
        print(f"\nAll Class Probabilities:")
        for intent, prob in sorted(result['all_probabilities'].items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"  - {intent}: {prob:.2%}")
    
    # Show top-k neighbors
    if show_neighbors:
        print(f"\n{'='*80}")
        print(f"TOP-{k} NEAREST NEIGHBORS")
        print(f"{'='*80}\n")
        
        # Vectorize query
        query_vec = classifier.vectorizer.transform([text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, classifier.X_train).flatten()
        
        # Get top-k indices
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        for rank, idx in enumerate(top_k_indices, 1):
            neighbor_text = classifier.training_queries[idx]
            neighbor_label = classifier.training_labels[idx]
            similarity = similarities[idx]
            
            print(f"{rank}. Label: {neighbor_label}")
            print(f"   Similarity: {similarity:.4f}")
            print(f"   Text: {neighbor_text}")
            print()


def interactive_mode(classifier):
    """Interactive classification mode"""
    print("\n" + "="*80)
    print("INTERACTIVE CLASSIFICATION MODE")
    print("="*80)
    print("\nType 'quit' or 'exit' to stop")
    print("Type 'help' for available commands\n")
    
    while True:
        try:
            text = input("Enter text to classify: ").strip()
            
            if not text:
                continue
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if text.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Type any text to classify it")
                print("  - 'quit' or 'exit': Exit the program")
                print("  - 'help': Show this help message")
                print()
                continue
            
            # Classify
            classify_text(text, classifier, show_neighbors=True, k=3)
            print()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print()


def main():
    """Main function"""
    print("="*80)
    print("DOCUMENT CLASSIFICATION CLI")
    print("="*80)
    print()
    
    # Load classifier
    classifier = load_classifier()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Batch mode: classify text from command line
        text = ' '.join(sys.argv[1:])
        classify_text(text, classifier, show_neighbors=True, k=3)
    else:
        # Interactive mode
        interactive_mode(classifier)


if __name__ == "__main__":
    main()
