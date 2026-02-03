#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sentiment Analysis CLI Demo
Untuk Soal 5: sentiment_demo.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sentiment_analysis import LexiconSentimentAnalyzer, MLSentimentAnalyzer, create_sample_sentiment_dataset
from collections import Counter
import pandas as pd


def analyze_sentiment(text, lexicon_analyzer, ml_analyzer=None):
    """
    Analyze sentiment using both methods
    
    Args:
        text: Input text
        lexicon_analyzer: LexiconSentimentAnalyzer instance
        ml_analyzer: Optional MLSentimentAnalyzer instance
    """
    print("="*80)
    print("SENTIMENT ANALYSIS RESULT")
    print("="*80)
    
    print(f"\nInput Text: {text}\n")
    
    # Lexicon-based analysis
    print("-" * 40)
    print("LEXICON-BASED ANALYSIS")
    print("-" * 40)
    
    lexicon_result = lexicon_analyzer.analyze(text)
    
    print(f"Sentiment: {lexicon_result['label'].upper()}")
    print(f"Score: {lexicon_result['score']}")
    print(f"Positive words found ({lexicon_result['positive_count']}): {', '.join(lexicon_result['positive_words_found'][:10])}")
    print(f"Negative words found ({lexicon_result['negative_count']}): {', '.join(lexicon_result['negative_words_found'][:10])}")
    
    # ML-based analysis (if available)
    if ml_analyzer and ml_analyzer.is_trained:
        print("\n" + "-" * 40)
        print("ML-BASED ANALYSIS")
        print("-" * 40)
        
        ml_label = ml_analyzer.predict(text)
        ml_proba = ml_analyzer.predict_proba(text)
        
        print(f"Sentiment: {ml_label.upper()}")
        print(f"Probabilities:")
        for label, prob in sorted(ml_proba.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {label}: {prob:.2%}")
    
    print()


def analyze_by_topic(texts, labels, topics):
    """
    Analyze sentiment distribution by topic
    
    Args:
        texts: List of texts
        labels: List of sentiment labels
        topics: List of topics for each text
    """
    print("="*80)
    print("SENTIMENT DISTRIBUTION BY TOPIC")
    print("="*80 + "\n")
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'sentiment': labels,
        'topic': topics
    })
    
    # Calculate distribution
    for topic in sorted(df['topic'].unique()):
        topic_data = df[df['topic'] == topic]
        sentiment_counts = topic_data['sentiment'].value_counts()
        total = len(topic_data)
        
        print(f"Topic: {topic}")
        print(f"Total: {total} reviews")
        print("Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total) * 100
            print(f"  - {sentiment}: {count} ({percentage:.1f}%)")
        print()


def interactive_mode():
    """Interactive sentiment analysis mode"""
    print("="*80)
    print("SENTIMENT ANALYSIS INTERACTIVE MODE")
    print("="*80)
    print("\nInitializing analyzers...")
    
    # Initialize lexicon analyzer
    lexicon_analyzer = LexiconSentimentAnalyzer()
    print("✓ Lexicon analyzer ready")
    
    # Initialize and train ML analyzer
    print("\nTraining ML analyzer...")
    texts, labels = create_sample_sentiment_dataset()
    ml_analyzer = MLSentimentAnalyzer(algorithm='naive_bayes')
    ml_analyzer.train(texts, labels)
    print("✓ ML analyzer trained\n")
    
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for available commands")
    print("Type 'demo' to see topic analysis demo\n")
    
    while True:
        try:
            text = input("Enter text to analyze: ").strip()
            
            if not text:
                continue
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if text.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Type any text to analyze sentiment")
                print("  - 'demo': Show sentiment distribution by topic")
                print("  - 'quit' or 'exit': Exit the program")
                print("  - 'help': Show this help message")
                print()
                continue
            
            if text.lower() == 'demo':
                # Demo topic analysis
                print("\nGenerating demo data...")
                
                # Create sample data with topics
                topics = ['Action Films', 'Drama Films', 'Comedy Films']
                demo_texts = texts[:60]  # Use first 60 samples
                demo_labels = labels[:60]
                demo_topics = [topics[i % 3] for i in range(60)]
                
                analyze_by_topic(demo_texts, demo_labels, demo_topics)
                continue
            
            # Analyze sentiment
            analyze_sentiment(text, lexicon_analyzer, ml_analyzer)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            print()


def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Batch mode: analyze text from command line
        text = ' '.join(sys.argv[1:])
        
        print("Initializing analyzers...")
        lexicon_analyzer = LexiconSentimentAnalyzer()
        
        # Train ML analyzer
        texts, labels = create_sample_sentiment_dataset()
        ml_analyzer = MLSentimentAnalyzer(algorithm='naive_bayes')
        ml_analyzer.train(texts, labels)
        
        analyze_sentiment(text, lexicon_analyzer, ml_analyzer)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
