# -*- coding: utf-8 -*-
"""
Extractive Text Summarization
Untuk Soal 4: Peringkasan Dokumen
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.tokenize import sent_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class ExtractiveSummarizer:
    """
    Extractive summarization menggunakan TF-IDF sentence scoring
    """
    
    def __init__(self, num_sentences=3):
        """
        Initialize summarizer
        
        Args:
            num_sentences: Number of sentences to include in summary
        """
        self.num_sentences = num_sentences
        
    def preprocess_text(self, text):
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def split_sentences(self, text):
        """Split text into sentences"""
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback to simple split
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def score_sentences_tfidf(self, sentences):
        """
        Score sentences using TF-IDF
        
        Args:
            sentences: List of sentences
            
        Returns:
            scores: Array of sentence scores
        """
        if len(sentences) == 0:
            return np.array([])
        
        if len(sentences) == 1:
            return np.array([1.0])
        
        # Vectorize sentences
        vectorizer = TfidfVectorizer(stop_words=None)
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores (sum of TF-IDF values)
            scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            # Normalize scores
            if scores.max() > 0:
                scores = scores / scores.max()
                
        except:
            # If vectorization fails, return equal scores
            scores = np.ones(len(sentences))
        
        return scores
    
    def score_sentences_position(self, sentences):
        """
        Score sentences based on position (first sentences are more important)
        
        Args:
            sentences: List of sentences
            
        Returns:
            scores: Array of position scores
        """
        n = len(sentences)
        if n == 0:
            return np.array([])
        
        # Linear decay: first sentence gets 1.0, last gets 0.1
        scores = np.linspace(1.0, 0.1, n)
        return scores
    
    def score_sentences_length(self, sentences):
        """
        Score sentences based on length (prefer medium-length sentences)
        
        Args:
            sentences: List of sentences
            
        Returns:
            scores: Array of length scores
        """
        if len(sentences) == 0:
            return np.array([])
        
        lengths = np.array([len(s.split()) for s in sentences])
        
        # Prefer sentences with 10-30 words
        optimal_length = 20
        scores = 1.0 - np.abs(lengths - optimal_length) / optimal_length
        scores = np.clip(scores, 0.1, 1.0)
        
        return scores
    
    def summarize(self, text, method='tfidf', combine_scores=True):
        """
        Generate extractive summary
        
        Args:
            text: Input text to summarize
            method: 'tfidf', 'position', 'length', or 'combined'
            combine_scores: If True, combine multiple scoring methods
            
        Returns:
            summary: Extracted summary text
            selected_sentences: List of selected sentences with scores
        """
        # Preprocess
        text = self.preprocess_text(text)
        
        # Split into sentences
        sentences = self.split_sentences(text)
        
        if len(sentences) == 0:
            return "", []
        
        if len(sentences) <= self.num_sentences:
            return text, [(s, 1.0, i) for i, s in enumerate(sentences)]
        
        # Score sentences
        if method == 'tfidf' or combine_scores:
            tfidf_scores = self.score_sentences_tfidf(sentences)
        else:
            tfidf_scores = np.zeros(len(sentences))
        
        if method == 'position' or combine_scores:
            position_scores = self.score_sentences_position(sentences)
        else:
            position_scores = np.zeros(len(sentences))
        
        if method == 'length' or combine_scores:
            length_scores = self.score_sentences_length(sentences)
        else:
            length_scores = np.zeros(len(sentences))
        
        # Combine scores
        if combine_scores:
            # Weighted combination
            final_scores = (
                0.6 * tfidf_scores +
                0.3 * position_scores +
                0.1 * length_scores
            )
        else:
            if method == 'tfidf':
                final_scores = tfidf_scores
            elif method == 'position':
                final_scores = position_scores
            elif method == 'length':
                final_scores = length_scores
            else:
                final_scores = tfidf_scores
        
        # Select top N sentences
        top_indices = final_scores.argsort()[-self.num_sentences:][::-1]
        
        # Sort by original order to maintain coherence
        top_indices_sorted = sorted(top_indices)
        
        # Build summary
        selected_sentences = [
            (sentences[i], final_scores[i], i) 
            for i in top_indices_sorted
        ]
        
        summary = ' '.join([sentences[i] for i in top_indices_sorted])
        
        return summary, selected_sentences
    
    def summarize_with_details(self, text):
        """
        Generate summary with detailed scoring information
        
        Returns:
            dict with summary, scores, and metadata
        """
        summary, selected = self.summarize(text, combine_scores=True)
        
        sentences = self.split_sentences(text)
        
        result = {
            'original_text': text,
            'summary': summary,
            'original_sentences': len(sentences),
            'summary_sentences': len(selected),
            'compression_ratio': len(summary) / len(text) if len(text) > 0 else 0,
            'selected_sentences': [
                {
                    'text': sent,
                    'score': float(score),
                    'position': pos
                }
                for sent, score, pos in selected
            ]
        }
        
        return result


def demo_summarization():
    """Demo summarization dengan dataset film"""
    import glob
    import os
    
    print("="*80)
    print("EXTRACTIVE SUMMARIZATION DEMO")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    documents = []
    doc_ids = []
    
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    
    for filepath in glob.glob(os.path.join(dataset_path, '*.txt'))[:3]:  # First 3 files
        file_name = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if '->' in line and i < 5:  # First 5 from each file
                    q, a = line.split('->')
                    documents.append(a.strip())
                    doc_ids.append(f"{file_name}_{i}")
    
    print(f"Loaded {len(documents)} documents for demo")
    
    # Initialize summarizer
    summarizer = ExtractiveSummarizer(num_sentences=3)
    
    # Summarize each document
    print("\n" + "="*80)
    print("SUMMARIES")
    print("="*80 + "\n")
    
    for doc_id, doc in zip(doc_ids[:5], documents[:5]):  # Show first 5
        print(f"Document: {doc_id}")
        print(f"Original ({len(doc)} chars):")
        print(f"  {doc[:200]}...")
        print()
        
        result = summarizer.summarize_with_details(doc)
        
        print(f"Summary ({len(result['summary'])} chars, {result['compression_ratio']:.1%} of original):")
        print(f"  {result['summary']}")
        print()
        
        print("Selected sentences:")
        for sent_info in result['selected_sentences']:
            print(f"  [Score: {sent_info['score']:.3f}, Pos: {sent_info['position']}] {sent_info['text'][:100]}...")
        
        print("\n" + "-"*80 + "\n")
    
    # Compare methods
    print("="*80)
    print("COMPARING SUMMARIZATION METHODS")
    print("="*80 + "\n")
    
    test_doc = documents[0]
    
    for method in ['tfidf', 'position', 'length', 'combined']:
        summary, _ = summarizer.summarize(test_doc, method=method, 
                                         combine_scores=(method=='combined'))
        print(f"{method.upper()}:")
        print(f"  {summary[:200]}...")
        print()


if __name__ == "__main__":
    demo_summarization()
