#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Search with Summarization
Untuk Soal 4: search_plus.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from search_engine import SearchEngine
from summarization import ExtractiveSummarizer
import glob


def load_dataset():
    """Load dataset from files"""
    questions = []
    answers = []
    file_names = []
    
    dataset_path = 'dataset'
    
    for filepath in glob.glob(os.path.join(dataset_path, '*.txt')):
        file_name = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if '->' in line:
                    q, a = line.split('->')
                    questions.append(q.strip("- ").lower().strip())
                    answers.append(a.strip())
                    file_names.append(file_name)
    
    return questions, answers, file_names


def search_with_summary(query, search_engine, summarizer, top_k=5):
    """
    Search and return results with summaries
    
    Args:
        query: Search query
        search_engine: SearchEngine instance
        summarizer: ExtractiveSummarizer instance
        top_k: Number of results to return
    """
    print("="*80)
    print(f"SEARCH RESULTS FOR: {query}")
    print("="*80 + "\n")
    
    # Search
    results = search_engine.search(query, model='vsm', k=top_k)
    
    if not results:
        print("No results found.")
        return
    
    print(f"Found {len(results)} results:\n")
    
    # Display results with summaries
    for rank, (doc_id, score) in enumerate(results, 1):
        document = search_engine.docs[doc_id]
        question = search_engine.questions[doc_id]
        
        # Generate summary
        summary_result = summarizer.summarize_with_details(document)
        
        print(f"{rank}. [Score: {score:.4f}]")
        print(f"   Question: {question}")
        print(f"   Source: Document #{doc_id}")
        print(f"\n   SUMMARY ({summary_result['summary_sentences']} sentences, "
              f"{summary_result['compression_ratio']:.1%} of original):")
        print(f"   {summary_result['summary']}")
        print(f"\n   FULL TEXT ({len(document)} chars):")
        print(f"   {document[:200]}...")
        print("\n" + "-"*80 + "\n")


def interactive_search():
    """Interactive search mode"""
    print("="*80)
    print("ENHANCED SEARCH WITH SUMMARIZATION")
    print("="*80)
    print("\nLoading dataset and initializing search engine...")
    
    # Load dataset
    questions, answers, file_names = load_dataset()
    print(f"✓ Loaded {len(questions)} documents")
    
    # Initialize search engine
    search_engine = SearchEngine(answers, questions)
    print("✓ Search engine initialized")
    
    # Initialize summarizer
    summarizer = ExtractiveSummarizer(num_sentences=3)
    print("✓ Summarizer initialized\n")
    
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for available commands\n")
    
    while True:
        try:
            query = input("Enter search query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if query.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Type any query to search")
                print("  - 'quit' or 'exit': Exit the program")
                print("  - 'help': Show this help message")
                print()
                continue
            
            # Search with summary
            search_with_summary(query, search_engine, summarizer, top_k=5)
            
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
        # Batch mode: search from command line
        query = ' '.join(sys.argv[1:])
        
        print("Loading dataset...")
        questions, answers, file_names = load_dataset()
        
        search_engine = SearchEngine(answers, questions)
        summarizer = ExtractiveSummarizer(num_sentences=3)
        
        search_with_summary(query, search_engine, summarizer, top_k=5)
    else:
        # Interactive mode
        interactive_search()


if __name__ == "__main__":
    main()
