# -*- coding: utf-8 -*-
"""
Feature Selection untuk Text Classification
Untuk Soal 4: Meningkatkan Efisiensi/Performansi
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd


class FeatureSelector:
    """
    Feature Selection untuk text classification
    Mendukung: DF threshold, Chi-square, Mutual Information, LSA
    """
    
    def __init__(self, method='chi2', n_features=100):
        """
        Initialize feature selector
        
        Args:
            method: 'df_threshold', 'chi2', 'mutual_info', 'lsa'
            n_features: Number of features to select
        """
        self.method = method
        self.n_features = n_features
        self.selector = None
        self.selected_features = None
        
    def fit(self, X, y=None):
        """
        Fit feature selector
        
        Args:
            X: Feature matrix (sparse or dense)
            y: Labels (required for chi2 and mutual_info)
        """
        if self.method == 'df_threshold':
            # Already handled in TfidfVectorizer with min_df/max_df
            self.selector = None
            self.selected_features = None
            
        elif self.method == 'chi2':
            if y is None:
                raise ValueError("Labels required for chi2 feature selection")
            self.selector = SelectKBest(chi2, k=self.n_features)
            self.selector.fit(X, y)
            self.selected_features = self.selector.get_support(indices=True)
            
        elif self.method == 'mutual_info':
            if y is None:
                raise ValueError("Labels required for mutual_info feature selection")
            self.selector = SelectKBest(mutual_info_classif, k=self.n_features)
            self.selector.fit(X, y)
            self.selected_features = self.selector.get_support(indices=True)
            
        elif self.method == 'lsa':
            self.selector = TruncatedSVD(n_components=self.n_features, random_state=42)
            self.selector.fit(X)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def transform(self, X):
        """Transform features"""
        if self.selector is None:
            return X
        return self.selector.transform(X)
    
    def fit_transform(self, X, y=None):
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names(self, original_feature_names):
        """Get selected feature names"""
        if self.selected_features is None:
            return original_feature_names
        return [original_feature_names[i] for i in self.selected_features]


def compare_feature_selection_methods(X_train, X_test, y_train, y_test, 
                                      feature_names, n_features_list=[50, 100, 200, 300]):
    """
    Compare different feature selection methods
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        feature_names: Original feature names
        n_features_list: List of n_features to test
        
    Returns:
        results: DataFrame with comparison results
    """
    methods = ['baseline', 'chi2', 'mutual_info', 'lsa']
    results = []
    
    print("\n" + "="*80)
    print("FEATURE SELECTION COMPARISON")
    print("="*80 + "\n")
    
    # Baseline (no feature selection)
    print("Testing BASELINE (all features)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    f1_baseline = f1_score(y_test, y_pred, average='macro')
    
    results.append({
        'method': 'baseline',
        'n_features': X_train.shape[1],
        'macro_f1': f1_baseline,
        'vocab_size': X_train.shape[1]
    })
    
    print(f"  Vocab size: {X_train.shape[1]}")
    print(f"  Macro F1: {f1_baseline:.4f}\n")
    
    # Test each method with different n_features
    for method in ['chi2', 'mutual_info', 'lsa']:
        for n_feat in n_features_list:
            if n_feat >= X_train.shape[1]:
                continue
                
            print(f"Testing {method.upper()} with {n_feat} features...")
            
            # Select features
            selector = FeatureSelector(method=method, n_features=n_feat)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Train classifier
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train_selected, y_train)
            y_pred = knn.predict(X_test_selected)
            f1 = f1_score(y_test, y_pred, average='macro')
            
            results.append({
                'method': method,
                'n_features': n_feat,
                'macro_f1': f1,
                'vocab_size': n_feat
            })
            
            print(f"  Vocab size: {n_feat}")
            print(f"  Macro F1: {f1:.4f}")
            print(f"  F1 change: {f1 - f1_baseline:+.4f}\n")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Visualize
    _plot_feature_selection_comparison(df_results)
    
    return df_results


def _plot_feature_selection_comparison(df_results):
    """Plot feature selection comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: F1 score vs number of features
    for method in df_results['method'].unique():
        if method == 'baseline':
            continue
        method_data = df_results[df_results['method'] == method]
        ax1.plot(method_data['n_features'], method_data['macro_f1'], 
                marker='o', label=method.upper(), linewidth=2)
    
    # Add baseline
    baseline_f1 = df_results[df_results['method'] == 'baseline']['macro_f1'].values[0]
    ax1.axhline(y=baseline_f1, color='red', linestyle='--', 
               label='Baseline (all features)', linewidth=2)
    
    ax1.set_xlabel('Number of Features', fontsize=12)
    ax1.set_ylabel('Macro F1 Score', fontsize=12)
    ax1.set_title('Feature Selection: F1 Score vs Number of Features', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bar chart comparison
    methods_to_compare = df_results[df_results['method'] != 'baseline'].groupby('method')['macro_f1'].max()
    baseline_f1 = df_results[df_results['method'] == 'baseline']['macro_f1'].values[0]
    
    x = range(len(methods_to_compare) + 1)
    values = [baseline_f1] + list(methods_to_compare.values)
    labels = ['Baseline'] + [m.upper() for m in methods_to_compare.index]
    
    bars = ax2.bar(x, values, color=['red'] + ['blue']*len(methods_to_compare))
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('Best Macro F1 Score', fontsize=12)
    ax2.set_title('Feature Selection Methods Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('feature_selection_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVisualisasi disimpan: feature_selection_comparison.png")
    plt.close()


def demo_feature_selection():
    """Demo feature selection dengan dataset"""
    import glob
    import os
    from sklearn.preprocessing import LabelEncoder
    
    print("="*80)
    print("FEATURE SELECTION DEMO")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    documents = []
    labels = []
    
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    
    for filepath in glob.glob(os.path.join(dataset_path, '*.txt')):
        file_name = os.path.basename(filepath).replace('.txt', '')
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if '->' in line:
                    q, a = line.split('->')
                    documents.append(a.strip())
                    labels.append(file_name)
    
    print(f"Loaded {len(documents)} documents")
    print(f"Number of categories: {len(set(labels))}")
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Vectorize
    print("\nVectorizing documents...")
    vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8)
    X = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"Original vocabulary size: {len(feature_names)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compare methods
    results = compare_feature_selection_methods(
        X_train, X_test, y_train, y_test, 
        feature_names, 
        n_features_list=[50, 100, 200, 300, 500]
    )
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    print(results.to_string(index=False))
    
    # Save results
    results.to_csv('feature_selection_results.csv', index=False)
    print("\nResults saved to: feature_selection_results.csv")


if __name__ == "__main__":
    demo_feature_selection()
