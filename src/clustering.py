# -*- coding: utf-8 -*-
"""
Document Clustering menggunakan K-Means
Untuk Soal 3: Clustering Dokumen Teks
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from collections import Counter
import pandas as pd
import os
import glob


class DocumentClustering:
    """
    Clustering dokumen menggunakan K-Means dengan evaluasi silhouette score
    """
    
    def __init__(self, documents, doc_ids=None):
        """
        Initialize clustering
        
        Args:
            documents: List of document texts
            doc_ids: Optional list of document IDs
        """
        self.documents = documents
        self.doc_ids = doc_ids if doc_ids else [f"Doc_{i}" for i in range(len(documents))]
        self.vectorizer = None
        self.tfidf_matrix = None
        self.kmeans_models = {}
        self.silhouette_scores = {}
        
    def vectorize_documents(self, max_features=500):
        """
        Vectorize documents using TF-IDF
        
        Args:
            max_features: Maximum number of features to keep
        """
        print(f"Vectorizing {len(self.documents)} documents...")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
    def find_optimal_k(self, k_range=range(3, 9), method='silhouette'):
        """
        Find optimal number of clusters using silhouette score
        
        Args:
            k_range: Range of k values to test
            method: 'silhouette' or 'elbow'
            
        Returns:
            optimal_k: Best k value
        """
        if self.tfidf_matrix is None:
            raise ValueError("Documents not vectorized. Call vectorize_documents() first.")
        
        print(f"\nTesting k values: {list(k_range)}")
        silhouette_scores = []
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.tfidf_matrix)
            
            # Silhouette score
            score = silhouette_score(self.tfidf_matrix, labels)
            silhouette_scores.append(score)
            self.silhouette_scores[k] = score
            
            # Inertia (for elbow method)
            inertias.append(kmeans.inertia_)
            
            print(f"k={k}: Silhouette Score = {score:.4f}, Inertia = {kmeans.inertia_:.2f}")
        
        # Find optimal k (highest silhouette score)
        optimal_k = list(k_range)[np.argmax(silhouette_scores)]
        print(f"\nOptimal k: {optimal_k} (Silhouette Score: {max(silhouette_scores):.4f})")
        
        # Visualize
        self._plot_evaluation(k_range, silhouette_scores, inertias)
        
        return optimal_k
    
    def _plot_evaluation(self, k_range, silhouette_scores, inertias):
        """Plot silhouette scores and elbow curve"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Silhouette scores
        ax1.plot(list(k_range), silhouette_scores, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Silhouette Score', fontsize=12)
        ax1.set_title('Silhouette Score vs Number of Clusters', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Mark optimal k
        optimal_idx = np.argmax(silhouette_scores)
        optimal_k = list(k_range)[optimal_idx]
        ax1.plot(optimal_k, silhouette_scores[optimal_idx], 'r*', markersize=20, 
                label=f'Optimal k={optimal_k}')
        ax1.legend()
        
        # Elbow curve
        ax2.plot(list(k_range), inertias, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
        ax2.set_title('Elbow Method', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('clustering_evaluation.png', dpi=300, bbox_inches='tight')
        print("\nVisualisasi disimpan: clustering_evaluation.png")
        plt.close()
    
    def cluster_documents(self, k):
        """
        Cluster documents with specified k
        
        Args:
            k: Number of clusters
            
        Returns:
            labels: Cluster labels for each document
        """
        if self.tfidf_matrix is None:
            raise ValueError("Documents not vectorized. Call vectorize_documents() first.")
        
        print(f"\nClustering with k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.tfidf_matrix)
        
        self.kmeans_models[k] = kmeans
        
        # Print cluster distribution
        cluster_counts = Counter(labels)
        print(f"\nCluster distribution:")
        for cluster_id in sorted(cluster_counts.keys()):
            print(f"  Cluster {cluster_id}: {cluster_counts[cluster_id]} documents")
        
        return labels
    
    def get_top_terms_per_cluster(self, k, top_n=10):
        """
        Get top N terms for each cluster
        
        Args:
            k: Number of clusters
            top_n: Number of top terms to return
            
        Returns:
            dict: {cluster_id: [(term, score), ...]}
        """
        if k not in self.kmeans_models:
            raise ValueError(f"No model found for k={k}. Run cluster_documents() first.")
        
        kmeans = self.kmeans_models[k]
        feature_names = self.vectorizer.get_feature_names_out()
        
        cluster_terms = {}
        
        print(f"\n{'='*80}")
        print(f"TOP {top_n} TERMS PER CLUSTER (k={k})")
        print(f"{'='*80}\n")
        
        for cluster_id in range(k):
            # Get centroid for this cluster
            centroid = kmeans.cluster_centers_[cluster_id]
            
            # Get top N terms
            top_indices = centroid.argsort()[-top_n:][::-1]
            top_terms = [(feature_names[i], centroid[i]) for i in top_indices]
            
            cluster_terms[cluster_id] = top_terms
            
            # Print
            print(f"Cluster {cluster_id}:")
            for term, score in top_terms:
                print(f"  - {term}: {score:.4f}")
            print()
        
        return cluster_terms
    
    def get_closest_documents_to_centroid(self, k, n_docs=5):
        """
        Get N closest documents to each cluster centroid
        
        Args:
            k: Number of clusters
            n_docs: Number of documents to return per cluster
            
        Returns:
            dict: {cluster_id: [(doc_id, distance), ...]}
        """
        if k not in self.kmeans_models:
            raise ValueError(f"No model found for k={k}. Run cluster_documents() first.")
        
        kmeans = self.kmeans_models[k]
        labels = kmeans.labels_
        
        cluster_docs = {}
        
        print(f"\n{'='*80}")
        print(f"TOP {n_docs} DOCUMENTS CLOSEST TO CENTROID (k={k})")
        print(f"{'='*80}\n")
        
        for cluster_id in range(k):
            # Get documents in this cluster
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            # Calculate distances to centroid
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = []
            
            for idx in cluster_indices:
                doc_vector = self.tfidf_matrix[idx].toarray().flatten()
                distance = np.linalg.norm(doc_vector - centroid)
                distances.append((idx, distance))
            
            # Sort by distance (ascending)
            distances.sort(key=lambda x: x[1])
            
            # Get top N
            top_docs = distances[:n_docs]
            cluster_docs[cluster_id] = top_docs
            
            # Print
            print(f"Cluster {cluster_id}:")
            for idx, dist in top_docs:
                doc_preview = self.documents[idx][:100].replace('\n', ' ')
                print(f"  - {self.doc_ids[idx]}: {doc_preview}... (distance: {dist:.4f})")
            print()
        
        return cluster_docs
    
    def interpret_clusters(self, k, cluster_names=None):
        """
        Interpret and name clusters based on top terms
        
        Args:
            k: Number of clusters
            cluster_names: Optional dict of {cluster_id: name}
            
        Returns:
            dict: {cluster_id: interpretation}
        """
        if cluster_names is None:
            # Auto-generate names based on top terms
            cluster_terms = self.get_top_terms_per_cluster(k, top_n=3)
            cluster_names = {}
            
            for cluster_id, terms in cluster_terms.items():
                top_3_terms = [term for term, _ in terms[:3]]
                cluster_names[cluster_id] = " + ".join(top_3_terms)
        
        print(f"\n{'='*80}")
        print(f"CLUSTER INTERPRETATIONS (k={k})")
        print(f"{'='*80}\n")
        
        for cluster_id, name in cluster_names.items():
            print(f"Cluster {cluster_id}: {name}")
        
        return cluster_names
    
    def evaluate_clustering(self, k, true_labels=None):
        """
        Evaluate clustering quality
        
        Args:
            k: Number of clusters
            true_labels: Optional ground truth labels for purity/NMI
            
        Returns:
            dict: Evaluation metrics
        """
        if k not in self.kmeans_models:
            raise ValueError(f"No model found for k={k}. Run cluster_documents() first.")
        
        kmeans = self.kmeans_models[k]
        labels = kmeans.labels_
        
        # Silhouette score
        silhouette = silhouette_score(self.tfidf_matrix, labels)
        
        metrics = {
            'silhouette_score': silhouette,
            'inertia': kmeans.inertia_,
            'n_iter': kmeans.n_iter_
        }
        
        # If true labels provided, calculate purity and NMI
        if true_labels is not None:
            from sklearn.metrics import normalized_mutual_info_score
            
            # Purity
            purity = self._calculate_purity(labels, true_labels)
            metrics['purity'] = purity
            
            # NMI
            nmi = normalized_mutual_info_score(true_labels, labels)
            metrics['nmi'] = nmi
        
        print(f"\n{'='*80}")
        print(f"CLUSTERING EVALUATION METRICS (k={k})")
        print(f"{'='*80}\n")
        
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def _calculate_purity(self, cluster_labels, true_labels):
        """Calculate purity score"""
        from scipy.stats import mode
        
        total_correct = 0
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_true_labels = true_labels[cluster_mask]
            
            if len(cluster_true_labels) > 0:
                most_common = mode(cluster_true_labels, keepdims=True)[0][0]
                total_correct += np.sum(cluster_true_labels == most_common)
        
        purity = total_correct / len(cluster_labels)
        return purity
    
    def visualize_clusters(self, k, method='tsne'):
        """
        Visualize clusters in 2D using dimensionality reduction
        
        Args:
            k: Number of clusters
            method: 'tsne' or 'pca'
        """
        if k not in self.kmeans_models:
            raise ValueError(f"No model found for k={k}. Run cluster_documents() first.")
        
        labels = self.kmeans_models[k].labels_
        
        # Dimensionality reduction
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        else:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
        
        coords = reducer.fit_transform(self.tfidf_matrix.toarray())
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='viridis', 
                            s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
        plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
        plt.title(f'Document Clustering Visualization (k={k}, {method.upper()})', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'clustering_visualization_{method}.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualisasi disimpan: clustering_visualization_{method}.png")
        plt.close()


def main():
    """Demo clustering dengan dataset film"""
    print("="*80)
    print("DOCUMENT CLUSTERING - K-MEANS")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    documents = []
    doc_ids = []
    
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    
    for filepath in glob.glob(os.path.join(dataset_path, '*.txt')):
        file_name = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if '->' in line:
                    q, a = line.split('->')
                    documents.append(a.strip())
                    doc_ids.append(f"{file_name}_{i}")
    
    print(f"Loaded {len(documents)} documents")
    
    # Initialize clustering
    clustering = DocumentClustering(documents, doc_ids)
    
    # Vectorize
    clustering.vectorize_documents(max_features=500)
    
    # Find optimal k
    optimal_k = clustering.find_optimal_k(k_range=range(3, 9))
    
    # Cluster with optimal k
    labels = clustering.cluster_documents(optimal_k)
    
    # Get top terms per cluster
    cluster_terms = clustering.get_top_terms_per_cluster(optimal_k, top_n=10)
    
    # Get closest documents to centroid
    cluster_docs = clustering.get_closest_documents_to_centroid(optimal_k, n_docs=5)
    
    # Interpret clusters
    cluster_names = clustering.interpret_clusters(optimal_k)
    
    # Evaluate
    metrics = clustering.evaluate_clustering(optimal_k)
    
    # Visualize
    clustering.visualize_clusters(optimal_k, method='tsne')
    clustering.visualize_clusters(optimal_k, method='pca')
    
    print("\n" + "="*80)
    print("CLUSTERING COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
