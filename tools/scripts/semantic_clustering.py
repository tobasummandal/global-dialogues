#!/usr/bin/env python3
"""
Semantic Clustering Analysis for Global Dialogues

This script performs k-means clustering on response embeddings to discover
natural groupings and themes in participant responses.

Usage:
    python semantic_clustering.py --gd 3 --clusters 10
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import os
import argparse
import warnings
from datetime import datetime
import uuid
from collections import Counter

# Configuration
EXPECTED_EMBEDDING_DIM = 1024
DEFAULT_N_CLUSTERS = 8
RANDOM_STATE = 42

# Column names
EMBEDDING_COLUMN = 'embedding'
TEXT_COLUMN = 'English Responses'
QUESTION_ID_COLUMN = 'Question ID'
QUESTION_TEXT_COLUMN = 'Question'
PARTICIPANT_ID_COLUMN = 'Participant ID'

def load_data_with_embeddings(file_path):
    """Load and validate embeddings data."""
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        print("Please download embeddings first: make download-embeddings GD=<N>")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        
        if not isinstance(data_list, list) or not data_list:
            print(f"Error: Expected non-empty list in {file_path}")
            return None
        
        # Combine all data items into single DataFrame
        df_list = []
        for item in data_list:
            df_part = pd.DataFrame(item)
            df_list.append(df_part)
        
        combined_df = pd.concat(df_list, ignore_index=True)
        
        if EMBEDDING_COLUMN not in combined_df.columns:
            print(f"Error: No '{EMBEDDING_COLUMN}' column found")
            return None
        
        print(f"Loaded {len(combined_df)} responses with embeddings")
        return combined_df
    
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

def validate_and_prepare_embeddings(df):
    """Validate embeddings and prepare matrix for clustering."""
    print("Validating embeddings...")
    
    # Extract embeddings
    embeddings_list = df[EMBEDDING_COLUMN].tolist()
    
    # Validate each embedding
    valid_embeddings = []
    valid_indices = []
    
    for idx, emb in enumerate(embeddings_list):
        if (isinstance(emb, list) and 
            len(emb) == EXPECTED_EMBEDDING_DIM and
            all(np.isfinite(val) for val in emb) and
            not all(val == 0 for val in emb)):
            
            valid_embeddings.append(emb)
            valid_indices.append(idx)
    
    if not valid_embeddings:
        print("Error: No valid embeddings found")
        return None, None
    
    print(f"Found {len(valid_embeddings)} valid embeddings out of {len(embeddings_list)}")
    
    # Create embeddings matrix
    embeddings_matrix = np.array(valid_embeddings)
    
    # Filter DataFrame to valid rows only
    valid_df = df.iloc[valid_indices].copy().reset_index(drop=True)
    
    return embeddings_matrix, valid_df

def determine_optimal_clusters(embeddings_matrix, max_k=15):
    """Use elbow method and silhouette analysis to find optimal number of clusters."""
    print("Determining optimal number of clusters...")
    
    # Limit analysis if we have too few samples
    n_samples = embeddings_matrix.shape[0]
    max_k = min(max_k, n_samples // 10, 20)  # At least 10 samples per cluster
    
    if max_k < 2:
        print("Too few samples for clustering analysis")
        return DEFAULT_N_CLUSTERS
    
    k_range = range(2, max_k + 1)
    inertias = []
    silhouette_scores = []
    
    print(f"Testing k from 2 to {max_k}...")
    
    for k in k_range:
        # Fit k-means
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_matrix)
        
        # Calculate metrics
        inertias.append(kmeans.inertia_)
        
        # Silhouette score (skip if too few samples)
        if n_samples > k * 2:
            sil_score = silhouette_score(embeddings_matrix, cluster_labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(0)
        
        print(f"  k={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={silhouette_scores[-1]:.3f}")
    
    # Find optimal k using silhouette score
    if silhouette_scores:
        optimal_k = k_range[np.argmax(silhouette_scores)]
        max_silhouette = max(silhouette_scores)
        print(f"Optimal k based on silhouette score: {optimal_k} (score: {max_silhouette:.3f})")
    else:
        optimal_k = DEFAULT_N_CLUSTERS
        print(f"Using default k: {optimal_k}")
    
    return optimal_k

def perform_clustering(embeddings_matrix, n_clusters):
    """Perform k-means clustering on embeddings."""
    print(f"Performing k-means clustering with {n_clusters} clusters...")
    
    # Fit k-means
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=RANDOM_STATE, 
        n_init=10,
        max_iter=300
    )
    
    cluster_labels = kmeans.fit_predict(embeddings_matrix)
    
    # Calculate clustering metrics
    inertia = kmeans.inertia_
    if len(set(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(embeddings_matrix, cluster_labels)
    else:
        silhouette_avg = 0
    
    print(f"Clustering complete:")
    print(f"  Inertia: {inertia:.0f}")
    print(f"  Silhouette score: {silhouette_avg:.3f}")
    print(f"  Cluster distribution: {dict(Counter(cluster_labels))}")
    
    return kmeans, cluster_labels

def analyze_clusters(df, cluster_labels):
    """Analyze the content and characteristics of each cluster."""
    print("\nAnalyzing cluster content...")
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    cluster_analysis = {}
    
    for cluster_id in sorted(set(cluster_labels)):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        
        analysis = {
            'size': len(cluster_data),
            'responses': cluster_data[TEXT_COLUMN].tolist()[:10],  # Sample responses
            'questions': cluster_data[QUESTION_ID_COLUMN].value_counts().head(5).to_dict() if QUESTION_ID_COLUMN in df.columns else {},
            'sample_text': cluster_data[TEXT_COLUMN].head(3).tolist()
        }
        
        cluster_analysis[cluster_id] = analysis
        
        print(f"\nCluster {cluster_id} ({analysis['size']} responses):")
        print(f"  Sample responses:")
        for i, text in enumerate(analysis['sample_text']):
            print(f"    {i+1}. {text[:100]}...")
    
    return df_clustered, cluster_analysis

def create_visualizations(embeddings_matrix, cluster_labels, output_dir):
    """Create visualizations of the clustering results."""
    print("Creating visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    
    # 1. PCA Visualization (2D)
    print("  Creating PCA visualization...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_result = pca.fit_transform(embeddings_matrix)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'K-Means Clustering Results (PCA)\n{len(set(cluster_labels))} Clusters, {len(embeddings_matrix)} Responses')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.grid(True, alpha=0.3)
    
    # Add cluster centers
    kmeans_pca = KMeans(n_clusters=len(set(cluster_labels)), random_state=RANDOM_STATE)
    kmeans_pca.fit(pca_result)
    centers = kmeans_pca.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_pca.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. t-SNE Visualization (if we have enough samples)
    if len(embeddings_matrix) > 50:
        print("  Creating t-SNE visualization...")
        
        # Use subset for t-SNE if dataset is very large
        if len(embeddings_matrix) > 5000:
            indices = np.random.choice(len(embeddings_matrix), 5000, replace=False)
            tsne_embeddings = embeddings_matrix[indices]
            tsne_labels = cluster_labels[indices]
        else:
            tsne_embeddings = embeddings_matrix
            tsne_labels = cluster_labels
        
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, len(tsne_embeddings)//4))
        tsne_result = tsne.fit_transform(tsne_embeddings)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                             c=tsne_labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'K-Means Clustering Results (t-SNE)\n{len(set(cluster_labels))} Clusters')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'clustering_tsne.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Cluster size distribution
    plt.figure(figsize=(10, 6))
    cluster_counts = Counter(cluster_labels)
    clusters, counts = zip(*sorted(cluster_counts.items()))
    
    plt.bar(clusters, counts, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Responses')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        plt.text(clusters[i], count + max(counts)*0.01, str(count), 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualizations saved to {output_dir}")

def save_results(df_clustered, cluster_analysis, output_dir):
    """Save clustering results to CSV files."""
    print("Saving results...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Timestamp for files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())[:8]
    
    # 1. Full results with cluster assignments
    results_columns = [TEXT_COLUMN, QUESTION_ID_COLUMN, PARTICIPANT_ID_COLUMN, 'cluster']
    if QUESTION_TEXT_COLUMN in df_clustered.columns:
        results_columns.insert(-1, QUESTION_TEXT_COLUMN)
    
    existing_columns = [col for col in results_columns if col in df_clustered.columns]
    
    df_clustered[existing_columns].to_csv(
        os.path.join(output_dir, f'clustering_results_{timestamp}.csv'),
        index=False, encoding='utf-8'
    )
    
    # 2. Cluster summary
    summary_data = []
    for cluster_id, analysis in cluster_analysis.items():
        summary_data.append({
            'cluster_id': cluster_id,
            'size': analysis['size'],
            'percentage': analysis['size'] / len(df_clustered) * 100,
            'sample_response_1': analysis['sample_text'][0] if analysis['sample_text'] else '',
            'sample_response_2': analysis['sample_text'][1] if len(analysis['sample_text']) > 1 else '',
            'sample_response_3': analysis['sample_text'][2] if len(analysis['sample_text']) > 2 else '',
            'run_id': run_id,
            'timestamp': timestamp
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(
        os.path.join(output_dir, f'cluster_summary_{timestamp}.csv'),
        index=False, encoding='utf-8'
    )
    
    print(f"Results saved to {output_dir}")
    return run_id, timestamp

def get_data_paths(gd_number):
    """Get file paths for the specified GD number."""
    data_file = os.path.join("Data", f"GD{gd_number}", f"GD{gd_number}_embeddings.json")
    output_dir = os.path.join("analysis_output", f"GD{gd_number}", "semantic_clustering")
    return data_file, output_dir

def main():
    parser = argparse.ArgumentParser(
        description="Perform k-means clustering analysis on Global Dialogues embeddings"
    )
    parser.add_argument('--gd', type=int, required=True,
                       help='Global Dialogue number (1, 2, 3, or 4)')
    parser.add_argument('--clusters', type=int, default=None,
                       help='Number of clusters (if not specified, will determine optimal)')
    parser.add_argument('--max-k', type=int, default=15,
                       help='Maximum number of clusters to test for optimization')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Get file paths
    data_file, output_dir = get_data_paths(args.gd)
    
    print(f"=== Semantic Clustering Analysis for GD{args.gd} ===")
    print(f"Data file: {data_file}")
    print(f"Output directory: {output_dir}")
    
    # Load and validate data
    df = load_data_with_embeddings(data_file)
    if df is None:
        return
    
    embeddings_matrix, valid_df = validate_and_prepare_embeddings(df)
    if embeddings_matrix is None:
        return
    
    # Determine number of clusters
    if args.clusters is None:
        n_clusters = determine_optimal_clusters(embeddings_matrix, args.max_k)
    else:
        n_clusters = args.clusters
    
    # Perform clustering
    kmeans_model, cluster_labels = perform_clustering(embeddings_matrix, n_clusters)
    
    # Analyze clusters
    df_clustered, cluster_analysis = analyze_clusters(valid_df, cluster_labels)
    
    # Create visualizations
    if not args.skip_viz:
        create_visualizations(embeddings_matrix, cluster_labels, output_dir)
    
    # Save results
    run_id, timestamp = save_results(df_clustered, cluster_analysis, output_dir)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Found {n_clusters} clusters in {len(valid_df)} responses")
    print(f"Results saved with run ID: {run_id}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main() 