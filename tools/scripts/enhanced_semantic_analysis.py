#!/usr/bin/env python3
"""
Enhanced Semantic Analysis for Global Dialogues

This script combines semantic clustering with PRI (Participant Reliability Index) 
analysis and Global Dialogues Indicators for comprehensive insights.

Features:
- Semantic clustering with PRI weighting
- Reliability-based filtering and analysis
- Indicator-specific analysis
- Enhanced visualizations with PRI integration
- Cross-dialogue comparison capabilities

Usage:
    python enhanced_semantic_analysis.py --gd 3 --include-pri --indicators-only
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os
import argparse
import warnings
from datetime import datetime
import uuid
from collections import Counter
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
EXPECTED_EMBEDDING_DIM = 1024
DEFAULT_N_CLUSTERS = 8
RANDOM_STATE = 42
PRI_THRESHOLD_LOW = 2.5  # PRI scale 1-5, filter below this
PRI_THRESHOLD_HIGH = 4.0  # Consider above this as "high quality"

# Column names
EMBEDDING_COLUMN = 'embedding'
TEXT_COLUMN = 'English Responses'
QUESTION_ID_COLUMN = 'Question ID'
QUESTION_TEXT_COLUMN = 'Question'
PARTICIPANT_ID_COLUMN = 'Participant ID'

def load_indicator_codesheet():
    """Load the indicator codesheet to identify indicator questions."""
    try:
        codesheet_path = "Data/Documentation/INDICATOR_CODESHEET.csv"
        if os.path.exists(codesheet_path):
            return pd.read_csv(codesheet_path)
        else:
            print(f"Warning: Indicator codesheet not found at {codesheet_path}")
            return None
    except Exception as e:
        print(f"Error loading indicator codesheet: {e}")
        return None

def load_pri_scores(gd_number):
    """Load PRI scores for the specified GD."""
    try:
        pri_path = f"analysis_output/GD{gd_number}/pri/GD{gd_number}_pri_scores.csv"
        if os.path.exists(pri_path):
            pri_df = pd.read_csv(pri_path)
            print(f"Loaded PRI scores for {len(pri_df)} participants")
            return pri_df
        else:
            print(f"Warning: PRI scores not found at {pri_path}")
            print("Run: python tools/scripts/calculate_pri.py {gd_number} to generate PRI scores")
            return None
    except Exception as e:
        print(f"Error loading PRI scores: {e}")
        return None

def load_data_with_embeddings(file_path):
    """Load and validate embeddings data."""
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        
        df_list = []
        for item in data_list:
            df_part = pd.DataFrame(item)
            df_list.append(df_part)
        
        combined_df = pd.concat(df_list, ignore_index=True)
        print(f"Loaded {len(combined_df)} responses with embeddings")
        return combined_df
    
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

def filter_by_indicators(df, codesheet_df):
    """Filter data to only include indicator questions."""
    if codesheet_df is None:
        print("No indicator codesheet available - using all questions")
        return df
    
    # The codesheet has question_text, we need to match it to actual questions
    # For now, we'll use a simpler approach - filter by question type for GD3+ indicators
    print("Filtering to indicator questions based on content patterns...")
    
    # Common indicator question patterns (these are the recurring "pulse" questions)
    indicator_patterns = [
        "artificial intelligence.*makes you feel",
        "how often.*noticed AI systems",
        "how often.*human interactions.*replaced.*automated",
        "how often.*expected to use an AI system at work",
        "how often.*used an AI system voluntarily at work", 
        "trust.*governments",
        "trust.*technology companies",
        "trust.*academic institutions",
        "trust.*news media",
        "AI.*societal impact",
        "AI.*personal impact",
        "automation.*affected.*job market",
        "importance.*human oversight",
        "importance.*transparency"
    ]
    
    # Filter questions that match indicator patterns
    import re
    indicator_mask = df[QUESTION_TEXT_COLUMN].str.contains(
        '|'.join(indicator_patterns), 
        case=False, 
        na=False, 
        regex=True
    ) if QUESTION_TEXT_COLUMN in df.columns else pd.Series([False] * len(df))
    
    # Also include questions that are Poll Single Select (many indicators are polls)
    if 'Question Type' in df.columns:
        poll_mask = df['Question Type'] == 'Poll Single Select'
        # Combine with pattern matching for a broader filter
        indicator_mask = indicator_mask | poll_mask
    
    indicator_df = df[indicator_mask]
    
    unique_questions = indicator_df[QUESTION_ID_COLUMN].nunique() if QUESTION_ID_COLUMN in indicator_df.columns else 0
    print(f"Filtered to {len(indicator_df)} responses from {unique_questions} potential indicator questions")
    
    # If we got very few results, fall back to all data
    if len(indicator_df) < 100:
        print("Too few indicator responses found, using all data instead")
        return df
    
    return indicator_df

def integrate_pri_scores(df, pri_df):
    """Integrate PRI scores with embeddings data."""
    if pri_df is None:
        print("No PRI scores available - proceeding without PRI integration")
        df['PRI_Score'] = np.nan
        df['PRI_Scale_1_5'] = np.nan
        df['Reliability_Category'] = 'Unknown'
        return df
    
    # Merge PRI scores
    df_with_pri = df.merge(
        pri_df[['Participant ID', 'PRI_Score', 'PRI_Scale_1_5']], 
        left_on=PARTICIPANT_ID_COLUMN, 
        right_on='Participant ID', 
        how='left'
    )
    
    # Create reliability categories
    def categorize_reliability(pri_score):
        if pd.isna(pri_score):
            return 'Unknown'
        elif pri_score >= PRI_THRESHOLD_HIGH:
            return 'High'
        elif pri_score >= PRI_THRESHOLD_LOW:
            return 'Medium'
        else:
            return 'Low'
    
    df_with_pri['Reliability_Category'] = df_with_pri['PRI_Scale_1_5'].apply(categorize_reliability)
    
    print(f"PRI integration complete:")
    print(f"  High reliability: {sum(df_with_pri['Reliability_Category'] == 'High')} responses")
    print(f"  Medium reliability: {sum(df_with_pri['Reliability_Category'] == 'Medium')} responses")
    print(f"  Low reliability: {sum(df_with_pri['Reliability_Category'] == 'Low')} responses")
    print(f"  Unknown reliability: {sum(df_with_pri['Reliability_Category'] == 'Unknown')} responses")
    
    return df_with_pri

def filter_by_reliability(df, min_pri_score=None):
    """Filter responses based on PRI reliability threshold."""
    if min_pri_score is None or 'PRI_Scale_1_5' not in df.columns:
        return df
    
    # Filter out low-reliability participants
    reliable_df = df[
        (df['PRI_Scale_1_5'] >= min_pri_score) | 
        (df['PRI_Scale_1_5'].isna())  # Keep unknown if no PRI available
    ]
    
    removed_count = len(df) - len(reliable_df)
    print(f"Filtered out {removed_count} low-reliability responses (PRI < {min_pri_score})")
    
    return reliable_df

def validate_and_prepare_embeddings(df):
    """Validate embeddings and prepare matrix for clustering."""
    print("Validating embeddings...")
    
    embeddings_list = df[EMBEDDING_COLUMN].tolist()
    valid_embeddings = []
    valid_indices = []
    
    for idx, emb in enumerate(embeddings_list):
        if (isinstance(emb, list) and 
            len(emb) == EXPECTED_EMBEDDING_DIM and
            all(np.isfinite(val) for val in emb) and
            not all(val == 0 for val in emb)):
            
            valid_embeddings.append(emb)
            valid_indices.append(idx)
    
    print(f"Found {len(valid_embeddings)} valid embeddings out of {len(embeddings_list)}")
    
    embeddings_matrix = np.array(valid_embeddings)
    valid_df = df.iloc[valid_indices].copy().reset_index(drop=True)
    
    return embeddings_matrix, valid_df

def perform_weighted_clustering(embeddings_matrix, valid_df, n_clusters, use_pri_weights=False):
    """Perform clustering with optional PRI weighting."""
    print(f"Performing k-means clustering with {n_clusters} clusters...")
    
    if use_pri_weights and 'PRI_Scale_1_5' in valid_df.columns:
        # Use PRI scores as sample weights (higher weight for more reliable participants)
        weights = valid_df['PRI_Scale_1_5'].fillna(valid_df['PRI_Scale_1_5'].mean())
        weights = np.clip(weights, 1.0, 5.0)  # Ensure weights are in reasonable range
        print("Using PRI-weighted clustering")
    else:
        weights = None
        print("Using standard (unweighted) clustering")
    
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=RANDOM_STATE, 
        n_init=10,
        max_iter=300
    )
    
    cluster_labels = kmeans.fit_predict(embeddings_matrix, sample_weight=weights)
    
    # Calculate metrics
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

def analyze_clusters_with_pri(df_clustered):
    """Enhanced cluster analysis including PRI statistics."""
    print("\nAnalyzing clusters with PRI integration...")
    
    cluster_analysis = {}
    
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        
        # Basic statistics
        analysis = {
            'size': len(cluster_data),
            'sample_text': cluster_data[TEXT_COLUMN].head(3).tolist()
        }
        
        # PRI statistics if available
        if 'PRI_Scale_1_5' in cluster_data.columns and cluster_data['PRI_Scale_1_5'].notna().any():
            pri_data = cluster_data['PRI_Scale_1_5'].dropna()
            analysis.update({
                'mean_pri': pri_data.mean(),
                'median_pri': pri_data.median(),
                'pri_std': pri_data.std(),
                'high_reliability_pct': (pri_data >= PRI_THRESHOLD_HIGH).mean() * 100,
                'low_reliability_pct': (pri_data < PRI_THRESHOLD_LOW).mean() * 100
            })
        
        # Reliability category distribution
        if 'Reliability_Category' in cluster_data.columns:
            reliability_dist = cluster_data['Reliability_Category'].value_counts()
            analysis['reliability_distribution'] = reliability_dist.to_dict()
        
        cluster_analysis[cluster_id] = analysis
        
        # Print cluster summary
        print(f"\nCluster {cluster_id} ({analysis['size']} responses):")
        if 'mean_pri' in analysis:
            print(f"  Mean PRI: {analysis['mean_pri']:.2f} ± {analysis['pri_std']:.2f}")
            print(f"  High reliability: {analysis['high_reliability_pct']:.1f}%")
        
        print(f"  Sample responses:")
        for i, text in enumerate(analysis['sample_text']):
            print(f"    {i+1}. {text[:100]}...")
    
    return cluster_analysis

def create_enhanced_visualizations(embeddings_matrix, df_clustered, output_dir, gd_number):
    """Create comprehensive visualizations including PRI integration."""
    print("Creating enhanced visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("tab10")
    
    # 1. PCA with PRI coloring
    if 'PRI_Scale_1_5' in df_clustered.columns:
        create_pri_pca_visualization(embeddings_matrix, df_clustered, output_dir)
    
    # 2. Cluster quality analysis
    create_cluster_quality_plots(df_clustered, output_dir)
    
    # 3. PRI distribution by cluster
    if 'PRI_Scale_1_5' in df_clustered.columns:
        create_pri_cluster_analysis(df_clustered, output_dir)
    
    # 4. Interactive plotly visualizations
    create_interactive_visualizations(embeddings_matrix, df_clustered, output_dir, gd_number)
    
    print(f"Enhanced visualizations saved to {output_dir}")

def create_pri_pca_visualization(embeddings_matrix, df_clustered, output_dir):
    """Create PCA visualization colored by PRI scores."""
    print("  Creating PRI-enhanced PCA visualization...")
    
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_result = pca.fit_transform(embeddings_matrix)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Colored by cluster
    scatter1 = axes[0].scatter(pca_result[:, 0], pca_result[:, 1], 
                              c=df_clustered['cluster'], cmap='tab10', alpha=0.6)
    axes[0].set_title('PCA: Colored by Cluster')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    # Right plot: Colored by PRI
    pri_scores = df_clustered['PRI_Scale_1_5'].fillna(df_clustered['PRI_Scale_1_5'].mean())
    scatter2 = axes[1].scatter(pca_result[:, 0], pca_result[:, 1], 
                              c=pri_scores, cmap='viridis', alpha=0.6)
    axes[1].set_title('PCA: Colored by PRI Score')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter2, ax=axes[1], label='PRI Score (1-5)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_cluster_pri_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_cluster_quality_plots(df_clustered, output_dir):
    """Create cluster quality and distribution analysis plots."""
    print("  Creating cluster quality plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Cluster size distribution
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    axes[0,0].bar(cluster_counts.index, cluster_counts.values, alpha=0.7, color='skyblue')
    axes[0,0].set_title('Cluster Size Distribution')
    axes[0,0].set_xlabel('Cluster ID')
    axes[0,0].set_ylabel('Number of Responses')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add count labels
    for i, count in enumerate(cluster_counts.values):
        axes[0,0].text(cluster_counts.index[i], count + max(cluster_counts.values)*0.01, 
                      str(count), ha='center', va='bottom')
    
    # 2. Reliability category distribution
    if 'Reliability_Category' in df_clustered.columns:
        reliability_counts = df_clustered['Reliability_Category'].value_counts()
        colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red', 'Unknown': 'gray'}
        bar_colors = [colors.get(cat, 'blue') for cat in reliability_counts.index]
        
        axes[0,1].bar(reliability_counts.index, reliability_counts.values, 
                     color=bar_colors, alpha=0.7)
        axes[0,1].set_title('Overall Reliability Distribution')
        axes[0,1].set_xlabel('Reliability Category')
        axes[0,1].set_ylabel('Number of Responses')
        axes[0,1].tick_params(axis='x', rotation=45)
    else:
        axes[0,1].text(0.5, 0.5, 'No PRI data available', ha='center', va='center', 
                      transform=axes[0,1].transAxes)
        axes[0,1].set_title('Reliability Distribution')
    
    # 3. Questions per cluster
    if QUESTION_ID_COLUMN in df_clustered.columns:
        cluster_question_counts = df_clustered.groupby('cluster')[QUESTION_ID_COLUMN].nunique()
        axes[1,0].bar(cluster_question_counts.index, cluster_question_counts.values, 
                     alpha=0.7, color='lightcoral')
        axes[1,0].set_title('Unique Questions per Cluster')
        axes[1,0].set_xlabel('Cluster ID')
        axes[1,0].set_ylabel('Number of Unique Questions')
        axes[1,0].grid(True, alpha=0.3)
    
    # 4. Response length distribution by cluster
    if TEXT_COLUMN in df_clustered.columns:
        df_clustered['response_length'] = df_clustered[TEXT_COLUMN].str.len()
        
        cluster_lengths = []
        cluster_labels = []
        for cluster_id in sorted(df_clustered['cluster'].unique()):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            cluster_lengths.extend(cluster_data['response_length'].tolist())
            cluster_labels.extend([f'Cluster {cluster_id}'] * len(cluster_data))
        
        # Create boxplot
        unique_clusters = sorted(df_clustered['cluster'].unique())
        boxplot_data = [df_clustered[df_clustered['cluster'] == c]['response_length'].tolist() 
                       for c in unique_clusters]
        
        axes[1,1].boxplot(boxplot_data, labels=[f'C{c}' for c in unique_clusters])
        axes[1,1].set_title('Response Length Distribution by Cluster')
        axes[1,1].set_xlabel('Cluster')
        axes[1,1].set_ylabel('Response Length (characters)')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_quality_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_pri_cluster_analysis(df_clustered, output_dir):
    """Create detailed PRI vs cluster analysis."""
    print("  Creating PRI-cluster analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. PRI distribution by cluster (boxplot)
    clusters = sorted(df_clustered['cluster'].unique())
    pri_by_cluster = [df_clustered[df_clustered['cluster'] == c]['PRI_Scale_1_5'].dropna().tolist() 
                     for c in clusters]
    
    axes[0,0].boxplot(pri_by_cluster, labels=[f'C{c}' for c in clusters])
    axes[0,0].set_title('PRI Score Distribution by Cluster')
    axes[0,0].set_xlabel('Cluster')
    axes[0,0].set_ylabel('PRI Score (1-5)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add horizontal lines for thresholds
    axes[0,0].axhline(y=PRI_THRESHOLD_LOW, color='red', linestyle='--', alpha=0.7, label='Low threshold')
    axes[0,0].axhline(y=PRI_THRESHOLD_HIGH, color='green', linestyle='--', alpha=0.7, label='High threshold')
    axes[0,0].legend()
    
    # 2. Reliability category stacked bar by cluster
    reliability_cluster = pd.crosstab(df_clustered['cluster'], df_clustered['Reliability_Category'])
    reliability_cluster_pct = reliability_cluster.div(reliability_cluster.sum(axis=1), axis=0) * 100
    
    reliability_cluster_pct.plot(kind='bar', stacked=True, ax=axes[0,1], 
                               color={'High': 'green', 'Medium': 'orange', 'Low': 'red', 'Unknown': 'gray'})
    axes[0,1].set_title('Reliability Distribution by Cluster (%)')
    axes[0,1].set_xlabel('Cluster')
    axes[0,1].set_ylabel('Percentage')
    axes[0,1].legend(title='Reliability', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,1].tick_params(axis='x', rotation=0)
    
    # 3. Mean PRI by cluster
    mean_pri_by_cluster = df_clustered.groupby('cluster')['PRI_Scale_1_5'].mean()
    std_pri_by_cluster = df_clustered.groupby('cluster')['PRI_Scale_1_5'].std()
    
    axes[1,0].bar(mean_pri_by_cluster.index, mean_pri_by_cluster.values, 
                 yerr=std_pri_by_cluster.values, alpha=0.7, color='lightblue', capsize=5)
    axes[1,0].set_title('Mean PRI Score by Cluster')
    axes[1,0].set_xlabel('Cluster')
    axes[1,0].set_ylabel('Mean PRI Score')
    axes[1,0].grid(True, alpha=0.3)
    
    # Add threshold lines
    axes[1,0].axhline(y=PRI_THRESHOLD_LOW, color='red', linestyle='--', alpha=0.7)
    axes[1,0].axhline(y=PRI_THRESHOLD_HIGH, color='green', linestyle='--', alpha=0.7)
    
    # 4. PRI vs Response Length scatter
    if 'response_length' in df_clustered.columns:
        scatter = axes[1,1].scatter(df_clustered['response_length'], df_clustered['PRI_Scale_1_5'], 
                                   c=df_clustered['cluster'], cmap='tab10', alpha=0.6)
        axes[1,1].set_title('PRI Score vs Response Length')
        axes[1,1].set_xlabel('Response Length (characters)')
        axes[1,1].set_ylabel('PRI Score (1-5)')
        axes[1,1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1,1], label='Cluster')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pri_cluster_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_visualizations(embeddings_matrix, df_clustered, output_dir, gd_number):
    """Create interactive Plotly visualizations."""
    print("  Creating interactive visualizations...")
    
    # PCA for interactive plot
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_result = pca.fit_transform(embeddings_matrix)
    
    # Prepare data for plotting
    plot_df = df_clustered.copy()
    plot_df['PC1'] = pca_result[:, 0]
    plot_df['PC2'] = pca_result[:, 1]
    plot_df['text_preview'] = plot_df[TEXT_COLUMN].str[:100] + '...'
    
    # Handle NaN values in PRI scores for plotting
    size_column = None
    hover_data = []
    if 'PRI_Scale_1_5' in plot_df.columns:
        # Fill NaN values with median for size mapping
        plot_df['PRI_Size'] = plot_df['PRI_Scale_1_5'].fillna(plot_df['PRI_Scale_1_5'].median())
        size_column = 'PRI_Size'
        hover_data = ['Reliability_Category', 'PRI_Scale_1_5']
    
    # Create interactive scatter plot
    fig = px.scatter(
        plot_df, 
        x='PC1', 
        y='PC2', 
        color='cluster',
        size=size_column,
        hover_data=hover_data,
        hover_name='text_preview',
        title=f'Interactive Semantic Clustering Results - GD{gd_number}',
        labels={
            'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
            'cluster': 'Cluster',
            'PRI_Scale_1_5': 'PRI Score',
            'PRI_Size': 'PRI Score'
        }
    )
    
    fig.update_traces(marker=dict(line=dict(width=0.5, color='white')))
    fig.update_layout(
        width=1000,
        height=700,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Save interactive plot
    fig.write_html(os.path.join(output_dir, f'interactive_clustering_GD{gd_number}.html'))

def save_enhanced_results(df_clustered, cluster_analysis, output_dir, gd_number, run_metadata):
    """Save comprehensive results including PRI integration."""
    print("Saving enhanced results...")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())[:8]
    
    # 1. Full results with all columns
    results_columns = [TEXT_COLUMN, QUESTION_ID_COLUMN, PARTICIPANT_ID_COLUMN, 'cluster']
    
    # Add PRI columns if available
    pri_columns = ['PRI_Score', 'PRI_Scale_1_5', 'Reliability_Category']
    for col in pri_columns:
        if col in df_clustered.columns:
            results_columns.append(col)
    
    # Add other available columns
    optional_columns = [QUESTION_TEXT_COLUMN, 'Submitted By', 'Language']
    for col in optional_columns:
        if col in df_clustered.columns:
            results_columns.append(col)
    
    existing_columns = [col for col in results_columns if col in df_clustered.columns]
    
    # Save detailed results
    df_clustered[existing_columns].to_csv(
        os.path.join(output_dir, f'enhanced_clustering_results_{timestamp}.csv'),
        index=False, encoding='utf-8'
    )
    
    # 2. Enhanced cluster summary
    summary_data = []
    for cluster_id, analysis in cluster_analysis.items():
        summary_row = {
            'cluster_id': cluster_id,
            'size': analysis['size'],
            'percentage': analysis['size'] / len(df_clustered) * 100,
            'sample_response_1': analysis['sample_text'][0] if analysis['sample_text'] else '',
            'sample_response_2': analysis['sample_text'][1] if len(analysis['sample_text']) > 1 else '',
            'sample_response_3': analysis['sample_text'][2] if len(analysis['sample_text']) > 2 else '',
            'run_id': run_id,
            'timestamp': timestamp,
            'gd_number': gd_number
        }
        
        # Add PRI statistics if available
        if 'mean_pri' in analysis:
            summary_row.update({
                'mean_pri': analysis['mean_pri'],
                'median_pri': analysis['median_pri'],
                'pri_std': analysis['pri_std'],
                'high_reliability_pct': analysis['high_reliability_pct'],
                'low_reliability_pct': analysis['low_reliability_pct']
            })
        
        # Add reliability distribution
        if 'reliability_distribution' in analysis:
            for category, count in analysis['reliability_distribution'].items():
                summary_row[f'reliability_{category.lower()}'] = count
        
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(
        os.path.join(output_dir, f'enhanced_cluster_summary_{timestamp}.csv'),
        index=False, encoding='utf-8'
    )
    
    # 3. Analysis metadata
    metadata = {
        'run_id': run_id,
        'timestamp': timestamp,
        'gd_number': gd_number,
        'total_responses': len(df_clustered),
        'num_clusters': len(cluster_analysis),
        'pri_integrated': 'PRI_Scale_1_5' in df_clustered.columns,
        'indicators_only': run_metadata.get('indicators_only', False),
        'min_pri_threshold': run_metadata.get('min_pri_threshold', None),
        'weighted_clustering': run_metadata.get('weighted_clustering', False)
    }
    
    with open(os.path.join(output_dir, f'analysis_metadata_{timestamp}.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Enhanced results saved to {output_dir}")
    return run_id, timestamp

def get_data_paths(gd_number):
    """Get file paths for the specified GD number."""
    data_file = os.path.join("Data", f"GD{gd_number}", f"GD{gd_number}_embeddings.json")
    output_dir = os.path.join("analysis_output", f"GD{gd_number}", "enhanced_semantic_analysis")
    return data_file, output_dir

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced semantic clustering analysis with PRI integration"
    )
    parser.add_argument('--gd', type=int, required=True,
                       help='Global Dialogue number (1, 2, 3, or 4)')
    parser.add_argument('--clusters', type=int, default=None,
                       help='Number of clusters (if not specified, will determine optimal)')
    parser.add_argument('--include-pri', action='store_true',
                       help='Include PRI scores in analysis')
    parser.add_argument('--pri-weighted', action='store_true',
                       help='Use PRI scores to weight clustering algorithm')
    parser.add_argument('--min-pri', type=float, default=None,
                       help='Minimum PRI score threshold for filtering')
    parser.add_argument('--indicators-only', action='store_true',
                       help='Analyze only Global Dialogues Indicator questions')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Get file paths
    data_file, output_dir = get_data_paths(args.gd)
    
    print(f"=== Enhanced Semantic Analysis for GD{args.gd} ===")
    print(f"PRI Integration: {args.include_pri}")
    print(f"Indicators Only: {args.indicators_only}")
    print(f"PRI Weighted Clustering: {args.pri_weighted}")
    if args.min_pri:
        print(f"Minimum PRI Threshold: {args.min_pri}")
    
    # Load indicator codesheet
    codesheet_df = load_indicator_codesheet() if args.indicators_only else None
    
    # Load embeddings data
    df = load_data_with_embeddings(data_file)
    if df is None:
        return
    
    # Filter to indicators if requested
    if args.indicators_only:
        df = filter_by_indicators(df, codesheet_df)
    
    # Load and integrate PRI scores
    if args.include_pri:
        pri_df = load_pri_scores(args.gd)
        df = integrate_pri_scores(df, pri_df)
        
        # Filter by PRI if threshold specified
        if args.min_pri:
            df = filter_by_reliability(df, args.min_pri)
    
    # Validate embeddings
    embeddings_matrix, valid_df = validate_and_prepare_embeddings(df)
    if embeddings_matrix is None or len(embeddings_matrix) == 0:
        if args.indicators_only:
            print("No embeddings found for indicator questions (they may be Poll questions without embeddings)")
            print("Falling back to all questions with embeddings...")
            # Load all data again
            df = load_data_with_embeddings(data_file)
            if df is None:
                return
            # Re-integrate PRI
            if args.include_pri:
                df = integrate_pri_scores(df, pri_df)
                if args.min_pri:
                    df = filter_by_reliability(df, args.min_pri)
            # Re-validate embeddings
            embeddings_matrix, valid_df = validate_and_prepare_embeddings(df)
            if embeddings_matrix is None or len(embeddings_matrix) == 0:
                print("No valid embeddings found in entire dataset")
                return
        else:
            print("No valid embeddings found")
            return
    
    # Determine optimal clusters or use specified
    if args.clusters is None:
        # Simple optimal cluster detection (could be enhanced)
        n_samples = len(embeddings_matrix)
        max_k = min(15, n_samples // 50)  # At least 50 samples per cluster
        n_clusters = min(8, max(2, max_k))  # Default reasonable range
    else:
        n_clusters = args.clusters
    
    # Perform clustering
    kmeans_model, cluster_labels = perform_weighted_clustering(
        embeddings_matrix, valid_df, n_clusters, 
        use_pri_weights=args.pri_weighted and args.include_pri
    )
    
    # Add cluster labels to dataframe
    valid_df['cluster'] = cluster_labels
    
    # Analyze clusters
    cluster_analysis = analyze_clusters_with_pri(valid_df)
    
    # Create visualizations
    if not args.skip_viz:
        create_enhanced_visualizations(embeddings_matrix, valid_df, output_dir, args.gd)
    
    # Save results
    run_metadata = {
        'indicators_only': args.indicators_only,
        'min_pri_threshold': args.min_pri,
        'weighted_clustering': args.pri_weighted and args.include_pri
    }
    
    run_id, timestamp = save_enhanced_results(
        valid_df, cluster_analysis, output_dir, args.gd, run_metadata
    )
    
    print(f"\n=== Enhanced Analysis Complete ===")
    print(f"Found {n_clusters} clusters in {len(valid_df)} responses")
    print(f"Results saved with run ID: {run_id}")
    print(f"Output directory: {output_dir}")
    
    # Print summary statistics
    if args.include_pri and 'PRI_Scale_1_5' in valid_df.columns:
        print(f"\nPRI Summary:")
        pri_stats = valid_df['PRI_Scale_1_5'].describe()
        print(f"  Mean PRI: {pri_stats['mean']:.2f}")
        print(f"  Median PRI: {pri_stats['50%']:.2f}")
        print(f"  PRI Range: {pri_stats['min']:.2f} - {pri_stats['max']:.2f}")

if __name__ == "__main__":
    main() 