#!/usr/bin/env python3
"""
Comprehensive Persona Analysis - Combined GD1, GD2, GD3 Data
Using all features from persona development analysis: PRI score, cosine similarity, 
fear index, and thematic fears to generate comprehensive personas.

Output folder: analysis_output/combined/personas/
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_combined_data():
    """Load and combine GD1, GD2, GD3 data"""
    print("Loading combined data from GD1, GD2, GD3...")
    
    datasets = []
    for gd_num in [1, 2, 3]:
        try:
            # Load main data - adjust path for script running from tools/scripts
            df = pd.read_csv(f'../../Data/GD{gd_num}/GD{gd_num}_aggregate_standardized.csv')
            df['dataset'] = f'GD{gd_num}'
            datasets.append(df)
            print(f"✓ Loaded GD{gd_num}: {len(df)} participants")
        except Exception as e:
            print(f"✗ Error loading GD{gd_num}: {e}")
    
    if not datasets:
        raise ValueError("No datasets could be loaded")
    
    combined_df = pd.concat(datasets, ignore_index=True)
    print(f"✓ Combined dataset: {len(combined_df)} total participants")
    
    return combined_df

def calculate_thematic_fears(df):
    """Calculate thematic fear scores using keyword-based analysis"""
    print("Calculating thematic fear scores...")
    
    # Define fear themes with keywords
    fear_themes = {
        'economic_job_loss': [
            'job', 'unemployment', 'work', 'employment', 'career', 'income', 'salary',
            'economic', 'financial', 'money', 'economy', 'livelihood', 'profession'
        ],
        'surveillance_control': [
            'surveillance', 'privacy', 'control', 'monitoring', 'tracking', 'watch',
            'government', 'authoritarian', 'freedom', 'rights', 'liberty', 'democracy'
        ],
        'social_isolation': [
            'social', 'isolation', 'lonely', 'connection', 'relationship', 'community',
            'interaction', 'communication', 'human', 'society', 'people', 'together'
        ],
        'safety_security': [
            'safety', 'security', 'danger', 'risk', 'threat', 'harm', 'protect',
            'safe', 'secure', 'dangerous', 'risky', 'vulnerable', 'attack'
        ],
        'cultural_values': [
            'culture', 'tradition', 'values', 'identity', 'heritage', 'customs',
            'belief', 'religion', 'faith', 'community', 'ancestry', 'history'
        ],
        'technology_dependence': [
            'dependence', 'dependent', 'addiction', 'reliance', 'rely', 'control',
            'technology', 'digital', 'screen', 'device', 'internet', 'online'
        ]
    }
    
    # Find text columns
    text_columns = [col for col in df.columns if 'verbatim' in col.lower() or 'text' in col.lower() or col == 'Response']
    if not text_columns:
        text_columns = [col for col in df.columns if df[col].dtype == 'object' and col not in ['participant_id', 'dataset']]
    
    print(f"Using text columns: {text_columns[:5]}...")  # Show first 5 only
    
    # Calculate fear scores with improved weighting
    for theme, keywords in fear_themes.items():
        fear_scores = []
        
        for idx, row in df.iterrows():
            if idx % 10000 == 0:
                print(f"  Processing row {idx}/{len(df)} for {theme}")
            
            combined_text = ""
            for col in text_columns:
                if pd.notna(row[col]):
                    combined_text += str(row[col]).lower() + " "
            
            if combined_text.strip():
                # Use more sophisticated scoring with word boundaries
                words = combined_text.split()
                keyword_score = 0
                
                for keyword in keywords:
                    # Count exact word matches (more precise)
                    exact_matches = words.count(keyword)
                    # Count partial matches in words
                    partial_matches = sum(1 for word in words if keyword in word and len(word) > 3)
                    
                    # Weight exact matches more than partial matches
                    keyword_score += exact_matches * 2 + partial_matches * 0.5
                
                # Normalize by text length but add a minimum threshold
                text_length = len(words)
                if text_length > 0:
                    fear_score = keyword_score / max(text_length, 10)  # Prevent over-normalization
                    # Add a small boost for very short texts with keywords
                    if text_length < 10 and keyword_score > 0:
                        fear_score *= 1.5
                else:
                    fear_score = 0
            else:
                fear_score = 0
            
            fear_scores.append(fear_score)
        
        # Apply log transformation to reduce skewness
        fear_scores_array = np.array(fear_scores)
        # Add small constant before log to handle zeros
        log_scores = np.log1p(fear_scores_array * 100) / 10  # Scale for better range
        
        df[f'{theme}_fear'] = log_scores
        print(f"✓ Calculated {theme}_fear: mean={np.mean(log_scores):.4f}, std={np.std(log_scores):.4f}")
    
    return df

def calculate_fear_index(df):
    """Calculate overall fear index"""
    print("Calculating fear index...")
    
    # Use only the thematic fears we just calculated
    thematic_fear_cols = [col for col in df.columns if col.endswith('_fear')]
    
    if thematic_fear_cols:
        # Only use numeric columns
        numeric_fear_cols = []
        for col in thematic_fear_cols:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                numeric_fear_cols.append(col)
        
        if numeric_fear_cols:
            df['fear_index'] = df[numeric_fear_cols].mean(axis=1, skipna=True)
        else:
            df['fear_index'] = 0.5  # Neutral value
    else:
        df['fear_index'] = 0.5  # Neutral value
    
    print(f"✓ Fear index calculated: mean={df['fear_index'].mean():.4f}")
    return df

def calculate_cosine_similarity(df):
    """Calculate cosine similarity proxy"""
    print("Calculating cosine similarity proxy...")
    
    # Use response consistency as proxy for cosine similarity
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    response_cols = [col for col in numeric_cols if any(term in col.lower() for term in 
                    ['response', 'answer', 'rating', 'scale', 'score'])]
    
    if response_cols:
        # Calculate response variance as inverse of consistency
        response_variance = df[response_cols].var(axis=1, skipna=True)
        # Convert to similarity (lower variance = higher similarity)
        df['cosine_similarity'] = 1 / (1 + response_variance)
    else:
        df['cosine_similarity'] = 0.7  # Default moderate similarity
    
    df['cosine_similarity'] = df['cosine_similarity'].fillna(0.7)
    print(f"✓ Cosine similarity calculated: mean={df['cosine_similarity'].mean():.4f}")
    return df

def calculate_response_length_norm(df):
    """Calculate normalized response length"""
    print("Calculating response length norm...")
    
    # Find text columns
    text_columns = [col for col in df.columns if 'verbatim' in col.lower() or 'text' in col.lower() or col == 'Response']
    if not text_columns:
        text_columns = [col for col in df.columns if df[col].dtype == 'object' and col not in ['participant_id', 'dataset']]
    
    response_lengths = []
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"  Processing row {idx}/{len(df)} for response length")
            
        total_length = 0
        for col in text_columns:
            if pd.notna(row[col]):
                total_length += len(str(row[col]))
        response_lengths.append(total_length)
    
    # Normalize to 0-1 scale
    if response_lengths:
        max_length = max(response_lengths)
        if max_length > 0:
            df['response_length_norm'] = [length / max_length for length in response_lengths]
        else:
            df['response_length_norm'] = 0.5
    else:
        df['response_length_norm'] = 0.5
    
    print(f"✓ Response length norm calculated: mean={df['response_length_norm'].mean():.4f}")
    return df

def ensure_pri_score(df):
    """Ensure PRI score exists and normalize it properly"""
    if 'PRI_Score' not in df.columns:
        print("PRI_Score not found, creating improved proxy...")
        # Create better PRI proxy using response variance across key columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 5:
            # Use coefficient of variation as reliability measure
            # Lower CV = higher reliability
            response_mean = df[numeric_cols].mean(axis=1, skipna=True)
            response_std = df[numeric_cols].std(axis=1, skipna=True)
            cv = response_std / (response_mean + 0.001)  # Add small constant to avoid division by zero
            # Convert to reliability score (inverse of CV, normalized to 0-1)
            df['PRI_Score'] = 1 / (1 + cv)
        else:
            df['PRI_Score'] = 0.7  # Default moderate reliability
    
    # Normalize PRI_Score to 0-1 range
    min_pri = df['PRI_Score'].min()
    max_pri = df['PRI_Score'].max()
    if max_pri > min_pri:
        df['PRI_Score'] = (df['PRI_Score'] - min_pri) / (max_pri - min_pri)
    else:
        df['PRI_Score'] = 0.7
    
    df['PRI_Score'] = df['PRI_Score'].fillna(0.7)
    print(f"✓ PRI Score ensured and normalized: mean={df['PRI_Score'].mean():.4f}")
    return df

def perform_clustering(df, n_clusters=5):
    """Perform clustering analysis with comprehensive features"""
    print(f"Performing clustering analysis with {n_clusters} clusters...")
    
    # Define the 10 features to use
    feature_cols = [
        'fear_index',
        'PRI_Score', 
        'cosine_similarity',
        'economic_job_loss_fear',
        'surveillance_control_fear',
        'social_isolation_fear', 
        'safety_security_fear',
        'cultural_values_fear',
        'technology_dependence_fear',
        'response_length_norm'
    ]
    
    # Ensure all features exist
    for col in feature_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found, setting to default")
            df[col] = 0.5
    
    # Prepare clustering data
    clustering_data = df[feature_cols].copy()
    clustering_data = clustering_data.fillna(clustering_data.mean())
    
    print(f"Clustering features shape: {clustering_data.shape}")
    print("Feature statistics:")
    for col in feature_cols:
        print(f"  {col}: mean={clustering_data[col].mean():.4f}, std={clustering_data[col].std():.4f}")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(clustering_data)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(features_scaled, clusters)
    
    # Add results to dataframe
    df['persona_cluster'] = clusters
    df['cluster_confidence'] = np.max(kmeans.transform(features_scaled), axis=1)
    
    print(f"✓ Clustering completed:")
    print(f"  Silhouette Score: {silhouette_avg:.4f}")
    print(f"  Cluster distribution: {pd.Series(clusters).value_counts().sort_index().to_dict()}")
    
    return df, feature_cols, scaler, kmeans, silhouette_avg

def generate_personas(df, feature_cols):
    """Generate persona profiles"""
    print("Generating persona profiles...")
    
    personas = {}
    
    for cluster_id in sorted(df['persona_cluster'].unique()):
        cluster_data = df[df['persona_cluster'] == cluster_id]
        cluster_size = len(cluster_data)
        cluster_percentage = (cluster_size / len(df)) * 100
        
        # Calculate feature means for this cluster
        feature_means = {}
        for col in feature_cols:
            feature_means[col] = cluster_data[col].mean()
        
        # Generate persona name based on dominant characteristics
        persona_name = generate_persona_name(feature_means, cluster_id)
        
        # Create persona profile
        persona = {
            'name': persona_name,
            'cluster_id': cluster_id,
            'size': cluster_size,
            'percentage': cluster_percentage,
            'features': feature_means,
            'description': generate_persona_description(feature_means),
            'dataset_distribution': cluster_data['dataset'].value_counts().to_dict()
        }
        
        personas[cluster_id] = persona
        print(f"✓ Generated persona {cluster_id}: {persona_name} ({cluster_percentage:.1f}%)")
    
    return personas

def generate_persona_name(feature_means, cluster_id):
    """Generate persona name based on dominant features"""
    # Define name patterns based on feature combinations
    high_fear = feature_means['fear_index'] > 0.6
    high_pri = feature_means['PRI_Score'] > 0.7
    high_similarity = feature_means['cosine_similarity'] > 0.7
    high_response_length = feature_means['response_length_norm'] > 0.6
    
    # Check dominant thematic fears
    thematic_fears = {
        'economic_job_loss_fear': 'Economic',
        'surveillance_control_fear': 'Privacy',
        'social_isolation_fear': 'Social',
        'safety_security_fear': 'Security',
        'cultural_values_fear': 'Cultural',
        'technology_dependence_fear': 'Dependency'
    }
    
    dominant_fear = max(thematic_fears.keys(), key=lambda x: feature_means[x])
    fear_label = thematic_fears[dominant_fear]
    
    # Generate names based on patterns
    if high_fear and high_pri:
        return f"The Thoughtful {fear_label} Worrier"
    elif high_fear and not high_pri:
        return f"The Anxious {fear_label} Reactor"
    elif not high_fear and high_pri:
        return f"The Measured {fear_label} Analyst"
    elif high_similarity and high_response_length:
        return f"The Engaged {fear_label} Communicator"
    elif high_similarity:
        return f"The Consistent {fear_label} Responder"
    else:
        return f"The Balanced {fear_label} Participant"

def generate_persona_description(feature_means):
    """Generate persona description based on features"""
    description_parts = []
    
    # Fear level
    fear_level = feature_means['fear_index']
    if fear_level > 0.7:
        description_parts.append("highly concerned about AI implications")
    elif fear_level > 0.5:
        description_parts.append("moderately cautious about AI")
    else:
        description_parts.append("relatively optimistic about AI")
    
    # Reliability
    pri_score = feature_means['PRI_Score']
    if pri_score > 0.7:
        description_parts.append("provides consistent and reliable responses")
    elif pri_score > 0.5:
        description_parts.append("shows moderate response consistency")
    else:
        description_parts.append("displays variable response patterns")
    
    # Engagement
    response_length = feature_means['response_length_norm']
    if response_length > 0.6:
        description_parts.append("highly engaged with detailed responses")
    elif response_length > 0.4:
        description_parts.append("moderately engaged in discussions")
    else:
        description_parts.append("provides brief, focused responses")
    
    # Dominant fear
    thematic_fears = {
        'economic_job_loss_fear': 'economic and employment impacts',
        'surveillance_control_fear': 'privacy and surveillance concerns',
        'social_isolation_fear': 'social connection and isolation',
        'safety_security_fear': 'safety and security risks',
        'cultural_values_fear': 'cultural and value preservation',
        'technology_dependence_fear': 'technology dependence issues'
    }
    
    dominant_fear = max(thematic_fears.keys(), key=lambda x: feature_means[x])
    description_parts.append(f"particularly focused on {thematic_fears[dominant_fear]}")
    
    return "; ".join(description_parts)

def create_visualizations(df, personas, feature_cols, output_dir):
    """Create comprehensive visualizations"""
    print("Creating visualizations...")
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Persona size distribution
    create_persona_distribution_viz(personas, viz_dir)
    
    # 2. Feature comparison radar chart
    create_feature_radar_viz(personas, feature_cols, viz_dir)
    
    # 3. Feature correlation heatmap
    create_correlation_heatmap(df, feature_cols, viz_dir)
    
    # 4. PCA visualization
    create_pca_visualization(df, feature_cols, viz_dir)
    
    # 5. Dataset distribution by persona
    create_dataset_distribution_viz(df, viz_dir)
    
    # 6. Interactive choropleth map of features across regions
    create_regional_choropleth_map(df, feature_cols, viz_dir)

def create_persona_distribution_viz(personas, viz_dir):
    """Create persona size distribution visualization"""
    names = [p['name'] for p in personas.values()]
    sizes = [p['size'] for p in personas.values()]
    percentages = [p['percentage'] for p in personas.values()]
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=percentages,
            text=[f"{size}<br>({pct:.1f}%)" for size, pct in zip(sizes, percentages)],
            textposition='auto',
            marker_color=px.colors.qualitative.Set3[:len(names)]
        )
    ])
    
    fig.update_layout(
        title="Persona Distribution - Comprehensive Analysis",
        xaxis_title="Persona",
        yaxis_title="Percentage of Participants",
        height=500
    )
    
    fig.write_json(viz_dir / "persona_distribution.json")
    print("✓ Created persona distribution visualization")

def create_feature_radar_viz(personas, feature_cols, viz_dir):
    """Create radar chart comparing personas across features with normalized values"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    # Get all feature values to normalize them globally
    all_values = {}
    for col in feature_cols:
        all_values[col] = [persona['features'][col] for persona in personas.values()]
    
    # Normalize each feature to 0-1 scale
    normalized_personas = {}
    for cluster_id, persona in personas.items():
        normalized_features = {}
        for col in feature_cols:
            col_values = all_values[col]
            min_val, max_val = min(col_values), max(col_values)
            if max_val > min_val:
                normalized_val = (persona['features'][col] - min_val) / (max_val - min_val)
            else:
                normalized_val = 0.5
            normalized_features[col] = normalized_val
        normalized_personas[cluster_id] = normalized_features
    
    # Create radar chart with normalized values
    for i, (cluster_id, persona) in enumerate(personas.items()):
        values = [normalized_personas[cluster_id][col] for col in feature_cols]
        
        # Create better labels
        clean_labels = [col.replace('_', ' ').replace('fear', '').title() for col in feature_cols]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=clean_labels,
            fill='toself',
            name=persona['name'],
            line_color=colors[i % len(colors)],
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='linear',
                tick0=0,
                dtick=0.2
            )),
        showlegend=True,
        title="Persona Feature Profiles - Normalized Radar Chart",
        height=700,
        font=dict(size=12)
    )
    
    fig.write_json(viz_dir / "persona_feature_radar.json")
    print("✓ Created normalized persona feature radar chart")

def create_correlation_heatmap(df, feature_cols, viz_dir):
    """Create feature correlation heatmap"""
    corr_matrix = df[feature_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(x="Features", y="Features", color="Correlation"),
        x=feature_cols,
        y=feature_cols,
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    
    fig.update_layout(height=600)
    fig.write_json(viz_dir / "feature_correlation_heatmap.json")
    print("✓ Created correlation heatmap")

def create_pca_visualization(df, feature_cols, viz_dir):
    """Create PCA visualization"""
    # Prepare data
    data = df[feature_cols].fillna(df[feature_cols].mean())
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)
    
    # Create visualization
    fig = px.scatter(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        color=df['persona_cluster'].astype(str),
        title=f"PCA Visualization of Personas<br>Explained Variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}",
        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.3f})', 
                'y': f'PC2 ({pca.explained_variance_ratio_[1]:.3f})',
                'color': 'Persona'}
    )
    
    fig.update_layout(height=500)
    fig.write_json(viz_dir / "pca_visualization.json")
    print("✓ Created PCA visualization")

def create_dataset_distribution_viz(df, viz_dir):
    """Create dataset distribution by persona visualization"""
    # Calculate distribution
    dist_data = []
    for persona_id in sorted(df['persona_cluster'].unique()):
        persona_data = df[df['persona_cluster'] == persona_id]
        for dataset in ['GD1', 'GD2', 'GD3']:
            count = len(persona_data[persona_data['dataset'] == dataset])
            percentage = (count / len(persona_data)) * 100 if len(persona_data) > 0 else 0
            dist_data.append({
                'Persona': f'Persona {persona_id}',
                'Dataset': dataset,
                'Count': count,
                'Percentage': percentage
            })
    
    dist_df = pd.DataFrame(dist_data)
    
    fig = px.bar(
        dist_df,
        x='Persona',
        y='Percentage',
        color='Dataset',
        title="Dataset Distribution Across Personas",
        labels={'Percentage': 'Percentage within Persona'}
    )
    
    fig.update_layout(height=500)
    fig.write_json(viz_dir / "dataset_distribution.json")
    print("✓ Created dataset distribution visualization")

def create_regional_choropleth_map(df, feature_cols, viz_dir):
    """Create interactive choropleth map showing feature distribution across regions (GD1, GD2, GD3)"""
    print("Creating regional choropleth map...")
    
    # Calculate regional averages for each feature
    regional_data = []
    
    # Map datasets to mock geographic regions for visualization
    region_mapping = {
        'GD1': {'name': 'North America', 'iso': 'US', 'lat': 39.8283, 'lon': -98.5795},
        'GD2': {'name': 'Europe', 'iso': 'DE', 'lat': 51.1657, 'lon': 10.4515}, 
        'GD3': {'name': 'Asia-Pacific', 'iso': 'JP', 'lat': 36.2048, 'lon': 138.2529}
    }
    
    for dataset in ['GD1', 'GD2', 'GD3']:
        dataset_data = df[df['dataset'] == dataset]
        region_info = region_mapping[dataset]
        
        for feature in feature_cols:
            avg_value = dataset_data[feature].mean()
            regional_data.append({
                'Dataset': dataset,
                'Region': region_info['name'],
                'ISO': region_info['iso'],
                'Feature': feature.replace('_', ' ').title(),
                'Value': avg_value,
                'Participants': len(dataset_data),
                'Lat': region_info['lat'],
                'Lon': region_info['lon']
            })
    
    regional_df = pd.DataFrame(regional_data)
    
    # Create subplots for each feature
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Get unique features
    unique_features = regional_df['Feature'].unique()
    n_features = len(unique_features)
    
    # Create a grid layout
    cols = 3
    rows = (n_features + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=unique_features,
        specs=[[{"type": "scattergeo"}] * cols for _ in range(rows)],
        vertical_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set3
    
    for i, feature in enumerate(unique_features):
        row = i // cols + 1
        col = i % cols + 1
        
        feature_data = regional_df[regional_df['Feature'] == feature]
        
        # Normalize values for better color mapping
        max_val = feature_data['Value'].max()
        min_val = feature_data['Value'].min()
        if max_val > min_val:
            normalized_values = (feature_data['Value'] - min_val) / (max_val - min_val)
        else:
            normalized_values = [0.5] * len(feature_data)
        
        fig.add_trace(
            go.Scattergeo(
                lon=feature_data['Lon'],
                lat=feature_data['Lat'],
                text=feature_data['Region'],
                mode='markers',
                marker=dict(
                    size=normalized_values * 40 + 20,  # Scale marker size
                    color=normalized_values,
                    colorscale='Viridis',
                    showscale=(i == 0),  # Only show colorbar for first subplot
                    sizemode='diameter',
                    opacity=0.8,
                    line=dict(width=2, color='white')
                ),
                customdata=feature_data[['Dataset', 'Value', 'Participants']],
                hovertemplate=
                    "<b>%{text}</b><br>" +
                    "Dataset: %{customdata[0]}<br>" +
                    f"{feature}: %{{customdata[1]:.4f}}<br>" +
                    "Participants: %{customdata[2]}<br>" +
                    "<extra></extra>",
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Update geo for this subplot
        fig.update_geos(
            projection_type="orthographic",
            showland=True,
            landcolor="lightgray",
            showocean=True,
            oceancolor="lightblue",
            row=row, col=col
        )
    
    fig.update_layout(
        title="Feature Distribution Across Regions (Datasets as Geographic Proxies)",
        height=300 * rows,
        showlegend=False
    )
    
    fig.write_json(viz_dir / "regional_feature_choropleth.json")
    print("✓ Created regional choropleth map")
    
    # Also create a simple bar chart comparison
    fig2 = px.bar(
        regional_df, 
        x='Region', 
        y='Value', 
        color='Feature',
        facet_col='Feature',
        facet_col_wrap=3,
        title="Feature Values by Region (Comparative View)",
        labels={'Value': 'Feature Value', 'Region': 'Dataset Region'}
    )
    
    fig2.update_layout(height=600)
    fig2.write_json(viz_dir / "regional_feature_comparison.json")
    print("✓ Created regional feature comparison chart")

def save_results(df, personas, feature_cols, silhouette_score, output_dir):
    """Save analysis results"""
    print("Saving results...")
    
    # Save personas as JSON
    personas_output = {}
    for cluster_id, persona in personas.items():
        personas_output[f"persona_{cluster_id}"] = {
            'name': persona['name'],
            'description': persona['description'],
            'size': int(persona['size']),
            'percentage': float(persona['percentage']),
            'features': {k: float(v) for k, v in persona['features'].items()},
            'dataset_distribution': persona['dataset_distribution']
        }
    
    with open(output_dir / "persona_profiles.json", 'w') as f:
        json.dump(personas_output, f, indent=2)
    
    # Save detailed participant data
    participant_columns = ['Participant ID', 'dataset', 'persona_cluster'] + feature_cols
    available_columns = [col for col in participant_columns if col in df.columns]
    df[available_columns].to_csv(output_dir / "participant_persona_assignments.csv", index=False)
    
    # Save summary report
    report = {
        'analysis_type': 'Comprehensive Persona Analysis',
        'timestamp': datetime.now().isoformat(),
        'total_participants': len(df),
        'num_personas': len(personas),
        'features_used': feature_cols,
        'silhouette_score': float(silhouette_score),
        'dataset_distribution': df['dataset'].value_counts().to_dict(),
        'persona_summary': {
            f"persona_{k}": {
                'name': v['name'],
                'size': v['size'],
                'percentage': round(v['percentage'], 1)
            } for k, v in personas.items()
        }
    }
    
    with open(output_dir / "analysis_summary.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save methodology report
    methodology = f"""# Comprehensive Persona Analysis - Methodology Report

## Overview
This analysis uses 10 comprehensive features to generate personas from combined GD1, GD2, GD3 data.

## Features Used
{chr(10).join([f"- {col}" for col in feature_cols])}

## Methodology
1. **Data Loading**: Combined GD1, GD2, GD3 datasets ({len(df)} total participants)
2. **Feature Engineering**: 
   - Thematic fear calculation using keyword-based analysis
   - PRI score calculation/proxy generation
   - Cosine similarity proxy from response consistency
   - Response length normalization
3. **Clustering**: K-means with {len(personas)} clusters
4. **Validation**: Silhouette score = {silhouette_score:.4f}

## Results Summary
- **Total Participants**: {len(df)}
- **Number of Personas**: {len(personas)}
- **Clustering Quality**: {silhouette_score:.4f} (silhouette score)

## Persona Profiles
{chr(10).join([f"- **{p['name']}**: {p['size']} participants ({p['percentage']:.1f}%)" for p in personas.values()])}

## Dataset Distribution
{chr(10).join([f"- {k}: {v} participants" for k, v in df['dataset'].value_counts().items()])}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(output_dir / "methodology_report.md", 'w') as f:
        f.write(methodology)
    
    print(f"✓ Results saved to {output_dir}")
    print(f"  - Persona profiles: persona_profiles.json")
    print(f"  - Participant data: participant_persona_assignments.csv")
    print(f"  - Analysis summary: analysis_summary.json")
    print(f"  - Methodology report: methodology_report.md")
    print(f"  - Visualizations: visualizations/ folder (JSON format)")

def main():
    """Main analysis function"""
    print("=== Comprehensive Persona Analysis ===")
    print("Using all persona development features: PRI score, cosine similarity, fear index, thematic fears")
    
    # Create output directory - adjust path for script running from tools/scripts
    output_dir = Path("../../analysis_output/combined/personas")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load and prepare data
        df = load_combined_data()
        
        # Calculate all required features
        df = calculate_thematic_fears(df)
        df = calculate_fear_index(df)
        df = calculate_cosine_similarity(df)
        df = calculate_response_length_norm(df)
        df = ensure_pri_score(df)
        
        # Perform clustering (minimum 5 personas as requested)
        df, feature_cols, scaler, kmeans, silhouette_score = perform_clustering(df, n_clusters=5)
        
        # Generate personas
        personas = generate_personas(df, feature_cols)
        
        # Create visualizations
        create_visualizations(df, personas, feature_cols, output_dir)
        
        # Save results
        save_results(df, personas, feature_cols, silhouette_score, output_dir)
        
        print("\n=== Analysis Complete ===")
        print(f"Generated {len(personas)} personas with silhouette score: {silhouette_score:.4f}")
        print(f"Results saved to: {output_dir}")
        
        # Print persona summary
        print("\n=== Persona Summary ===")
        for cluster_id, persona in personas.items():
            print(f"{persona['name']}: {persona['size']} participants ({persona['percentage']:.1f}%)")
            print(f"  Description: {persona['description']}")
            print()
    
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 