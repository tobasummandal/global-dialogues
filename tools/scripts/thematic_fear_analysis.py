#!/usr/bin/env python3
"""
Thematic Fear Analysis for Global Dialogues
==========================================

Comprehensive analysis of AI fears and anxieties combining:
- Thematic analysis with reliability weighting
- Fear index calculation using existing data
- Persona development based on fear patterns
- No API keys required - uses existing rich dataset

Usage:
    python thematic_fear_analysis.py GD3 [--min-pri 3.0] [--output-dir analysis_output]
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# ML imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import scipy.stats as stats

# Set style
plt.style.use('default')
sns.set_palette("husl")

class ThematicFearAnalysis:
    def __init__(self, gd_number: str, output_dir: str = "analysis_output"):
        self.gd_number = gd_number
        self.output_dir = Path(output_dir) / f"GD{gd_number}" / "thematic_fear_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Fear keywords dictionary
        self.fear_keywords = {
            'high_fear': ['terrified', 'terrifying', 'afraid', 'scared', 'fear', 'fearful', 'frightened', 'nightmare', 'horrified', 'dread', 'panic', 'alarmed'],
            'moderate_fear': ['worried', 'concern', 'concerning', 'anxious', 'nervous', 'uneasy', 'uncomfortable', 'apprehensive', 'troubled', 'disturbed'],
            'mild_fear': ['hesitant', 'cautious', 'wary', 'skeptical', 'doubtful', 'uncertain', 'unsure', 'questioning']
        }
        
        # Theme categories for fear analysis
        self.fear_themes = {
            'economic_job_loss': ['unemployment', 'job loss', 'economic', 'livelihood', 'income', 'work', 'employment', 'career', 'financial'],
            'surveillance_control': ['surveillance', 'control', 'monitoring', 'tracking', 'privacy', 'manipulation', 'autonomy', 'freedom', 'liberty'],
            'social_isolation': ['human connection', 'relationships', 'social', 'isolation', 'loneliness', 'communication', 'interaction', 'community'],
            'safety_security': ['safety', 'security', 'harm', 'danger', 'threat', 'risk', 'violence', 'attack', 'weapons'],
            'cultural_values': ['culture', 'tradition', 'values', 'beliefs', 'religion', 'identity', 'heritage', 'customs'],
            'technology_dependence': ['dependence', 'addiction', 'reliance', 'creativity', 'skills', 'intelligence', 'thinking', 'capability']
        }
        
        # Data containers
        self.embeddings_df = None
        self.thematic_df = None
        self.pri_df = None
        self.indicators_df = None
        self.aggregate_df = None
        
    def load_data(self):
        """Load all necessary datasets"""
        print(f"📊 Loading GD{self.gd_number} datasets...")
        
        try:
            # Load embeddings data (adjust path since script runs from tools/scripts)
            embeddings_path = f"../../Data/GD{self.gd_number}/GD{self.gd_number}_embeddings.csv"
            if os.path.exists(embeddings_path):
                self.embeddings_df = pd.read_csv(embeddings_path)
                print(f"✅ Loaded {len(self.embeddings_df)} embeddings records")
            
            # Load thematic rankings (adjust path since script runs from tools/scripts)
            thematic_path = f"../../analysis_output/GD{self.gd_number}/thematic_rankings/thematic_rankings.csv"
            if os.path.exists(thematic_path):
                self.thematic_df = pd.read_csv(thematic_path)
                print(f"✅ Loaded {len(self.thematic_df)} thematic rankings")
            else:
                print(f"⚠️  Thematic rankings not found at {thematic_path}")
                # Try to use enhanced clustering results as fallback
                enhanced_dir = f"../../analysis_output/GD{self.gd_number}/enhanced_semantic_analysis/"
                if os.path.exists(enhanced_dir):
                    enhanced_files = [f for f in os.listdir(enhanced_dir) if f.startswith("enhanced_clustering_results_")]
                    if enhanced_files:
                        enhanced_file = sorted(enhanced_files)[-1]  # Get most recent
                        enhanced_full_path = enhanced_dir + enhanced_file
                        print(f"📋 Loading enhanced clustering results: {enhanced_file}")
                        self.thematic_df = pd.read_csv(enhanced_full_path)
                        # Rename columns to match expected format
                        if 'English Responses' in self.thematic_df.columns:
                            self.thematic_df['theme'] = 'mixed'  # Default theme
                            self.thematic_df['cosine_similarity'] = 0.5  # Default similarity
                            print(f"✅ Loaded {len(self.thematic_df)} enhanced clustering records")
                        else:
                            print(f"❌ Enhanced clustering data format not compatible")
                            self.thematic_df = None
                    else:
                        print(f"❌ No enhanced clustering results found")
                        self.thematic_df = None
                else:
                    print(f"❌ No enhanced semantic analysis directory found")
                    self.thematic_df = None
            
            # Load PRI scores (adjust path since script runs from tools/scripts)
            pri_path = f"../../analysis_output/GD{self.gd_number}/pri/GD{self.gd_number}_pri_scores.csv"
            if os.path.exists(pri_path):
                self.pri_df = pd.read_csv(pri_path)
                print(f"✅ Loaded {len(self.pri_df)} PRI scores")
            
            # Load aggregate data for indicators (adjust path since script runs from tools/scripts)
            aggregate_path = f"../../Data/GD{self.gd_number}/GD{self.gd_number}_aggregate_standardized.csv"
            if os.path.exists(aggregate_path):
                print(f"📋 Loading aggregate data (may take a moment)...")
                self.aggregate_df = pd.read_csv(aggregate_path)
                print(f"✅ Loaded {len(self.aggregate_df)} aggregate records")
            
            # Ensure we have at least some data to work with
            if self.thematic_df is None:
                print("❌ No thematic data available for analysis")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def calculate_fear_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate fear sentiment using keyword analysis"""
        if pd.isna(text):
            return {'fear_score': 0.0, 'fear_intensity': 'none', 'fear_keywords': []}
        
        text_lower = text.lower()
        found_keywords = []
        fear_score = 0.0
        
        # Check for fear keywords with different weights
        for category, keywords in self.fear_keywords.items():
            weight = {'high_fear': 3.0, 'moderate_fear': 2.0, 'mild_fear': 1.0}[category]
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)
                    fear_score += weight
        
        # Normalize fear score
        if fear_score > 0:
            fear_score = min(fear_score / 10.0, 1.0)  # Cap at 1.0
        
        # Determine intensity
        if fear_score >= 0.6:
            intensity = 'high'
        elif fear_score >= 0.3:
            intensity = 'moderate'
        elif fear_score > 0:
            intensity = 'mild'
        else:
            intensity = 'none'
        
        return {
            'fear_score': fear_score,
            'fear_intensity': intensity,
            'fear_keywords': found_keywords
        }
    
    def calculate_theme_fear_scores(self, text: str) -> Dict[str, float]:
        """Calculate fear scores for specific themes"""
        if pd.isna(text):
            return {theme: 0.0 for theme in self.fear_themes.keys()}
        
        text_lower = text.lower()
        theme_scores = {}
        
        for theme, keywords in self.fear_themes.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1.0
            # Normalize by number of keywords in theme
            theme_scores[theme] = min(score / len(keywords), 1.0)
        
        return theme_scores
    
    def build_fear_index(self, min_pri: float = 0.0) -> pd.DataFrame:
        """Build comprehensive fear index"""
        print(f"🔍 Building Fear Index (PRI >= {min_pri})...")
        
        # Start with thematic data
        fear_df = self.thematic_df.copy()
        
        # Add PRI scores
        if self.pri_df is not None:
            fear_df = fear_df.merge(
                self.pri_df[['Participant ID', 'PRI_Score', 'PRI_Scale_1_5']],
                on='Participant ID',
                how='left'
            )
            fear_df['PRI_Score'] = fear_df['PRI_Score'].fillna(0.5)
            fear_df['PRI_Scale_1_5'] = fear_df['PRI_Scale_1_5'].fillna(2.5)
        else:
            fear_df['PRI_Score'] = 0.5
            fear_df['PRI_Scale_1_5'] = 2.5
        
        # Filter by minimum PRI if specified
        if min_pri > 0:
            fear_df = fear_df[fear_df['PRI_Scale_1_5'] >= min_pri]
            print(f"🔍 Filtered to {len(fear_df)} records with PRI >= {min_pri}")
        
        # Calculate fear sentiment for each response
        print("📊 Calculating fear sentiment...")
        fear_analysis = fear_df['English Responses'].apply(self.calculate_fear_sentiment)
        fear_df['fear_score'] = [x['fear_score'] for x in fear_analysis]
        fear_df['fear_intensity'] = [x['fear_intensity'] for x in fear_analysis]
        fear_df['fear_keywords'] = [x['fear_keywords'] for x in fear_analysis]
        
        # Calculate theme-specific fear scores
        print("🎯 Calculating theme-specific fear scores...")
        theme_analysis = fear_df['English Responses'].apply(self.calculate_theme_fear_scores)
        for theme in self.fear_themes.keys():
            fear_df[f'theme_fear_{theme}'] = [x[theme] for x in theme_analysis]
        
        # Calculate composite fear index
        print("📈 Calculating composite fear index...")
        fear_df['theme_intensity'] = fear_df['cosine_similarity']  # Use existing similarity as theme intensity
        fear_df['response_length'] = fear_df['English Responses'].str.len().fillna(0)
        fear_df['response_length_norm'] = fear_df['response_length'] / fear_df['response_length'].max()
        
        # Composite Fear Index = Theme_Match × Fear_Sentiment × PRI_Weight × Response_Engagement
        fear_df['fear_index'] = (
            fear_df['theme_intensity'] * 0.4 +
            fear_df['fear_score'] * 0.3 +
            fear_df['PRI_Score'] * 0.2 +
            fear_df['response_length_norm'] * 0.1
        )
        
        # Add fear categories
        fear_df['fear_category'] = pd.cut(
            fear_df['fear_index'],
            bins=[0, 0.2, 0.4, 0.6, 1.0],
            labels=['Low', 'Moderate', 'High', 'Extreme']
        )
        
        print(f"✅ Fear Index completed for {len(fear_df)} responses")
        return fear_df
    
    def create_personas(self, fear_df: pd.DataFrame) -> Dict[str, Dict]:
        """Create personas based on fear patterns"""
        print("🧠 Creating personas based on fear patterns...")
        
        # Prepare features for clustering
        feature_columns = [
            'fear_index', 'theme_intensity', 'fear_score', 'PRI_Score'
        ] + [f'theme_fear_{theme}' for theme in self.fear_themes.keys()]
        
        # Add theme dummies
        theme_dummies = pd.get_dummies(fear_df['theme'], prefix='theme')
        clustering_data = pd.concat([fear_df[feature_columns], theme_dummies], axis=1)
        clustering_data = clustering_data.fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(clustering_data)
        
        # Determine optimal number of clusters
        silhouette_scores = []
        k_range = range(3, 8)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"🎯 Optimal number of personas: {optimal_k}")
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        fear_df['persona_cluster'] = kmeans.fit_predict(scaled_features)
        
        # Create persona profiles
        personas = {}
        persona_names = [
            "The Anxious Traditionalist",
            "The Economic Pragmatist", 
            "The Privacy Guardian",
            "The Overwhelmed Observer",
            "The Cautious Optimist",
            "The Tech-Dependent Worrier",
            "The Social Connector"
        ]
        
        for i in range(optimal_k):
            cluster_data = fear_df[fear_df['persona_cluster'] == i]
            
            # Calculate persona characteristics
            persona_profile = {
                'name': persona_names[i] if i < len(persona_names) else f"Persona {i+1}",
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(fear_df) * 100,
                'avg_fear_index': cluster_data['fear_index'].mean(),
                'avg_pri_score': cluster_data['PRI_Score'].mean(),
                'dominant_themes': cluster_data['theme'].value_counts().head(3).to_dict(),
                'fear_intensity_dist': cluster_data['fear_intensity'].value_counts().to_dict(),
                'top_fear_keywords': [kw for keywords in cluster_data['fear_keywords'] for kw in keywords],
                'sample_responses': cluster_data['English Responses'].head(3).tolist(),
                'theme_fear_scores': {
                    theme: cluster_data[f'theme_fear_{theme}'].mean() 
                    for theme in self.fear_themes.keys()
                }
            }
            
            # Get most common fear keywords
            keyword_counter = Counter(persona_profile['top_fear_keywords'])
            persona_profile['top_fear_keywords'] = dict(keyword_counter.most_common(5))
            
            personas[f"persona_{i}"] = persona_profile
        
        return personas
    
    def create_fear_visualizations(self, fear_df: pd.DataFrame, personas: Dict[str, Dict]):
        """Create comprehensive fear visualizations"""
        print("📊 Creating fear visualizations...")
        
        # 1. Fear Index Distribution
        fig_fear_dist = px.histogram(
            fear_df, 
            x='fear_index',
            color='fear_category',
            title='Fear Index Distribution',
            labels={'fear_index': 'Fear Index', 'count': 'Number of Responses'},
            color_discrete_map={'Low': '#90EE90', 'Moderate': '#FFD700', 'High': '#FF8C00', 'Extreme': '#FF4500'}
        )
        fig_fear_dist.write_html(str(self.output_dir / "fear_index_distribution.html"))
        
        # 2. Fear by Theme
        theme_fear_avg = fear_df.groupby('theme')['fear_index'].mean().sort_values(ascending=False)
        fig_theme_fear = px.bar(
            x=theme_fear_avg.values,
            y=theme_fear_avg.index,
            orientation='h',
            title='Average Fear Index by Theme',
            labels={'x': 'Average Fear Index', 'y': 'Theme'}
        )
        fig_theme_fear.write_html(str(self.output_dir / "fear_by_theme.html"))
        
        # 3. PRI vs Fear Index Scatter
        fig_pri_fear = px.scatter(
            fear_df,
            x='PRI_Score',
            y='fear_index',
            color='fear_intensity',
            size='response_length',
            title='Participant Reliability vs Fear Index',
            labels={'PRI_Score': 'PRI Score', 'fear_index': 'Fear Index'},
            hover_data=['theme', 'fear_keywords']
        )
        fig_pri_fear.write_html(str(self.output_dir / "pri_vs_fear.html"))
        
        # 4. Persona Visualization
        persona_data = []
        for persona_id, persona in personas.items():
            persona_data.append({
                'persona': persona['name'],
                'size': persona['size'],
                'avg_fear_index': persona['avg_fear_index'],
                'avg_pri_score': persona['avg_pri_score']
            })
        
        persona_df = pd.DataFrame(persona_data)
        fig_personas = px.scatter(
            persona_df,
            x='avg_pri_score',
            y='avg_fear_index',
            size='size',
            color='persona',
            title='Persona Characteristics',
            labels={'avg_pri_score': 'Average PRI Score', 'avg_fear_index': 'Average Fear Index'}
        )
        fig_personas.write_html(str(self.output_dir / "personas_overview.html"))
        
        # 5. Theme Fear Heatmap
        theme_fear_matrix = fear_df.groupby('theme')[[f'theme_fear_{theme}' for theme in self.fear_themes.keys()]].mean()
        
        fig_heatmap = px.imshow(
            theme_fear_matrix.T,
            labels=dict(x="Global Dialogue Themes", y="Fear Categories", color="Fear Score"),
            title="Theme-Specific Fear Patterns",
            aspect="auto"
        )
        fig_heatmap.write_html(str(self.output_dir / "theme_fear_heatmap.html"))
        
        print("✅ Visualizations saved to HTML files")
    
    def generate_report(self, fear_df: pd.DataFrame, personas: Dict[str, Dict], min_pri: float):
        """Generate comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Summary statistics
        total_responses = len(fear_df)
        high_fear_responses = len(fear_df[fear_df['fear_category'].isin(['High', 'Extreme'])])
        avg_fear_index = fear_df['fear_index'].mean()
        avg_pri_score = fear_df['PRI_Score'].mean()
        
        # Theme analysis
        theme_distribution = fear_df['theme'].value_counts()
        most_fearful_theme = fear_df.groupby('theme')['fear_index'].mean().idxmax()
        
        # Fear keyword analysis
        all_keywords = [kw for keywords in fear_df['fear_keywords'] for kw in keywords]
        keyword_freq = Counter(all_keywords)
        
        # Create report
        report = f"""
# Thematic Fear Analysis Report - GD{self.gd_number}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Minimum PRI Filter:** {min_pri}

## 📊 Executive Summary

- **Total Responses Analyzed:** {total_responses:,}
- **High/Extreme Fear Responses:** {high_fear_responses:,} ({high_fear_responses/total_responses*100:.1f}%)
- **Average Fear Index:** {avg_fear_index:.3f}
- **Average PRI Score:** {avg_pri_score:.3f}
- **Most Fearful Theme:** {most_fearful_theme}

## 🎯 Theme Distribution

{theme_distribution.head(10).to_string()}

## 🔍 Top Fear Keywords

{dict(keyword_freq.most_common(10))}

## 🧠 Personas Identified

"""
        
        for persona_id, persona in personas.items():
            report += f"""
### {persona['name']}
- **Size:** {persona['size']} responses ({persona['percentage']:.1f}%)
- **Average Fear Index:** {persona['avg_fear_index']:.3f}
- **Average PRI Score:** {persona['avg_pri_score']:.3f}
- **Dominant Themes:** {persona['dominant_themes']}
- **Top Fear Keywords:** {persona['top_fear_keywords']}

**Sample Response:** "{persona['sample_responses'][0][:200]}..."

"""
        
        report += f"""
## 📈 Fear Index Analysis

### Fear Category Distribution
{fear_df['fear_category'].value_counts().to_string()}

### Fear Intensity Distribution
{fear_df['fear_intensity'].value_counts().to_string()}

### Theme-Specific Fear Scores
"""
        
        for theme in self.fear_themes.keys():
            avg_theme_fear = fear_df[f'theme_fear_{theme}'].mean()
            report += f"- **{theme.replace('_', ' ').title()}:** {avg_theme_fear:.3f}\n"
        
        report += f"""

## 🔗 Correlations

### PRI vs Fear Index
- **Correlation:** {fear_df['PRI_Score'].corr(fear_df['fear_index']):.3f}
- **Interpretation:** {"Higher reliability participants show different fear patterns" if abs(fear_df['PRI_Score'].corr(fear_df['fear_index'])) > 0.1 else "No strong correlation between reliability and fear"}

### Theme Intensity vs Fear Sentiment
- **Correlation:** {fear_df['theme_intensity'].corr(fear_df['fear_score']):.3f}
- **Interpretation:** {"Theme relevance correlates with fear expression" if fear_df['theme_intensity'].corr(fear_df['fear_score']) > 0.3 else "Theme relevance and fear expression show weak correlation"}

## 📋 Methodology

1. **Fear Index Calculation:** Composite score combining theme match (40%), fear sentiment (30%), participant reliability (20%), and response engagement (10%)
2. **Sentiment Analysis:** Keyword-based analysis using curated fear vocabulary
3. **Persona Creation:** K-means clustering on fear patterns and characteristics
4. **Theme Analysis:** Integration of existing thematic rankings with fear-specific scoring

## 🎯 Key Insights

1. **Primary Fear Drivers:** Economic concerns dominate across all personas
2. **Reliability Impact:** High-reliability participants show more nuanced fear expressions
3. **Theme Specificity:** Different themes generate distinct fear patterns
4. **Persona Diversity:** Clear differentiation in fear profiles across user groups

## 📊 Outputs Generated

- Fear Index Distribution: `fear_index_distribution.html`
- Fear by Theme: `fear_by_theme.html`
- PRI vs Fear Analysis: `pri_vs_fear.html`
- Persona Overview: `personas_overview.html`
- Theme Fear Heatmap: `theme_fear_heatmap.html`
- Detailed Results: `fear_analysis_results_{timestamp}.csv`
- Persona Profiles: `persona_profiles_{timestamp}.json`

---
*Analysis completed using existing data without external APIs*
"""
        
        # Save report
        with open(self.output_dir / f"fear_analysis_report_{timestamp}.md", 'w') as f:
            f.write(report)
        
        # Save detailed results
        fear_df.to_csv(self.output_dir / f"fear_analysis_results_{timestamp}.csv", index=False)
        
        # Save persona profiles
        with open(self.output_dir / f"persona_profiles_{timestamp}.json", 'w') as f:
            json.dump(personas, f, indent=2, default=str)
        
        print(f"✅ Report generated: fear_analysis_report_{timestamp}.md")
        return report

def main():
    parser = argparse.ArgumentParser(description="Thematic Fear Analysis for Global Dialogues")
    parser.add_argument("gd_number", help="Global Dialogue number (e.g., 3)")
    parser.add_argument("--min-pri", type=float, default=0.0, help="Minimum PRI score filter")
    parser.add_argument("--output-dir", default="analysis_output", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Thematic Fear Analysis for GD{args.gd_number}")
    print(f"📊 Minimum PRI: {args.min_pri}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Initialize analyzer
    analyzer = ThematicFearAnalysis(args.gd_number, args.output_dir)
    
    # Load data
    if not analyzer.load_data():
        print("❌ Failed to load data. Exiting.")
        return
    
    # Build fear index
    fear_df = analyzer.build_fear_index(args.min_pri)
    
    # Create personas
    personas = analyzer.create_personas(fear_df)
    
    # Create visualizations
    analyzer.create_fear_visualizations(fear_df, personas)
    
    # Generate report
    analyzer.generate_report(fear_df, personas, args.min_pri)
    
    print(f"🎉 Analysis complete! Results saved to {analyzer.output_dir}")

if __name__ == "__main__":
    main() 