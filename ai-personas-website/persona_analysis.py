"""
Persona Analysis for Global Dialogues
Creates AI personas based on participant clustering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings('ignore')

class PersonaAnalyzer:
    def __init__(self):
        self.features = None
        self.kmeans = None
        self.scaler = None
        self.personas = {}
        self.persona_names = [
            "The Optimistic Architect",
            "The Cautious Skeptic", 
            "The Balanced Moderator",
            "The Deep Thinker",
            "The Pragmatic Realist"
        ]
        
    def load_data(self, gd_numbers=[1, 2, 3, 4]):
        """Load and merge data from multiple GD rounds"""
        all_data = []
        
        for gd_num in gd_numbers:
            try:
                # Load data files
                base_path = f"Data/GD{gd_num}/"
                
                # Load standardized aggregate data
                aggregate = pd.read_csv(f"{base_path}GD{gd_num}_aggregate_standardized.csv")
                
                # Load binary votes
                binary = pd.read_csv(f"{base_path}GD{gd_num}_binary.csv")
                
                # Load verbatim text
                verbatim = pd.read_csv(f"{base_path}GD{gd_num}_verbatim_map.csv")
                
                # Load participants data
                participants = pd.read_csv(f"{base_path}GD{gd_num}_participants.csv")
                
                # Add GD round identifier
                binary['GD_Round'] = gd_num
                verbatim['GD_Round'] = gd_num
                participants['GD_Round'] = gd_num
                
                all_data.append({
                    'aggregate': aggregate,
                    'binary': binary,
                    'verbatim': verbatim,
                    'participants': participants
                })
                
                print(f"Loaded GD{gd_num} data successfully")
                
            except Exception as e:
                print(f"Error loading GD{gd_num}: {e}")
                continue
        
        return all_data
    
    def engineer_features(self, data_list):
        """Engineer features for clustering"""
        all_features = []
        
        for data in data_list:
            binary_df = data['binary']
            verbatim_df = data['verbatim']
            participants_df = data['participants']
            
            # Get unique participants
            participants = binary_df['Participant ID'].unique()
            
            for participant_id in participants:
                try:
                    # Get participant's votes
                    participant_votes = binary_df[binary_df['Participant ID'] == participant_id]
                    
                    # Get participant's text contributions
                    participant_text = verbatim_df[verbatim_df['Participant ID'] == participant_id]
                    
                    # Feature 1: Consensus Alignment (how often they agree with majority)
                    agree_votes = len(participant_votes[participant_votes['Vote'] == 'Agree'])
                    total_votes = len(participant_votes)
                    consensus_alignment = agree_votes / total_votes if total_votes > 0 else 0.5
                    
                    # Feature 2: Participation Level (total number of interactions)
                    participation_level = total_votes + len(participant_text)
                    
                    # Feature 3: Text Sentiment (average sentiment of their contributions)
                    if len(participant_text) > 0:
                        sentiments = []
                        text_lengths = []
                        
                        for text in participant_text['Thought Text']:
                            if pd.notna(text) and text.strip():
                                blob = TextBlob(str(text))
                                sentiments.append(blob.sentiment.polarity)
                                text_lengths.append(len(str(text).split()))
                        
                        avg_sentiment = np.mean(sentiments) if sentiments else 0
                        avg_text_length = np.mean(text_lengths) if text_lengths else 0
                        text_contributions = len(participant_text)
                    else:
                        avg_sentiment = 0
                        avg_text_length = 0
                        text_contributions = 0
                    
                    # Feature 4: Voting Consistency (standard deviation of voting timing)
                    vote_consistency = 1 - (np.std([1 if v == 'Agree' else 0 for v in participant_votes['Vote']]) if total_votes > 1 else 0)
                    
                    # Feature 5: Engagement Depth (ratio of text to votes)
                    engagement_depth = text_contributions / total_votes if total_votes > 0 else 0
                    
                    all_features.append({
                        'participant_id': participant_id,
                        'consensus_alignment': consensus_alignment,
                        'participation_level': min(participation_level / 100, 1),  # Normalize
                        'avg_sentiment': avg_sentiment,
                        'avg_text_length': min(avg_text_length / 100, 1),  # Normalize
                        'vote_consistency': vote_consistency,
                        'engagement_depth': engagement_depth,
                        'text_contributions': text_contributions
                    })
                    
                except Exception as e:
                    print(f"Error processing participant {participant_id}: {e}")
                    continue
        
        return pd.DataFrame(all_features)
    
    def find_optimal_clusters(self, X, max_k=8):
        """Find optimal number of clusters using silhouette score"""
        scores = []
        K_range = range(2, max_k + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            scores.append((k, score))
        
        # Find best k
        best_k = max(scores, key=lambda x: x[1])[0]
        return best_k, scores
    
    def perform_clustering(self, features_df):
        """Perform clustering analysis"""
        # Prepare feature matrix
        feature_columns = ['consensus_alignment', 'participation_level', 'avg_sentiment', 
                          'avg_text_length', 'vote_consistency', 'engagement_depth']
        
        X = features_df[feature_columns].values
        
        # Handle missing values
        X = np.nan_to_num(X)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Find optimal number of clusters
        best_k, scores = self.find_optimal_clusters(X_scaled)
        print(f"Optimal number of clusters: {best_k}")
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to features
        features_df['cluster'] = labels
        
        # Store features for later use
        self.features = features_df
        
        return features_df, X_scaled, labels
    
    def create_persona_profiles(self, features_df):
        """Create detailed persona profiles"""
        personas = {}
        
        for cluster_id in features_df['cluster'].unique():
            cluster_data = features_df[features_df['cluster'] == cluster_id]
            
            # Calculate cluster statistics
            profile = {
                'id': cluster_id,
                'name': self.persona_names[cluster_id] if cluster_id < len(self.persona_names) else f"Persona {cluster_id}",
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(features_df) * 100,
                'characteristics': {
                    'consensus_alignment': cluster_data['consensus_alignment'].mean(),
                    'participation_level': cluster_data['participation_level'].mean(),
                    'avg_sentiment': cluster_data['avg_sentiment'].mean(),
                    'avg_text_length': cluster_data['avg_text_length'].mean(),
                    'vote_consistency': cluster_data['vote_consistency'].mean(),
                    'engagement_depth': cluster_data['engagement_depth'].mean()
                }
            }
            
            # Create persona description based on characteristics
            profile['description'] = self.generate_persona_description(profile['characteristics'])
            
            personas[cluster_id] = profile
        
        self.personas = personas
        return personas
    
    def generate_persona_description(self, characteristics):
        """Generate natural language description of persona"""
        consensus = characteristics['consensus_alignment']
        sentiment = characteristics['avg_sentiment']
        engagement = characteristics['engagement_depth']
        participation = characteristics['participation_level']
        
        # Determine primary traits
        traits = []
        
        if consensus > 0.7:
            traits.append("tends to agree with majority opinions")
        elif consensus < 0.3:
            traits.append("often dissents from popular views")
        else:
            traits.append("maintains balanced perspective on issues")
        
        if sentiment > 0.2:
            traits.append("expresses optimistic viewpoints")
        elif sentiment < -0.2:
            traits.append("takes cautious or critical stances")
        else:
            traits.append("maintains neutral emotional tone")
        
        if engagement > 0.5:
            traits.append("contributes detailed thoughts and analysis")
        elif engagement < 0.2:
            traits.append("prefers concise interactions")
        else:
            traits.append("balances brevity with depth")
        
        if participation > 0.8:
            traits.append("highly active in discussions")
        elif participation < 0.4:
            traits.append("selective in participation")
        else:
            traits.append("moderately engaged")
        
        return f"This persona {', '.join(traits)}."
    
    def create_visualizations(self, features_df, X_scaled):
        """Create interactive visualizations"""
        # PCA for 2D visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create scatter plot
        fig = px.scatter(
            x=X_pca[:, 0], 
            y=X_pca[:, 1],
            color=features_df['cluster'].astype(str),
            title="AI Dialogue Personas - Cluster Visualization",
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'},
            color_discrete_map={
                '0': '#FF6B6B',  # Coral
                '1': '#4ECDC4',  # Turquoise  
                '2': '#45B7D1',  # Blue
                '3': '#96CEB4',  # Green
                '4': '#FFEAA7'   # Yellow
            }
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    
    def save_results(self, filename="persona_results.json"):
        """Save persona analysis results"""
        # Convert numpy types to Python types for JSON serialization
        personas_json = {}
        for k, v in self.personas.items():
            personas_json[str(k)] = {
                'id': int(v['id']),
                'name': v['name'],
                'size': int(v['size']),
                'percentage': float(v['percentage']),
                'characteristics': {
                    char: float(val) for char, val in v['characteristics'].items()
                },
                'description': v['description']
            }
        
        results = {
            'personas': personas_json,
            'feature_importance': {
                'consensus_alignment': 'How often participant agrees with majority',
                'participation_level': 'Overall activity level in discussions',
                'avg_sentiment': 'Emotional tone of contributions',
                'avg_text_length': 'Typical length of written responses',
                'vote_consistency': 'Consistency in voting patterns',
                'engagement_depth': 'Ratio of thoughtful contributions to simple votes'
            },
            'total_participants': len(self.features) if self.features is not None else 0
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")
        return results

def main():
    """Main analysis pipeline"""
    print("Starting AI Dialogue Persona Analysis...")
    
    # Initialize analyzer
    analyzer = PersonaAnalyzer()
    
    # Load data
    print("Loading data...")
    data_list = analyzer.load_data([3, 4])  # Start with GD3 and GD4
    
    if not data_list:
        print("No data loaded. Please check file paths.")
        return
    
    # Engineer features
    print("Engineering features...")
    features_df = analyzer.engineer_features(data_list)
    print(f"Created features for {len(features_df)} participants")
    
    # Perform clustering
    print("Performing clustering...")
    features_df, X_scaled, labels = analyzer.perform_clustering(features_df)
    
    # Create persona profiles
    print("Creating persona profiles...")
    personas = analyzer.create_persona_profiles(features_df)
    
    # Print results
    print("\n=== PERSONA ANALYSIS RESULTS ===")
    for persona_id, persona in personas.items():
        print(f"\n{persona['name']} ({persona['size']} participants, {persona['percentage']:.1f}%)")
        print(f"Description: {persona['description']}")
        print("Key characteristics:")
        for char, value in persona['characteristics'].items():
            print(f"  - {char}: {value:.3f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig = analyzer.create_visualizations(features_df, X_scaled)
    
    # Save results
    results = analyzer.save_results()
    
    return analyzer, fig, results

if __name__ == "__main__":
    analyzer, fig, results = main()
    fig.show()