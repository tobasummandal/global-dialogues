# Semantic Analysis Guide for Global Dialogues

This guide explains how to perform semantic analysis on Global Dialogues data using GPT embeddings. There are two main approaches available:

## 1. Thematic Ranking (Existing)

**What it does**: Finds responses most similar to predefined themes using cosine similarity.

**Use case**: When you have specific themes/topics you want to explore.

### Prerequisites
- OpenAI API key (in `.env` file)
- Downloaded embeddings file

### Quick Start
```bash
# Download embeddings for GD3
make download-embeddings GD=3

# Run thematic ranking
make run-thematic-ranking GD=3
```

### How it works
1. Uses pre-computed 1024-dimensional OpenAI embeddings for all responses
2. Generates embeddings for predefined themes using OpenAI API
3. Calculates cosine similarity between response embeddings and theme embeddings
4. Ranks top 100 most similar responses for each theme

### Current Themes
- Faith and religion
- Economic impacts and jobs
- Human-AI relationships and collaboration
- Cultural integrity and diversity
- Safety and security concerns
- Governance and regulation of AI

### Output
- **File**: `analysis_output/GD<N>/thematic_rankings/thematic_rankings.csv`
- **Contains**: Response text, similarity scores, question details, participant info

## 2. K-Means Clustering (New)

**What it does**: Discovers natural groupings in responses using unsupervised clustering.

**Use case**: When you want to discover emergent themes/patterns without predefined categories.

### Prerequisites
- Downloaded embeddings file (no API key needed)
- Python packages: `scikit-learn`, `numpy`, `pandas`, `matplotlib`

### Quick Start
```bash
# Download embeddings for GD3
make download-embeddings GD=3

# Run clustering with automatic optimal cluster detection
make run-semantic-clustering GD=3

# Or specify number of clusters manually
make run-semantic-clustering GD=3 CLUSTERS=8
```

### How it works
1. Uses the same 1024-dimensional embeddings as thematic ranking
2. Automatically determines optimal number of clusters using silhouette analysis
3. Groups responses into clusters using k-means algorithm
4. Creates visualizations and analyzes cluster content

### Advanced Usage
```bash
# Run directly with Python for more options
python tools/scripts/semantic_clustering.py --gd 3 --clusters 10 --max-k 20
python tools/scripts/semantic_clustering.py --gd 3 --skip-viz  # Skip visualizations
```

### Output
**Directory**: `analysis_output/GD<N>/semantic_clustering/`

**Files**:
- `clustering_results_<timestamp>.csv` - All responses with cluster assignments
- `cluster_summary_<timestamp>.csv` - Summary of each cluster with sample responses
- `clustering_pca.png` - PCA visualization of clusters
- `clustering_tsne.png` - t-SNE visualization of clusters  
- `cluster_distribution.png` - Bar chart of cluster sizes

**CSV Columns**:
- `cluster` - Cluster ID (0, 1, 2, ...)
- `English Responses` - Response text
- `Question ID` - Question identifier
- `Participant ID` - Participant identifier

## Comparison: When to Use Each Method

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Thematic Ranking** | Testing specific hypotheses about themes | • Interpretable results<br>• Targeted analysis<br>• Quantified similarity scores | • Limited to predefined themes<br>• Requires API key<br>• May miss novel patterns |
| **K-Means Clustering** | Discovering emergent patterns | • Finds natural groupings<br>• No predefined themes needed<br>• Good for exploration<br>• No API costs | • Requires interpretation<br>• Cluster meanings not obvious<br>• Need to choose K |

## Technical Details

### Embeddings
- **Model**: OpenAI `text-embedding-3-small`
- **Dimensions**: 1024
- **Language**: English responses only
- **File size**: ~500-800MB per GD round

### Data Requirements
All responses with valid embeddings from the specified Global Dialogue round.

### Performance
- **Thematic Ranking**: ~1-2 minutes for 6 themes on GD3
- **K-Means Clustering**: ~2-5 minutes depending on cluster optimization

## Interpreting Results

### Thematic Ranking Results
- **Cosine similarity**: 0.0 (no similarity) to 1.0 (identical)
- **Typical good matches**: 0.3-0.8 similarity
- **Review top 10-20 results** per theme for pattern identification

### Clustering Results
1. **Look at cluster sizes**: Very small clusters may be outliers
2. **Read sample responses**: Identify common themes in each cluster
3. **Check visualizations**: PCA/t-SNE plots show cluster separation
4. **Silhouette score**: Higher = better-defined clusters (>0.5 is good)

## Example Workflow

### For Hypothesis Testing
```bash
# 1. Test specific themes
make run-thematic-ranking GD=3

# 2. Review similarity scores for "economic impacts and jobs"
# 3. Examine top responses to validate theme presence
```

### For Exploratory Analysis
```bash
# 1. Discover natural groupings
make run-semantic-clustering GD=3

# 2. Review cluster_summary.csv to understand each cluster
# 3. Examine visualizations for cluster quality
# 4. Use insights to guide follow-up thematic ranking
```

### Combined Approach
```bash
# 1. Start with clustering to discover themes
make run-semantic-clustering GD=3

# 2. Identify interesting clusters
# 3. Create custom themes based on cluster content
# 4. Run thematic ranking with new themes
```

## Tips and Best Practices

1. **Start with clustering** if you're unsure what themes exist
2. **Use thematic ranking** to validate specific hypotheses
3. **Combine both methods** for comprehensive analysis
4. **Check cluster quality** using silhouette scores and visualizations
5. **Read actual responses** not just summaries to understand patterns
6. **Iterate**: Use clustering results to inform new thematic queries

## Troubleshooting

**Embeddings not found**:
```bash
make download-embeddings GD=3
```

**OpenAI API errors**:
- Check `.env` file has `OPENAI_API_KEY=your_key_here`
- Verify API key is valid and has credits

**Poor clustering results**:
- Try different numbers of clusters
- Check if you have enough data (>100 responses recommended)
- Review silhouette scores for quality assessment

**Memory issues**:
- Use `--skip-viz` flag to reduce memory usage
- Process smaller subsets if needed 