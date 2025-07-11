# Complete Analysis Workflows for Global Dialogues

This guide provides step-by-step workflows for different types of semantic analysis, integrating PRI (Participant Reliability Index), Global Dialogues Indicators, and various visualization approaches.

## 🚀 Quick Start - Choose Your Workflow

### 1. **Exploratory Analysis** (Discover what themes exist)
```bash
# Basic clustering to find natural patterns
make run-semantic-clustering GD=3

# Enhanced clustering with reliability filtering
make run-enhanced-analysis GD=3 PRI_FILTER=3.0
```

### 2. **Hypothesis Testing** (Test specific themes)
```bash
# Test predefined themes
make run-thematic-ranking GD=3

# Combine with clustering for validation
make run-semantic-clustering GD=3
make run-thematic-ranking GD=3
```

### 3. **Quality-Focused Analysis** (High-reliability participants only)
```bash
# Only analyze reliable participants
make run-enhanced-analysis GD=3 PRI_FILTER=4.0

# Use PRI weights in clustering algorithm
make run-enhanced-analysis GD=3 PRI_FILTER=3.0
```

### 4. **Indicators Tracking** (Track global pulse over time)
```bash
# Analyze just the indicator questions
make run-enhanced-analysis GD=3 INDICATORS=true

# Compare indicators across dialogues
make run-enhanced-analysis GD=2 INDICATORS=true
make run-enhanced-analysis GD=3 INDICATORS=true
```

---

## 📋 Prerequisites Setup

### Step 1: Generate PRI Scores
```bash
# Calculate PRI for each participant (required for enhanced analysis)
make pri GD=3

# Optional: Include LLM judge assessment (more accurate but costs money)
make pri-llm GD=3
```

### Step 2: Download Embeddings
```bash
# Download pre-computed embeddings
make download-embeddings GD=3
```

### Step 3: Install Dependencies
```bash
# Make sure you have all required packages
pip install -r requirements.txt
```

---

## 🔬 Detailed Workflows

### Workflow A: Complete Exploratory Analysis

**Goal**: Discover what themes naturally emerge from the data, with quality controls.

```bash
# 1. Generate PRI scores to assess participant reliability
make pri GD=3

# 2. Run enhanced clustering with PRI integration
make run-enhanced-analysis GD=3

# 3. Analyze only high-quality participants
make run-enhanced-analysis GD=3 PRI_FILTER=4.0

# 4. Compare results - look for stable themes across quality levels
```

**Outputs to Review**:
- `analysis_output/GD3/enhanced_semantic_analysis/enhanced_clustering_results_*.csv`
- `analysis_output/GD3/enhanced_semantic_analysis/enhanced_cluster_summary_*.csv`
- `analysis_output/GD3/enhanced_semantic_analysis/interactive_clustering_GD3.html`
- All PNG visualizations in the same directory

### Workflow B: Indicators Deep Dive

**Goal**: Track how responses to core indicator questions cluster and evolve.

```bash
# 1. Analyze indicators with PRI weighting
make run-enhanced-analysis GD=3 INDICATORS=true

# 2. Compare with previous dialogue
make run-enhanced-analysis GD=2 INDICATORS=true

# 3. Run without PRI filtering to see full spectrum
make run-enhanced-analysis GD=3 INDICATORS=true PRI_FILTER=2.0
```

**Analysis Questions**:
- Do indicators cluster consistently across GD rounds?
- Are there reliability differences in indicator responses?
- How do core attitudes (AI concern, trust, etc.) group together?

### Workflow C: Quality-First Analysis

**Goal**: Focus on most reliable participants for clean thematic insights.

```bash
# 1. Generate comprehensive PRI with LLM assessment
make pri-llm GD=3

# 2. High-reliability clustering (top 40% of participants)
make run-enhanced-analysis GD=3 PRI_FILTER=4.0

# 3. Compare high vs. medium reliability patterns
make run-enhanced-analysis GD=3 PRI_FILTER=3.0

# 4. Validate themes with thematic ranking
make run-thematic-ranking GD=3
```

### Workflow D: Cross-Dialogue Comparison

**Goal**: Compare themes and patterns across different GD rounds.

```bash
# Analyze each dialogue with same parameters
for gd in 2 3 4; do
    make pri GD=$gd
    make run-enhanced-analysis GD=$gd INDICATORS=true PRI_FILTER=3.0
done

# Then manually compare:
# - Cluster summaries across dialogues
# - PRI distributions
# - Theme stability
```

### Workflow E: Comprehensive Research Pipeline

**Goal**: Complete analysis pipeline for academic research.

```bash
# 1. Data quality assessment
make pri-llm GD=3
make export-unreliable GD=3

# 2. Full semantic analysis
make run-enhanced-analysis GD=3                    # All responses
make run-enhanced-analysis GD=3 PRI_FILTER=3.5     # Reliable only
make run-enhanced-analysis GD=3 INDICATORS=true    # Indicators only

# 3. Thematic validation
make run-thematic-ranking GD=3

# 4. Cross-validation with different cluster numbers
make run-enhanced-analysis GD=3 CLUSTERS=6 PRI_FILTER=3.5
make run-enhanced-analysis GD=3 CLUSTERS=10 PRI_FILTER=3.5
```

---

## 📊 Understanding Your Results

### Key Files and What They Tell You

#### 1. **Enhanced Cluster Summary** (`enhanced_cluster_summary_*.csv`)
**What to look for**:
- `mean_pri`: Average reliability of participants in each cluster
- `high_reliability_pct`: Percentage of highly reliable participants  
- `sample_response_*`: Representative responses for each theme
- `size`: How prevalent each theme is

#### 2. **Interactive Visualization** (`interactive_clustering_GD3.html`)
**How to use**:
- Open in web browser for interactive exploration
- Hover over points to see response previews
- Point size indicates PRI score (if available)
- Colors indicate clusters

#### 3. **PRI-Cluster Analysis** (`pri_cluster_analysis.png`)
**What it shows**:
- Whether certain themes attract more/less reliable participants
- If response quality varies by topic
- Correlation between reliability and response characteristics

#### 4. **Cluster Quality Analysis** (`cluster_quality_analysis.png`)
**What to check**:
- Balanced cluster sizes (no tiny clusters)
- Response length distributions
- Overall reliability distribution

### Key Metrics to Report

1. **Silhouette Score**: Cluster quality (>0.5 is good, >0.7 is excellent)
2. **PRI Distribution**: Participant reliability spread
3. **Cluster Stability**: Do themes persist with different parameters?
4. **Reliability by Theme**: Do certain topics attract more reliable responses?

---

## 🎯 Specific Use Cases

### Use Case 1: Academic Paper on AI Attitudes
```bash
# High-quality analysis for publication
make pri-llm GD=3
make run-enhanced-analysis GD=3 PRI_FILTER=4.0 INDICATORS=true
make run-thematic-ranking GD=3

# Report: Silhouette scores, PRI distributions, thematic consistency
```

### Use Case 2: Policy Briefing on Global AI Concerns  
```bash
# Broad analysis with reliability context
make run-enhanced-analysis GD=3 INDICATORS=true
make run-enhanced-analysis GD=4 INDICATORS=true

# Focus: Cross-dialogue trends, reliability by geography/demographics
```

### Use Case 3: Research Methodology Validation
```bash
# Test robustness across quality levels
make run-enhanced-analysis GD=3 PRI_FILTER=2.0  # Inclusive
make run-enhanced-analysis GD=3 PRI_FILTER=4.0  # Exclusive
make run-thematic-ranking GD=3                   # Validation

# Report: Theme stability, quality impact on findings
```

### Use Case 4: Platform Quality Assessment
```bash
# Focus on participation quality
make pri-llm GD=3
make export-unreliable GD=3
make run-enhanced-analysis GD=3  # Full analysis with PRI overlay

# Report: Reliability distributions, quality predictors, platform health
```

---

## 🔧 Advanced Customization

### Custom Theme Testing
```bash
# 1. Edit themes in tools/scripts/thematic_queries.txt
# 2. Add your specific themes of interest
# 3. Run thematic ranking
make run-thematic-ranking GD=3

# 4. Validate with clustering
make run-enhanced-analysis GD=3
```

### Custom PRI Thresholds
```bash
# Experiment with different quality thresholds
make run-enhanced-analysis GD=3 PRI_FILTER=2.5  # Lenient
make run-enhanced-analysis GD=3 PRI_FILTER=3.5  # Moderate  
make run-enhanced-analysis GD=3 PRI_FILTER=4.5  # Strict
```

### Cross-Dialogue Analysis
```bash
# Save results with timestamps for comparison
for gd in 1 2 3 4; do
    if [ -f "Data/GD$gd/GD${gd}_embeddings.json" ]; then
        make run-enhanced-analysis GD=$gd INDICATORS=true PRI_FILTER=3.0
    fi
done
```

---

## 🚨 Troubleshooting Common Issues

### Issue: No PRI Scores Available
**Solution**: 
```bash
make pri GD=3  # Generate basic PRI scores
# Or if you have time/budget:
make pri-llm GD=3  # Enhanced PRI with LLM assessment
```

### Issue: Poor Clustering Quality (Low Silhouette Score)
**Try**:
```bash
# Filter for higher quality participants
make run-enhanced-analysis GD=3 PRI_FILTER=3.5

# Try different cluster numbers
make run-enhanced-analysis GD=3 CLUSTERS=6
make run-enhanced-analysis GD=3 CLUSTERS=12
```

### Issue: Clusters Too Small or Unbalanced
**Solutions**:
- Reduce number of clusters: `CLUSTERS=4`
- Lower PRI filter: `PRI_FILTER=2.5`
- Include more data: Remove `INDICATORS=true`

### Issue: Themes Don't Match Expectations
**Try**:
1. Check cluster summaries for actual content
2. Validate with thematic ranking on known themes
3. Adjust PRI filtering - sometimes low-quality responses create noise

---

## 📈 Reporting Your Results

### For Academic Papers
- Report silhouette scores and cluster validation metrics
- Include PRI distribution analysis
- Show theme stability across quality thresholds
- Document methodology decisions (thresholds, filters)

### For Policy Reports  
- Focus on indicator questions and cross-dialogue trends
- Highlight reliability-weighted insights
- Show geographic/demographic clustering patterns
- Emphasize robust findings (high PRI participants)

### For Platform Assessment
- Report PRI distributions and unreliable participant rates
- Show correlation between engagement patterns and reliability
- Document quality predictors and recommendations 