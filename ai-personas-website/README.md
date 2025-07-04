# AI Dialogue Personas Website

A beautiful, futuristic website that analyzes AI conversation patterns from the Global Dialogues dataset and creates personality archetypes. Users can take an interactive quiz to discover their AI dialogue persona.

## Live Demo

🚀 **Website**: https://sefikaozturk--ai-personas-website-run.modal.run

## Features

- **5 AI Dialogue Personas** discovered through machine learning clustering analysis
- **Interactive Quiz** with 6 questions to determine your persona
- **Futuristic UI/UX** with neural network animations and gradient effects
- **Social Sharing** capabilities for quiz results
- **Analytics Dashboard** showing persona distributions
- **User Question Submission** system for community engagement

## The 5 Personas

1. **The Balanced Moderator** (41.1%) - Highly active with balanced perspectives
2. **The Cautious Skeptic** (28.6%) - Selective participation with neutral tone
3. **The Deep Thinker** (19.7%) - Agrees with majority, moderately engaged
4. **The Optimistic Architect** (10.5%) - Thoughtful and detailed contributions
5. **The Pragmatic Realist** (0.1%) - Rare, detailed but selective engagement

## Technical Architecture

### Data Processing
- **Source**: Global Dialogues dataset (GD3, GD4) with 1,990 participants
- **Features**: 6 dimensions including consensus alignment, participation level, sentiment, text length, vote consistency, and engagement depth
- **Clustering**: K-means with optimal k=5 determined by silhouette analysis

### Backend
- **Framework**: Flask web application
- **Analysis**: Python with scikit-learn, pandas, numpy
- **Deployment**: Modal Labs serverless platform
- **Database**: In-memory storage (scalable to database)

### Frontend
- **UI Framework**: Bootstrap 5 with custom CSS
- **Animations**: Three.js neural network visualization
- **Charts**: Chart.js for analytics
- **Styling**: Futuristic cyberpunk theme with gradients and neon effects

## Installation & Setup

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/tobasummandal/global-dialogues.git
cd global-dialogues/ai-personas-website
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run persona analysis** (optional - results already included)
```bash
python persona_analysis.py
```

5. **Start the web server**
```bash
python app.py
```

6. **Open your browser**
Navigate to `http://localhost:5000`

### Cloud Deployment (Modal)

1. **Install Modal**
```bash
pip install modal
```

2. **Authenticate with Modal**
```bash
modal token new
```

3. **Deploy the application**
```bash
modal deploy deploy.py
```

## Project Structure

```
ai-personas-website/
├── app.py                 # Flask web application
├── persona_analysis.py    # Data processing and clustering
├── deploy.py             # Modal deployment configuration
├── persona_results.json  # Computed persona profiles
├── requirements.txt      # Python dependencies
├── templates/           # HTML templates
│   ├── base.html        # Base template with navigation
│   ├── index.html       # Landing page
│   ├── quiz.html        # Interactive quiz
│   ├── personas.html    # Persona gallery
│   ├── result.html      # Quiz result page
│   └── analytics.html   # Analytics dashboard
└── static/             # CSS, JavaScript, and assets
    ├── css/
    │   └── style.css   # Futuristic styling
    └── js/
        └── main.js     # Interactive features
```

## API Endpoints

- `GET /` - Landing page
- `GET /personas` - Persona gallery
- `GET /quiz` - Interactive quiz
- `POST /submit_quiz` - Process quiz answers
- `GET /result/<id>` - Individual result page
- `POST /submit_question` - Submit user questions
- `GET /analytics` - Analytics dashboard
- `GET /api/personas` - JSON API for persona data

## Data Sources

This project analyzes conversation patterns from the Global Dialogues dataset:
- **GD3**: March 2025 dialogue round
- **GD4**: June 2025 dialogue round
- **Total**: 1,990 participants across global conversations about AI

## Methodology

### Feature Engineering
1. **Consensus Alignment**: How often participant agrees with majority (0-1)
2. **Participation Level**: Total votes + text contributions (normalized)
3. **Average Sentiment**: Emotional tone using TextBlob (-1 to +1)
4. **Average Text Length**: Verbosity measure (normalized)
5. **Vote Consistency**: Stability of voting patterns (0-1)
6. **Engagement Depth**: Ratio of text contributions to votes

### Clustering Process
1. **Standardization**: Features scaled to zero mean, unit variance
2. **Optimal K Selection**: Silhouette analysis across k=2 to k=8
3. **K-means Clustering**: Best k=5 with highest silhouette score
4. **Validation**: Manual inspection of cluster characteristics

### Quiz Matching Algorithm
Each quiz question maps to weighted feature impacts. User responses generate a feature profile that's matched to personas using Euclidean distance in 6D feature space.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Enhancements

- [ ] Database integration for persistent storage
- [ ] User accounts and history tracking
- [ ] More sophisticated matching algorithms
- [ ] Additional persona dimensions
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Advanced analytics and insights

## License

This project is part of the Global Dialogues research initiative. Please see the main repository for licensing information.

## Acknowledgments

- **Global Dialogues Team** for the dataset
- **Modal Labs** for deployment platform
- **Bootstrap & Three.js** for UI components
- **Scikit-learn** for machine learning capabilities

---

Built with ❤️ for understanding AI conversation patterns

🤖 Generated with [Memex](https://memex.tech)
Co-Authored-By: Memex <noreply@memex.tech>