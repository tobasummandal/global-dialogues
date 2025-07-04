"""
AI Personas Website - Main Flask Application
A futuristic AI-themed website for discovering your dialogue persona
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import json
import random
import os
from datetime import datetime
import uuid

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')
CORS(app)

# Load persona data
with open('persona_results.json', 'r') as f:
    PERSONA_DATA = json.load(f)

# Quiz questions based on persona characteristics
QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "When engaging in online discussions about AI, you typically:",
        "options": [
            {"text": "Participate actively and share your thoughts frequently", "weights": {"participation_level": 0.9}},
            {"text": "Read everything but only comment when you have something important to say", "weights": {"participation_level": 0.5, "engagement_depth": 0.7}},
            {"text": "Lurk and observe, rarely participating", "weights": {"participation_level": 0.1}},
            {"text": "Jump in with quick reactions and votes", "weights": {"participation_level": 0.6, "engagement_depth": 0.2}}
        ]
    },
    {
        "id": 2,
        "question": "When you encounter a popular opinion about AI that differs from yours, you:",
        "options": [
            {"text": "Usually find myself agreeing with the majority after considering their points", "weights": {"consensus_alignment": 0.8}},
            {"text": "Stick to my own perspective regardless of popular opinion", "weights": {"consensus_alignment": 0.2}},
            {"text": "Try to find a middle ground that incorporates different viewpoints", "weights": {"consensus_alignment": 0.6}},
            {"text": "Change my mind frequently based on new information", "weights": {"vote_consistency": 0.3}}
        ]
    },
    {
        "id": 3,
        "question": "Your typical response style when discussing AI topics is:",
        "options": [
            {"text": "Long, detailed explanations with examples and context", "weights": {"avg_text_length": 0.9, "engagement_depth": 0.8}},
            {"text": "Short, concise statements that get to the point", "weights": {"avg_text_length": 0.2}},
            {"text": "Balanced responses that are neither too brief nor too lengthy", "weights": {"avg_text_length": 0.5}},
            {"text": "Varied - sometimes short, sometimes long depending on the topic", "weights": {"avg_text_length": 0.6}}
        ]
    },
    {
        "id": 4,
        "question": "When discussing the future of AI, your emotional tone tends to be:",
        "options": [
            {"text": "Optimistic and excited about possibilities", "weights": {"avg_sentiment": 0.7}},
            {"text": "Cautious and concerned about potential risks", "weights": {"avg_sentiment": -0.5}},
            {"text": "Neutral and objective, focusing on facts", "weights": {"avg_sentiment": 0.0}},
            {"text": "Balanced between excitement and caution", "weights": {"avg_sentiment": 0.2}}
        ]
    },
    {
        "id": 5,
        "question": "In group discussions about AI policy, you:",
        "options": [
            {"text": "Consistently advocate for the same position", "weights": {"vote_consistency": 0.9}},
            {"text": "Adapt your stance based on new evidence presented", "weights": {"vote_consistency": 0.4}},
            {"text": "Often find yourself torn between different valid perspectives", "weights": {"vote_consistency": 0.3}},
            {"text": "Maintain core beliefs while being open to nuance", "weights": {"vote_consistency": 0.7}}
        ]
    },
    {
        "id": 6,
        "question": "When contributing to AI discussions, you prefer to:",
        "options": [
            {"text": "Share thoughtful, well-researched insights", "weights": {"engagement_depth": 0.9}},
            {"text": "Quickly vote or react to others' contributions", "weights": {"engagement_depth": 0.1}},
            {"text": "Mix both quick reactions and deeper thoughts", "weights": {"engagement_depth": 0.5}},
            {"text": "Focus on asking questions to understand better", "weights": {"engagement_depth": 0.6}}
        ]
    }
]

# User submissions storage
USER_SUBMISSIONS = []
QUIZ_RESULTS = []

@app.route('/')
def index():
    """Main landing page"""
    return render_template('index.html', personas=PERSONA_DATA['personas'])

@app.route('/personas')
def personas():
    """Persona gallery page"""
    return render_template('personas.html', personas=PERSONA_DATA['personas'])

@app.route('/quiz')
def quiz():
    """Interactive quiz page"""
    return render_template('quiz.html', questions=QUIZ_QUESTIONS)

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    """Process quiz submission and return persona result"""
    answers = request.json.get('answers', [])
    
    # Calculate persona scores
    persona_scores = {}
    for persona_id, persona in PERSONA_DATA['personas'].items():
        persona_scores[persona_id] = 0
    
    # Initialize feature weights
    user_features = {
        'consensus_alignment': 0,
        'participation_level': 0,
        'avg_sentiment': 0,
        'avg_text_length': 0,
        'vote_consistency': 0,
        'engagement_depth': 0
    }
    
    # Process answers
    for answer in answers:
        question_id = answer['question_id']
        option_index = answer['option_index']
        
        if question_id <= len(QUIZ_QUESTIONS) and option_index < len(QUIZ_QUESTIONS[question_id - 1]['options']):
            option = QUIZ_QUESTIONS[question_id - 1]['options'][option_index]
            weights = option.get('weights', {})
            
            for feature, weight in weights.items():
                user_features[feature] += weight
    
    # Normalize features
    num_questions = len(QUIZ_QUESTIONS)
    for feature in user_features:
        user_features[feature] /= num_questions
    
    # Find best matching persona
    best_persona = None
    best_score = float('inf')
    
    for persona_id, persona in PERSONA_DATA['personas'].items():
        score = 0
        for feature, user_value in user_features.items():
            persona_value = persona['characteristics'][feature]
            score += abs(user_value - persona_value)
        
        if score < best_score:
            best_score = score
            best_persona = persona_id
    
    # Generate unique result ID
    result_id = str(uuid.uuid4())
    
    # Store result
    result = {
        'id': result_id,
        'persona_id': best_persona,
        'persona': PERSONA_DATA['personas'][best_persona],
        'user_features': user_features,
        'timestamp': datetime.now().isoformat()
    }
    
    QUIZ_RESULTS.append(result)
    
    return jsonify({
        'result_id': result_id,
        'persona': PERSONA_DATA['personas'][best_persona],
        'match_score': max(0, 100 - (best_score * 100))
    })

@app.route('/result/<result_id>')
def result(result_id):
    """Show individual quiz result"""
    result = next((r for r in QUIZ_RESULTS if r['id'] == result_id), None)
    if not result:
        return render_template('error.html', message="Result not found"), 404
    
    return render_template('result.html', result=result)

@app.route('/submit_question', methods=['POST'])
def submit_question():
    """Handle user question submissions"""
    data = request.json
    
    question_data = {
        'id': str(uuid.uuid4()),
        'question': data.get('question', ''),
        'email': data.get('email', ''),
        'timestamp': datetime.now().isoformat()
    }
    
    USER_SUBMISSIONS.append(question_data)
    
    return jsonify({'status': 'success', 'message': 'Question submitted successfully!'})

@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    # Calculate statistics
    total_participants = PERSONA_DATA['total_participants']
    quiz_takers = len(QUIZ_RESULTS)
    
    # Distribution by persona
    persona_distribution = {}
    for result in QUIZ_RESULTS:
        persona_id = result['persona_id']
        persona_distribution[persona_id] = persona_distribution.get(persona_id, 0) + 1
    
    return render_template('analytics.html', 
                         total_participants=total_participants,
                         quiz_takers=quiz_takers,
                         persona_distribution=persona_distribution,
                         personas=PERSONA_DATA['personas'])

@app.route('/api/personas')
def api_personas():
    """API endpoint for persona data"""
    return jsonify(PERSONA_DATA)

@app.route('/api/quiz_stats')
def api_quiz_stats():
    """API endpoint for quiz statistics"""
    return jsonify({
        'total_quiz_takers': len(QUIZ_RESULTS),
        'total_submissions': len(USER_SUBMISSIONS),
        'persona_distribution': {
            result['persona_id']: len([r for r in QUIZ_RESULTS if r['persona_id'] == result['persona_id']])
            for result in QUIZ_RESULTS
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)