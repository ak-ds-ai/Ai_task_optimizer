# """
# Multi-Modal Emotion Analysis System
# Using:
# - GoEmotions (RoBERTa) for text/NLP emotion detection
# - FER-2013 (DeepFace) for facial emotion detection
# - Combined analysis with weighted scoring
# """

# from flask import Flask, render_template, request, jsonify, send_from_directory
# from flask_cors import CORS
# import cv2
# import numpy as np
# from deepface import DeepFace
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import base64
# from datetime import datetime
# import sqlite3
# import json
# import os
# from werkzeug.utils import secure_filename
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# import torch

# app = Flask(__name__,template_folder='templates')
# CORS(app)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Initialize NLP Model - GoEmotions RoBERTa
# print("Loading GoEmotions RoBERTa model...")
# nlp_model_name = "SamLowe/roberta-base-go_emotions"
# tokenizer = AutoTokenizer.from_pretrained(nlp_model_name)
# nlp_model = AutoModelForSequenceClassification.from_pretrained(nlp_model_name)
# emotion_classifier = pipeline(
#     "text-classification",
#     model=nlp_model,
#     tokenizer=tokenizer,
#     top_k=None,
#     device=0 if torch.cuda.is_available() else -1
# )

# # GoEmotions label mapping (28 emotions)
# GO_EMOTIONS_LABELS = [
#     'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
#     'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
#     'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
#     'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
#     'relief', 'remorse', 'sadness', 'surprise', 'neutral'
# ]

# # Map GoEmotions to 7 basic emotions for consistency
# EMOTION_MAPPING = {
#     'joy': 'happy',
#     'amusement': 'happy',
#     'excitement': 'happy',
#     'gratitude': 'happy',
#     'love': 'happy',
#     'optimism': 'happy',
#     'pride': 'happy',
#     'relief': 'happy',
#     'admiration': 'happy',
#     'approval': 'happy',
    
#     'sadness': 'sad',
#     'grief': 'sad',
#     'disappointment': 'sad',
#     'remorse': 'sad',
#     'embarrassment': 'sad',
    
#     'anger': 'angry',
#     'annoyance': 'angry',
#     'disapproval': 'angry',
    
#     'fear': 'fear',
#     'nervousness': 'fear',
    
#     'surprise': 'surprise',
#     'realization': 'surprise',
#     'curiosity': 'surprise',
    
#     'disgust': 'disgust',
    
#     'neutral': 'neutral',
#     'confusion': 'neutral',
#     'caring': 'neutral',
#     'desire': 'neutral'
# }

# # Comprehensive task recommendations
# TASK_RECOMMENDATIONS = {
#     'happy': {
#         'tasks': [
#             'Tackle high-priority strategic projects',
#             'Lead brainstorming and innovation sessions',
#             'Mentor team members and share knowledge',
#             'Present new ideas to management',
#             'Work on creative and challenging problems'
#         ],
#         'priority': 'low',
#         'alert_hr': False,
#         'category': 'productivity',
#         'motivation': 'You\'re in a great mindset! Channel this positive energy into impactful work.'
#     },
#     'sad': {
#         'tasks': [
#             'Watch motivational videos or inspiring content',
#             'Take a mindfulness break (10-15 minutes)',
#             'Work on familiar, comfortable tasks',
#             'Connect with supportive colleagues',
#             'Listen to uplifting music or podcasts',
#             'Set small, achievable goals for the day'
#         ],
#         'priority': 'high',
#         'alert_hr': True,
#         'category': 'wellbeing',
#         'motivation': 'It\'s okay to not feel 100%. Take care of yourself first.'
#     },
#     'angry': {
#         'tasks': [
#             'Take a 10-minute break immediately',
#             'Practice deep breathing exercises',
#             'Go for a short walk outside',
#             'Do physical activity or stretches',
#             'Organize your workspace',
#             'Postpone important decisions or meetings'
#         ],
#         'priority': 'high',
#         'alert_hr': True,
#         'category': 'stress_management',
#         'motivation': 'Step back and breathe. Your well-being comes first.'
#     },
#     'fear': {
#         'tasks': [
#             'Break tasks into smaller, manageable steps',
#             'Request support from experienced colleagues',
#             'Review available resources and documentation',
#             'Schedule a check-in with your manager',
#             'Practice positive affirmations',
#             'Focus on what you can control'
#         ],
#         'priority': 'high',
#         'alert_hr': True,
#         'category': 'support',
#         'motivation': 'You\'re not alone. Reach out for support when needed.'
#     },
#     'surprise': {
#         'tasks': [
#             'Document unexpected findings or insights',
#             'Share discoveries with your team',
#             'Explore related learning opportunities',
#             'Channel curiosity into research tasks',
#             'Update knowledge base or documentation'
#         ],
#         'priority': 'low',
#         'alert_hr': False,
#         'category': 'learning',
#         'motivation': 'Embrace the unexpected! New discoveries await.'
#     },
#     'disgust': {
#         'tasks': [
#             'Take a break from the current task',
#             'Switch to a different project',
#             'Discuss concerns with your team lead',
#             'Review task requirements and expectations',
#             'Seek clarification on unclear objectives'
#         ],
#         'priority': 'medium',
#         'alert_hr': False,
#         'category': 'task_adjustment',
#         'motivation': 'Sometimes a fresh perspective helps. Don\'t hesitate to ask for changes.'
#     },
#     'neutral': {
#         'tasks': [
#             'Continue with regular workflow',
#             'Review and organize your task list',
#             'Focus on routine administrative tasks',
#             'Plan for upcoming projects',
#             'Conduct code reviews or documentation'
#         ],
#         'priority': 'low',
#         'alert_hr': False,
#         'category': 'routine',
#         'motivation': 'Steady as you go. Maintain your productive momentum.'
#     }
# }

# # Database initialization
# def init_db():
#     conn = sqlite3.connect('emotion_data.db')
#     c = conn.cursor()
    
#     # Table for facial emotion logs
#     c.execute('''CREATE TABLE IF NOT EXISTS facial_emotion_logs
#                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                   user_id TEXT,
#                   timestamp TEXT,
#                   emotion TEXT,
#                   confidence REAL,
#                   age INTEGER,
#                   gender TEXT,
#                   race TEXT,
#                   image_path TEXT,
#                   hr_alerted BOOLEAN)''')
    
#     # Table for text emotion logs
#     c.execute('''CREATE TABLE IF NOT EXISTS text_emotion_logs
#                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                   user_id TEXT,
#                   timestamp TEXT,
#                   text TEXT,
#                   primary_emotion TEXT,
#                   all_emotions TEXT,
#                   confidence REAL)''')
    
#     # Table for combined analysis logs
#     c.execute('''CREATE TABLE IF NOT EXISTS combined_emotion_logs
#                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                   user_id TEXT,
#                   timestamp TEXT,
#                   facial_emotion TEXT,
#                   text_emotion TEXT,
#                   final_emotion TEXT,
#                   confidence REAL,
#                   modality TEXT,
#                   hr_alerted BOOLEAN)''')
    
#     conn.commit()
#     conn.close()

# init_db()

# def send_hr_alert(user_id, emotion, confidence, modality):
#     """Send alert to HR/Manager for negative emotions"""
#     SMTP_SERVER = "smtp.gmail.com"
#     SMTP_PORT = 587
#     SENDER_EMAIL = "your_email@gmail.com"
#     SENDER_PASSWORD = "your_password"
#     HR_EMAIL = "hr@company.com"
    
#     try:
#         message = MIMEMultipart()
#         message["From"] = SENDER_EMAIL
#         message["To"] = HR_EMAIL
#         message["Subject"] = f"🚨 Employee Wellness Alert - {emotion.upper()}"
        
#         body = f"""
#         <html>
#         <body style="font-family: Arial, sans-serif;">
#         <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; color: white; border-radius: 10px;">
#             <h2>🚨 Employee Wellness Alert</h2>
#         </div>
#         <div style="padding: 20px; background: #f5f5f5; margin-top: 10px; border-radius: 10px;">
#             <p><strong>User ID:</strong> {user_id}</p>
#             <p><strong>Detected Emotion:</strong> <span style="color: #e74c3c; font-size: 18px;">{emotion.upper()}</span></p>
#             <p><strong>Confidence:</strong> {confidence:.1f}%</p>
#             <p><strong>Detection Method:</strong> {modality}</p>
#             <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
#         </div>
#         <div style="padding: 20px; margin-top: 10px;">
#             <p>This employee may need support. Please check in with them promptly.</p>
#             <p style="color: #7f8c8d; font-size: 12px;">This is an automated alert from EmotionAI Wellness System.</p>
#         </div>
#         </body>
#         </html>
#         """
        
#         message.attach(MIMEText(body, "html"))
        
#         # Uncomment to enable email alerts
#         # with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
#         #     server.starttls()
#         #     server.login(SENDER_EMAIL, SENDER_PASSWORD)
#         #     server.send_message(message)
        
#         print(f"HR Alert: User {user_id} - {emotion} ({modality})")
#         return True
#     except Exception as e:
#         print(f"Error sending HR alert: {str(e)}")
#         return False

# def analyze_text_emotion(text):
#     """Analyze emotion from text using GoEmotions RoBERTa"""
#     try:
#         if not text or len(text.strip()) < 3:
#             return {
#                 'success': False,
#                 'error': 'Text too short for analysis'
#             }
        
#         # Get predictions from model
#         results = emotion_classifier(text)[0]
        
#         # Sort by score
#         sorted_emotions = sorted(results, key=lambda x: x['score'], reverse=True)
        
#         # Get top 5 emotions
#         top_emotions = sorted_emotions[:5]
        
#         # Map to basic 7 emotions
#         basic_emotion_scores = {}
#         for emotion_data in results:
#             label = emotion_data['label']
#             score = emotion_data['score'] * 100
            
#             basic_emotion = EMOTION_MAPPING.get(label, 'neutral')
#             if basic_emotion in basic_emotion_scores:
#                 basic_emotion_scores[basic_emotion] += score
#             else:
#                 basic_emotion_scores[basic_emotion] = score
        
#         # Normalize scores
#         total_score = sum(basic_emotion_scores.values())
#         if total_score > 0:
#             for emotion in basic_emotion_scores:
#                 basic_emotion_scores[emotion] = (basic_emotion_scores[emotion] / total_score) * 100
        
#         # Get primary emotion
#         primary_emotion = max(basic_emotion_scores, key=basic_emotion_scores.get)
        
#         return {
#             'success': True,
#             'primary_emotion': primary_emotion,
#             'confidence': basic_emotion_scores[primary_emotion],
#             'all_emotions': basic_emotion_scores,
#             'detailed_emotions': [
#                 {'emotion': e['label'], 'score': e['score'] * 100}
#                 for e in top_emotions
#             ],
#             'text_analyzed': text[:100]
#         }
#     except Exception as e:
#         return {
#             'success': False,
#             'error': str(e)
#         }

# def analyze_facial_emotion(image_path):
#     """Analyze emotion from face image using DeepFace with FER backend"""
#     try:
#         # Analyze with DeepFace (uses FER-2013 trained models)
#         analysis = DeepFace.analyze(
#             img_path=image_path,
#             actions=['emotion', 'age', 'gender', 'race'],
#             enforce_detection=False,
#             detector_backend='opencv'
#         )
        
#         if isinstance(analysis, list):
#             analysis = analysis[0]
        
#         emotion = analysis['dominant_emotion']
#         emotion_scores = analysis['emotion']
#         confidence = emotion_scores[emotion]
        
#         return {
#             'success': True,
#             'emotion': emotion,
#             'confidence': confidence,
#             'all_emotions': emotion_scores,
#             'age': analysis.get('age', 'N/A'),
#             'gender': analysis['dominant_gender'],
#             'race': analysis['dominant_race']
#         }
#     except Exception as e:
#         return {
#             'success': False,
#             'error': str(e)
#         }

# def combine_emotions(facial_data, text_data, weights={'facial': 0.6, 'text': 0.4}):
#     """Combine facial and text emotion analysis with weighted scoring"""
    
#     # If only one modality available
#     if facial_data and not text_data:
#         return facial_data['emotion'], facial_data['confidence'], 'facial_only'
#     elif text_data and not facial_data:
#         return text_data['primary_emotion'], text_data['confidence'], 'text_only'
    
#     # Combine both modalities
#     combined_scores = {}
    
#     # Add facial emotion scores
#     if facial_data and facial_data.get('all_emotions'):
#         for emotion, score in facial_data['all_emotions'].items():
#             combined_scores[emotion] = score * weights['facial']
    
#     # Add text emotion scores
#     if text_data and text_data.get('all_emotions'):
#         for emotion, score in text_data['all_emotions'].items():
#             if emotion in combined_scores:
#                 combined_scores[emotion] += score * weights['text']
#             else:
#                 combined_scores[emotion] = score * weights['text']
    
#     # Get final emotion
#     final_emotion = max(combined_scores, key=combined_scores.get)
#     final_confidence = combined_scores[final_emotion]
    
#     return final_emotion, final_confidence, 'multimodal'

# def save_analysis_log(user_id, facial_data, text_data, final_emotion, confidence, modality, hr_alerted):
#     """Save analysis to database"""
#     conn = sqlite3.connect('emotion_data.db')
#     c = conn.cursor()
    
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
#     # Save facial emotion if available
#     if facial_data:
#         c.execute('''INSERT INTO facial_emotion_logs 
#                      (user_id, timestamp, emotion, confidence, age, gender, race, image_path, hr_alerted)
#                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
#                   (user_id, timestamp, facial_data.get('emotion'), 
#                    facial_data.get('confidence'), facial_data.get('age'),
#                    facial_data.get('gender'), facial_data.get('race'),
#                    facial_data.get('image_path', ''), hr_alerted))
    
#     # Save text emotion if available
#     if text_data:
#         c.execute('''INSERT INTO text_emotion_logs 
#                      (user_id, timestamp, text, primary_emotion, all_emotions, confidence)
#                      VALUES (?, ?, ?, ?, ?, ?)''',
#                   (user_id, timestamp, text_data.get('text_analyzed', ''),
#                    text_data.get('primary_emotion'), 
#                    json.dumps(text_data.get('all_emotions', {})),
#                    text_data.get('confidence')))
    
#     # Save combined analysis
#     c.execute('''INSERT INTO combined_emotion_logs 
#                  (user_id, timestamp, facial_emotion, text_emotion, final_emotion, confidence, modality, hr_alerted)
#                  VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
#               (user_id, timestamp,
#                facial_data.get('emotion') if facial_data else None,
#                text_data.get('primary_emotion') if text_data else None,
#                final_emotion, confidence, modality, hr_alerted))
    
#     conn.commit()
#     conn.close()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     """Main endpoint for multi-modal emotion analysis"""
#     try:
#         user_id = request.form.get('user_id', 'anonymous')
#         text_input = request.form.get('text', '').strip()
        
#         facial_data = None
#         text_data = None
#         image_path = None
        
#         # Handle facial emotion analysis
#         if 'image' in request.files or 'image_data' in request.form:
#             if 'image' in request.files:
#                 file = request.files['image']
#                 if file.filename != '':
#                     filename = secure_filename(f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
#                     image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#                     file.save(image_path)
            
#             elif 'image_data' in request.form:
#                 image_data = request.form['image_data']
#                 image_data = image_data.split(',')[1]
#                 image_bytes = base64.b64decode(image_data)
                
#                 filename = secure_filename(f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
#                 image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
#                 with open(image_path, 'wb') as f:
#                     f.write(image_bytes)
            
#             if image_path:
#                 facial_result = analyze_facial_emotion(image_path)
#                 if facial_result['success']:
#                     facial_data = facial_result
#                     facial_data['image_path'] = image_path
        
#         # Handle text emotion analysis
#         if text_input:
#             text_result = analyze_text_emotion(text_input)
#             if text_result['success']:
#                 text_data = text_result
        
#         # Check if we have any data to analyze
#         if not facial_data and not text_data:
#             return jsonify({
#                 'success': False,
#                 'error': 'Please provide either an image or text for analysis'
#             })
        
#         # Combine emotions
#         final_emotion, final_confidence, modality = combine_emotions(facial_data, text_data)
        
#         # Get recommendations
#         recommendations = TASK_RECOMMENDATIONS.get(final_emotion, TASK_RECOMMENDATIONS['neutral'])
        
#         # Check HR alert
#         hr_alerted = False
#         if recommendations['alert_hr']:
#             hr_alerted = send_hr_alert(user_id, final_emotion, final_confidence, modality)
        
#         # Save to database
#         save_analysis_log(user_id, facial_data, text_data, final_emotion, 
#                          final_confidence, modality, hr_alerted)
        
#         # Prepare response
#         response = {
#             'success': True,
#             'final_emotion': final_emotion,
#             'final_confidence': final_confidence,
#             'modality': modality,
#             'facial_analysis': facial_data if facial_data else None,
#             'text_analysis': text_data if text_data else None,
#             'recommendations': recommendations['tasks'],
#             'priority': recommendations['priority'],
#             'category': recommendations['category'],
#             'motivation': recommendations['motivation'],
#             'hr_alerted': hr_alerted,
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         }
        
#         return jsonify(response)
        
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/analyze-text', methods=['POST'])
# def analyze_text_only():
#     """Endpoint for text-only emotion analysis"""
#     try:
#         data = request.get_json()
#         text = data.get('text', '').strip()
#         user_id = data.get('user_id', 'anonymous')
        
#         if not text:
#             return jsonify({'success': False, 'error': 'No text provided'})
        
#         result = analyze_text_emotion(text)
        
#         if result['success']:
#             # Save to database
#             conn = sqlite3.connect('emotion_data.db')
#             c = conn.cursor()
#             c.execute('''INSERT INTO text_emotion_logs 
#                          (user_id, timestamp, text, primary_emotion, all_emotions, confidence)
#                          VALUES (?, ?, ?, ?, ?, ?)''',
#                       (user_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                        text, result['primary_emotion'], 
#                        json.dumps(result['all_emotions']), result['confidence']))
#             conn.commit()
#             conn.close()
        
#         return jsonify(result)
        
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/history/<user_id>')
# def get_history(user_id):
#     """Get emotion history for user"""
#     conn = sqlite3.connect('emotion_data.db')
#     c = conn.cursor()
#     c.execute('''SELECT timestamp, final_emotion, confidence, modality, hr_alerted 
#                  FROM combined_emotion_logs WHERE user_id = ? 
#                  ORDER BY timestamp DESC LIMIT 50''', (user_id,))
#     rows = c.fetchall()
#     conn.close()
    
#     history = [{
#         'timestamp': row[0],
#         'emotion': row[1],
#         'confidence': row[2],
#         'modality': row[3],
#         'hr_alerted': bool(row[4])
#     } for row in rows]
    
#     return jsonify({'success': True, 'history': history})

# @app.route('/analytics/<user_id>')
# def get_analytics(user_id):
#     """Get emotion analytics for user"""
#     conn = sqlite3.connect('emotion_data.db')
#     c = conn.cursor()
    
#     # Overall emotion distribution
#     c.execute('''SELECT final_emotion, COUNT(*) as count 
#                  FROM combined_emotion_logs WHERE user_id = ? 
#                  GROUP BY final_emotion''', (user_id,))
#     emotion_dist = {row[0]: row[1] for row in c.fetchall()}
    
#     # Modality distribution
#     c.execute('''SELECT modality, COUNT(*) as count 
#                  FROM combined_emotion_logs WHERE user_id = ? 
#                  GROUP BY modality''', (user_id,))
#     modality_dist = {row[0]: row[1] for row in c.fetchall()}
    
#     # HR alerts count
#     c.execute('''SELECT COUNT(*) FROM combined_emotion_logs 
#                  WHERE user_id = ? AND hr_alerted = 1''', (user_id,))
#     hr_alerts = c.fetchone()[0]
    
#     conn.close()
    
#     return jsonify({
#         'success': True,
#         'emotion_distribution': emotion_dist,
#         'modality_distribution': modality_dist,
#         'total_hr_alerts': hr_alerts
#     })

# @app.route('/dashboard')
# def dashboard():
#     return render_template('dashboard.html')

# @app.route('/dashboard-stats')
# def dashboard_stats():
#     conn = sqlite3.connect('emotion_data.db')
#     c = conn.cursor()

#     # Count active alerts (example: count 'needs attention' status)
#     c.execute("SELECT COUNT(*) FROM combined_emotion_logs WHERE hr_alerted = 1")
#     active_alerts = c.fetchone()[0]

#     # Total monitored employees
#     c.execute("SELECT COUNT(DISTINCT user_id) FROM combined_emotion_logs")
#     total_employees = c.fetchone()[0]

#     # Today's analyses
#     today = datetime.now().strftime('%Y-%m-%d')
#     c.execute("SELECT COUNT(*) FROM combined_emotion_logs WHERE DATE(timestamp) = ?", (today,))
#     today_analyses = c.fetchone()[0]

#     # Average wellbeing score (example: average confidence)
#     c.execute("SELECT AVG(confidence) FROM combined_emotion_logs")
#     avg_wellbeing = round(c.fetchone()[0] or 0, 2)

#     conn.close()

#     return jsonify({
#         'active_alerts': active_alerts,
#         'total_employees': total_employees,
#         'today_analyses': today_analyses,
#         'avg_wellbeing': avg_wellbeing
#     })


# if __name__ == '__main__':
#     print("\n" + "="*60)
#     print("🚀 EmotionAI Multi-Modal System Starting...")
#     print("="*60)
#     print("✅ GoEmotions RoBERTa Model Loaded")
#     print("✅ DeepFace (FER-2013) Ready")
#     print("✅ Multi-Modal Analysis Enabled")
#     print("="*60)
#     print("\n🌐 Server running at: http://localhost:5000")
#     print("📊 Dashboard at: http://localhost:5000/dashboard\n")
    
#     app.run(debug=True, host='0.0.0.0', port=5000)






"""
EmotionAI Multi-Page Web Application
Flask Backend with Multi-Modal Emotion Analysis
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import cv2
import numpy as np
from deepface import DeepFace
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import base64
from datetime import datetime
import sqlite3
import json
import os
from werkzeug.utils import secure_filename
import torch

app = Flask(__name__,template_folder='templates')
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize NLP Model - GoEmotions RoBERTa
print("Loading GoEmotions RoBERTa model...")
nlp_model_name = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(nlp_model_name)
nlp_model = AutoModelForSequenceClassification.from_pretrained(nlp_model_name)
emotion_classifier = pipeline(
    "text-classification",
    model=nlp_model,
    tokenizer=tokenizer,
    top_k=None,
    device=0 if torch.cuda.is_available() else -1
)

# Emotion mapping and task recommendations
EMOTION_MAPPING = {
    'joy': 'happy', 'amusement': 'happy', 'excitement': 'happy',
    'gratitude': 'happy', 'love': 'happy', 'optimism': 'happy',
    'pride': 'happy', 'relief': 'happy', 'admiration': 'happy',
    'approval': 'happy', 'sadness': 'sad', 'grief': 'sad',
    'disappointment': 'sad', 'remorse': 'sad', 'embarrassment': 'sad',
    'anger': 'angry', 'annoyance': 'angry', 'disapproval': 'angry',
    'fear': 'fear', 'nervousness': 'fear', 'surprise': 'surprise',
    'realization': 'surprise', 'curiosity': 'surprise', 'disgust': 'disgust',
    'neutral': 'neutral', 'confusion': 'neutral', 'caring': 'neutral',
    'desire': 'neutral'
}

TASK_RECOMMENDATIONS = {
    'happy': {
        'tasks': [
            'Tackle high-priority strategic projects',
            'Lead brainstorming and innovation sessions',
            'Mentor team members and share knowledge',
            'Present new ideas to management',
            'Work on creative and challenging problems'
        ],
        'priority': 'low',
        'alert_hr': False,
        'category': 'productivity',
        'motivation': 'You\'re in a great mindset! Channel this positive energy into impactful work.'
    },
    'sad': {
        'tasks': [
            'Watch motivational videos or inspiring content',
            'Take a mindfulness break (10-15 minutes)',
            'Work on familiar, comfortable tasks',
            'Connect with supportive colleagues',
            'Listen to uplifting music or podcasts',
            'Set small, achievable goals for the day'
        ],
        'priority': 'high',
        'alert_hr': True,
        'category': 'wellbeing',
        'motivation': 'It\'s okay to not feel 100%. Take care of yourself first.'
    },
    'angry': {
        'tasks': [
            'Take a 10-minute break immediately',
            'Practice deep breathing exercises',
            'Go for a short walk outside',
            'Do physical activity or stretches',
            'Organize your workspace',
            'Postpone important decisions or meetings'
        ],
        'priority': 'high',
        'alert_hr': True,
        'category': 'stress_management',
        'motivation': 'Step back and breathe. Your well-being comes first.'
    },
    'fear': {
        'tasks': [
            'Break tasks into smaller, manageable steps',
            'Request support from experienced colleagues',
            'Review available resources and documentation',
            'Schedule a check-in with your manager',
            'Practice positive affirmations',
            'Focus on what you can control'
        ],
        'priority': 'high',
        'alert_hr': True,
        'category': 'support',
        'motivation': 'You\'re not alone. Reach out for support when needed.'
    },
    'surprise': {
        'tasks': [
            'Document unexpected findings or insights',
            'Share discoveries with your team',
            'Explore related learning opportunities',
            'Channel curiosity into research tasks',
            'Update knowledge base or documentation'
        ],
        'priority': 'low',
        'alert_hr': False,
        'category': 'learning',
        'motivation': 'Embrace the unexpected! New discoveries await.'
    },
    'disgust': {
        'tasks': [
            'Take a break from the current task',
            'Switch to a different project',
            'Discuss concerns with your team lead',
            'Review task requirements and expectations',
            'Seek clarification on unclear objectives'
        ],
        'priority': 'medium',
        'alert_hr': False,
        'category': 'task_adjustment',
        'motivation': 'Sometimes a fresh perspective helps.'
    },
    'neutral': {
        'tasks': [
            'Continue with regular workflow',
            'Review and organize your task list',
            'Focus on routine administrative tasks',
            'Plan for upcoming projects',
            'Conduct code reviews or documentation'
        ],
        'priority': 'low',
        'alert_hr': False,
        'category': 'routine',
        'motivation': 'Steady as you go. Maintain your productive momentum.'
    }
}

# Database initialization
def init_db():
    conn = sqlite3.connect('emotion_data.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS facial_emotion_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT, timestamp TEXT, emotion TEXT,
                  confidence REAL, age INTEGER, gender TEXT,
                  race TEXT, image_path TEXT, hr_alerted BOOLEAN)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS text_emotion_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT, timestamp TEXT, text TEXT,
                  primary_emotion TEXT, all_emotions TEXT, confidence REAL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS combined_emotion_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT, timestamp TEXT, facial_emotion TEXT,
                  text_emotion TEXT, final_emotion TEXT, confidence REAL,
                  modality TEXT, hr_alerted BOOLEAN)''')
    
    conn.commit()
    conn.close()

init_db()

def send_hr_alert(user_id, emotion, confidence, modality):
    """Send alert to HR/Manager for negative emotions"""
    print(f"🚨 HR Alert: User {user_id} - {emotion} ({confidence:.1f}%) - {modality}")
    return True

def analyze_text_emotion(text):
    """Analyze emotion from text using GoEmotions RoBERTa"""
    try:
        if not text or len(text.strip()) < 3:
            return {'success': False, 'error': 'Text too short'}
        
        results = emotion_classifier(text)[0]
        sorted_emotions = sorted(results, key=lambda x: x['score'], reverse=True)
        top_emotions = sorted_emotions[:5]
        
        basic_emotion_scores = {}
        for emotion_data in results:
            label = emotion_data['label']
            score = emotion_data['score'] * 100
            basic_emotion = EMOTION_MAPPING.get(label, 'neutral')
            basic_emotion_scores[basic_emotion] = basic_emotion_scores.get(basic_emotion, 0) + score
        
        total_score = sum(basic_emotion_scores.values())
        if total_score > 0:
            for emotion in basic_emotion_scores:
                basic_emotion_scores[emotion] = (basic_emotion_scores[emotion] / total_score) * 100
        
        primary_emotion = max(basic_emotion_scores, key=basic_emotion_scores.get)
        
        return {
            'success': True,
            'primary_emotion': primary_emotion,
            'confidence': basic_emotion_scores[primary_emotion],
            'all_emotions': basic_emotion_scores,
            'detailed_emotions': [
                {'emotion': e['label'], 'score': e['score'] * 100}
                for e in top_emotions
            ],
            'text_analyzed': text[:100]
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def analyze_facial_emotion(image_path):
    """Analyze emotion from face image using DeepFace"""
    try:
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion', 'age', 'gender', 'race'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        emotion = analysis['dominant_emotion']
        emotion_scores = analysis['emotion']
        confidence = emotion_scores[emotion]
        
        return {
            'success': True,
            'emotion': emotion,
            'confidence': confidence,
            'all_emotions': emotion_scores,
            'age': analysis.get('age', 'N/A'),
            'gender': analysis['dominant_gender'],
            'race': analysis['dominant_race']
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def combine_emotions(facial_data, text_data, weights={'facial': 0.6, 'text': 0.4}):
    """Combine facial and text emotion analysis"""
    if facial_data and not text_data:
        return facial_data['emotion'], facial_data['confidence'], 'facial_only'
    elif text_data and not facial_data:
        return text_data['primary_emotion'], text_data['confidence'], 'text_only'
    
    combined_scores = {}
    
    if facial_data and facial_data.get('all_emotions'):
        for emotion, score in facial_data['all_emotions'].items():
            combined_scores[emotion] = score * weights['facial']
    
    if text_data and text_data.get('all_emotions'):
        for emotion, score in text_data['all_emotions'].items():
            if emotion in combined_scores:
                combined_scores[emotion] += score * weights['text']
            else:
                combined_scores[emotion] = score * weights['text']
    
    final_emotion = max(combined_scores, key=combined_scores.get)
    final_confidence = combined_scores[final_emotion]
    
    return final_emotion, final_confidence, 'multimodal'

def save_analysis_log(user_id, facial_data, text_data, final_emotion, confidence, modality, hr_alerted):
    """Save analysis to database"""
    conn = sqlite3.connect('emotion_data.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if facial_data:
        c.execute('''INSERT INTO facial_emotion_logs 
                     (user_id, timestamp, emotion, confidence, age, gender, race, image_path, hr_alerted)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (user_id, timestamp, facial_data.get('emotion'), 
                   facial_data.get('confidence'), facial_data.get('age'),
                   facial_data.get('gender'), facial_data.get('race'),
                   facial_data.get('image_path', ''), hr_alerted))
    
    if text_data:
        c.execute('''INSERT INTO text_emotion_logs 
                     (user_id, timestamp, text, primary_emotion, all_emotions, confidence)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (user_id, timestamp, text_data.get('text_analyzed', ''),
                   text_data.get('primary_emotion'), 
                   json.dumps(text_data.get('all_emotions', {})),
                   text_data.get('confidence')))
    
    c.execute('''INSERT INTO combined_emotion_logs 
                 (user_id, timestamp, facial_emotion, text_emotion, final_emotion, confidence, modality, hr_alerted)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (user_id, timestamp,
               facial_data.get('emotion') if facial_data else None,
               text_data.get('primary_emotion') if text_data else None,
               final_emotion, confidence, modality, hr_alerted))
    
    conn.commit()
    conn.close()

# ===========================
# ROUTES - Multi-Page Website
# ===========================

@app.route('/')
def home():
    """Home page with intro and navigation"""
    return render_template('home.html')

@app.route('/projects')
def projects():
    """Projects showcase page"""
    return render_template('projects.html')

@app.route('/demo')
def demo():
    """Main EmotionAI demo/analysis page"""
    return render_template('demo.html')

@app.route('/documentation')
def documentation():
    """Documentation and guides page"""
    return render_template('documentation.html')

@app.route('/references')
def references():
    """External references and resources"""
    return render_template('references.html')

@app.route('/about')
def about():
    """About and contact page"""
    return render_template('about.html')

# API Endpoints (existing functionality)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Main endpoint for multi-modal emotion analysis"""
    try:
        user_id = request.form.get('user_id', 'anonymous')
        text_input = request.form.get('text', '').strip()
        
        facial_data = None
        text_data = None
        image_path = None
        
        if 'image' in request.files or 'image_data' in request.form:
            if 'image' in request.files:
                file = request.files['image']
                if file.filename != '':
                    filename = secure_filename(f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(image_path)
            
            elif 'image_data' in request.form:
                image_data = request.form['image_data']
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                
                filename = secure_filename(f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
            
            if image_path:
                facial_result = analyze_facial_emotion(image_path)
                if facial_result['success']:
                    facial_data = facial_result
                    facial_data['image_path'] = image_path
        
        if text_input:
            text_result = analyze_text_emotion(text_input)
            if text_result['success']:
                text_data = text_result
        
        if not facial_data and not text_data:
            return jsonify({
                'success': False,
                'error': 'Please provide either an image or text for analysis'
            })
        
        final_emotion, final_confidence, modality = combine_emotions(facial_data, text_data)
        recommendations = TASK_RECOMMENDATIONS.get(final_emotion, TASK_RECOMMENDATIONS['neutral'])
        
        hr_alerted = False
        if recommendations['alert_hr']:
            hr_alerted = send_hr_alert(user_id, final_emotion, final_confidence, modality)
        
        save_analysis_log(user_id, facial_data, text_data, final_emotion, 
                         final_confidence, modality, hr_alerted)
        
        response = {
            'success': True,
            'final_emotion': final_emotion,
            'final_confidence': final_confidence,
            'modality': modality,
            'facial_analysis': facial_data if facial_data else None,
            'text_analysis': text_data if text_data else None,
            'recommendations': recommendations['tasks'],
            'priority': recommendations['priority'],
            'category': recommendations['category'],
            'motivation': recommendations['motivation'],
            'hr_alerted': hr_alerted,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/history/<user_id>')
def get_history(user_id):
    """Get emotion history for user"""
    conn = sqlite3.connect('emotion_data.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp, final_emotion, confidence, modality, hr_alerted 
                 FROM combined_emotion_logs WHERE user_id = ? 
                 ORDER BY timestamp DESC LIMIT 50''', (user_id,))
    rows = c.fetchall()
    conn.close()
    
    history = [{
        'timestamp': row[0],
        'emotion': row[1],
        'confidence': row[2],
        'modality': row[3],
        'hr_alerted': bool(row[4])
    } for row in rows]
    
    return jsonify({'success': True, 'history': history})

@app.route('/analytics/<user_id>')
def get_analytics(user_id):
    """Get emotion analytics for user"""
    conn = sqlite3.connect('emotion_data.db')
    c = conn.cursor()
    
    c.execute('''SELECT final_emotion, COUNT(*) as count 
                 FROM combined_emotion_logs WHERE user_id = ? 
                 GROUP BY final_emotion''', (user_id,))
    emotion_dist = {row[0]: row[1] for row in c.fetchall()}
    
    c.execute('''SELECT modality, COUNT(*) as count 
                 FROM combined_emotion_logs WHERE user_id = ? 
                 GROUP BY modality''', (user_id,))
    modality_dist = {row[0]: row[1] for row in c.fetchall()}
    
    c.execute('''SELECT COUNT(*) FROM combined_emotion_logs 
                 WHERE user_id = ? AND hr_alerted = 1''', (user_id,))
    hr_alerts = c.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'success': True,
        'emotion_distribution': emotion_dist,
        'modality_distribution': modality_dist,
        'total_hr_alerts': hr_alerts
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 EmotionAI Multi-Page Website Starting...")
    print("="*60)
    print("✅ GoEmotions RoBERTa Model Loaded")
    print("✅ DeepFace (FER-2013) Ready")
    print("✅ Multi-Modal Analysis Enabled")
    print("="*60)
    print("\n🌐 Website Pages:")
    print("   📍 Home: http://localhost:5000/")
    print("   📍 Projects: http://localhost:5000/projects")
    print("   📍 Demo: http://localhost:5000/demo")
    print("   📍 Documentation: http://localhost:5000/documentation")
    print("   📍 References: http://localhost:5000/references")
    print("   📍 About: http://localhost:5000/about")
    print("="*60 + "\n")
    
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
    # app.run(debug=True, host='0.0.0.0', port=5000)