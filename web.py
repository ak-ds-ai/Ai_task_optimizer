import streamlit as st
from deepface import DeepFace
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

st.title("🧑‍💼 AI Employee Mood & Task Recommender")
st.write("Analyze your mood (via webcam or upload) and get recommended tasks tailored to your current state.")

# ---- Helper Functions ----

def analyze_face_emotion(image_array):
    try:
        # DeepFace requires either path or NumPy array, RGB format
        result = DeepFace.analyze(
            img_path=image_array,
            actions=['emotion', 'age', 'gender'],
            enforce_detection=False
        )
        if isinstance(result, list):
            if not result:
                return {'success': False, 'error': 'No face detected'}
            result = result[0]
        emotions = result['emotion']
        dominant_emotion = result['dominant_emotion']
        positive_emotions = ['happy', 'surprise']
        negative_emotions = ['angry', 'disgust', 'fear', 'sad']
        positive_score = sum(emotions.get(e, 0) for e in positive_emotions) / 100
        negative_score = sum(emotions.get(e, 0) for e in negative_emotions) / 100
        neutral_score = emotions.get('neutral', 0) / 100
        stress_score = (emotions.get('fear', 0) + emotions.get('sad', 0)) / 200
        scores = {'Positive': positive_score, 'Negative': negative_score, 'Neutral': neutral_score}
        category = max(scores, key=scores.get)
        return {
            'success': True,
            'emotions': emotions,
            'dominant_emotion': dominant_emotion,
            'category': category,
            'positive_score': positive_score,
            'negative_score': negative_score,
            'neutral_score': neutral_score,
            'stress_score': stress_score,
            'needs_attention': stress_score > 0.3 or negative_score > 0.5,
            'age': result.get('age'),
            'gender': result.get('dominant_gender')
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def recommend_task(category, stress_score, positive_score):
    if category == 'Positive' and stress_score < 0.2:
        level = 'High'
        tasks = ["Work on challenging project", "Lead a meeting", "Deep work session"]
    elif category == 'Negative' or stress_score > 0.3:
        level = 'Low'
        tasks = ["Take a short break", "Mindfulness activity", "Review light tasks"]
    else:
        level = 'Medium'
        tasks = ["Reply to emails", "Attend a routine meeting", "Organize workspace"]
    return {'energy_level': level, 'recommended_tasks': tasks}

# ---- Streamlit UI ----

# Option for webcam or file upload
source = st.radio("Select input method:", ("📤 Upload Image", "📷 Use Webcam"))

image = None

if source == "📤 Upload Image":
    upfile = st.file_uploader("Upload a JPG/JPEG/PNG face image", type=['jpg', 'jpeg', 'png'])
    if upfile is not None:
        img_pil = Image.open(upfile).convert("RGB")
        image = np.array(img_pil)
        st.image(img_pil, caption="Uploaded Image", use_container_width=True)
elif source == "📷 Use Webcam":
    camera_photo = st.camera_input("Take a photo using webcam")
    if camera_photo is not None:
        img_pil = Image.open(camera_photo).convert("RGB")
        image = np.array(img_pil)
        st.image(img_pil, caption="Webcam Capture", use_container_width=True)

if image is not None:
    with st.spinner("Analyzing emotions..."):
        result = analyze_face_emotion(image)
    if result['success']:
        st.success("Emotion analysis successful!")
        st.write(f"**Dominant Emotion:** {result['dominant_emotion']}")
        st.write(f"**Category:** {result['category']}")
        st.write(f"**Age:** {result['age']} | **Gender:** {result['gender']}")
        # Display scores
        st.write("#### Emotion Scores")
        emotion_scores_df = pd.DataFrame(list(result['emotions'].items()), columns=['Emotion', 'Score'])
        st.dataframe(emotion_scores_df)
        st.write(f"**Positive Score:** {result['positive_score']:.2f}")
        st.write(f"**Negative Score:** {result['negative_score']:.2f}")
        st.write(f"**Stress Score:** {result['stress_score']:.2f}")
        # Recommendations
        recs = recommend_task(result['category'], result['stress_score'], result['positive_score'])
        st.write(f"**Energy Level Recommended:** {recs['energy_level']}")
        st.write("#### Task Recommendations")
        for t in recs['recommended_tasks']:
            st.write(f"- {t}")
        if result['needs_attention']:
            st.warning("⚠️ Attention: Consider a break or seek support.")
    else:
        st.error(f"Error in emotion analysis: {result['error']}")
else:
    st.info("Please upload an image or use your webcam.")

