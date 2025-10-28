# Ai_task_optimizer

AI-driven system for analyzing employee emotional well-being and optimizing task assignments. Combines advanced NLP (GoEmotions + Transformers), stress detection, and visual analytics.

---

## 🚀 Project Goal

- Monitor and visualize employee emotions using text data (and optionally images).
- Detect stress, positivity, and negativity trends in teams.
- Provide dynamic task recommendations tailored to user mood and stress levels.
- Alert managers to employees needing support.

## 🔬 How it Works

1. **Data Source**: Uses the GoEmotions dataset (27 emotions, ~58k labeled texts). Option to use facial data if DeepFace or similar is added.
2. **Emotion Detection**: Applies a HuggingFace transformer model to classify employee messages into fine-grained emotions.
3. **Emotion Categorization**: Groups detailed emotions into workplace categories: Positive, Negative, Neutral, Stress.
4. **Task Recommendation**: Suggests work tasks based on identified mood and stress scores.
5. **Synthetic Data Simulation**: Generates realistic employee messages for testing, with simulated mood patterns.
6. **Alert System**: Tracks stress over days. If stress > threshold for 2+ consecutive days, raises an alert and suggests intervention.
7. **Visualization**: Plots emotion distributions and team trends with Plotly/Seaborn.

## 💡 Example Workflow

- Employee sends message → Emotion model predicts detailed emotions
- Categorization function maps these to high-level mood
- Task recommender advises work based on current mood (creative for positive, breaks for stressed, etc.)
- Team dashboard tracks everyone’s mood/stress over time
- Alerts highlight anyone at risk or needing a check-in

## 🧩 Main Libraries & Tools

- `transformers`, `torch`: Deep learning for emotion recognition.
- `datasets`: Loads GoEmotions.
- `pandas`, `numpy`: Data handling.
- `matplotlib`, `seaborn`, `plotly`: Data visualization.
- `Streamlit`: Interactive dashboard (planned).
- `TextBlob`, `re`: Extra text processing.
- `DeepFace` (optional): Facial emotion detection.

## 🗂 Project Structure

- `app.py`: Main app logic
- `models/`: ML model files
- `data/`: Sample and synthetic data
- `utils/`: Helper functions/scripts

## 🛠️ Usage

1. Clone repo, install requirements with `pip install -r requirements.txt`.
2. Run `app.py` (or Streamlit app) and follow instructions.
3. Optionally, feed in new team messages or images for analysis.

## 🙋‍♂️ Q&A Preparation

- Know the goal: Emotional intelligence for workforce wellness.
- Why GoEmotions/Transformers: Powerful, fine-grained emotion detection on natural text.
- Stress alert logic: High stress for consecutive days = flag/alert.
- Task assignment: Recommends appropriate workload/actions based on mood.
- Synthetic data: Simulates usage for testing/demo.

## 📄 License

MIT

---

*Ready for real-world well-being analytics and smarter work allocation!*
