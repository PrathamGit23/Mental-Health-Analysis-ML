from flask import Flask, request, render_template, jsonify
import joblib
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
import nltk

DISORDER_INFO = {
    "adhd": {
        "symptoms": [
            "Difficulty focusing",
            "Impulsivity",
            "Restlessness",
            "Poor time management"
        ],
        "resources": [
            "Structured daily routines",
            "Behavioral therapy",
            "Professional evaluation"
        ]
    },
    "depression": {
        "symptoms": [
            "Persistent sadness",
            "Low energy",
            "Loss of interest",
            "Sleep problems"
        ],
        "resources": [
            "Counseling or therapy",
            "Physical activity",
            "Social support"
        ]
    },
    "ocd": {
        "symptoms": [
            "Intrusive thoughts",
            "Compulsive behaviors",
            "Anxiety if rituals are avoided"
        ],
        "resources": [
            "CBT with exposure therapy",
            "Mindfulness techniques"
        ]
    },
    "ptsd": {
        "symptoms": [
            "Flashbacks",
            "Nightmares",
            "Hypervigilance"
        ],
        "resources": [
            "Trauma-focused therapy",
            "Grounding techniques"
        ]
    },
    "aspergers": {
        "symptoms": [
            "Difficulty with social interaction",
            "Repetitive behaviors",
            "Strong focus on specific interests"
        ],
        "resources": [
            "Social skills training",
            "Professional guidance"
        ]
    },
    "healthy": {
        "symptoms": [],
        "resources": [
            "Maintain healthy habits",
            "Continue self-care"
        ]
    }
}

# ------------------ Flask ------------------
app = Flask(__name__)

# ------------------ NLTK ------------------
nltk.download("wordnet")
nltk.download("omw-1.4")
lemmatizer = WordNetLemmatizer()

# ------------------ Load model ------------------
clf = joblib.load("logreg_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ------------------ Clean text ------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(lemmatizer.lemmatize(w) for w in text.split())

# ------------------ Predict ------------------
def predict_disorder(text, threshold=0.5):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    probs = clf.predict_proba(vec)[0]
    max_prob = probs.max()
    pred_class = clf.classes_[probs.argmax()]

    if max_prob < threshold:
        return "healthy", max_prob, probs
    return pred_class, max_prob, probs

# ------------------ Routes ------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    label, conf, probs = predict_disorder(data["text"])
    info = DISORDER_INFO.get(label, {"symptoms": [], "resources": []})

    return jsonify({
        "primary_prediction": {
            "disorder": label,
            "confidence": conf,
            "symptoms": info["symptoms"],
            "resources": info["resources"]
        },
        "all_predictions": [
            {
                "disorder": cls,
                "percentage": f"{p*100:.2f}%"
            }
            for cls, p in zip(clf.classes_, probs)
        ]
    })

# ------------------ Run ------------------
if __name__ == "__main__":
    app.run(debug=True)
