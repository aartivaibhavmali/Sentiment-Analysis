import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Amazon Sentiment AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 30px;
}

.result-box {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
}

.positive {
    background-color: #e8f7ee;
    border: 2px solid #28a745;
    color: #155724;
}

.negative {
    background-color: #fdecec;
    border: 2px solid #dc3545;
    color: #721c24;
}

.neutral {
    background-color: #fff8df;
    border: 2px solid #f0ad4e;
    color: #856404;
}

.result-text {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 10px;
}

.confidence-text {
    font-size: 18px;
    opacity: 0.9;
}

.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    try:
        model = joblib.load("sentiment_model.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

    try:
        metrics = joblib.load("sentiment_metrics.pkl")
    except:
        metrics = None

    return model, metrics


model, metrics = load_model()

if model is None:
    st.stop()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# CONFIDENCE CALCULATION
# ============================================================

def calculate_confidence(model, text):
    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            classes = model.classes_
        else:
            scores = model.decision_function([text])
            scores = np.asarray(scores).flatten()
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / exp_scores.sum()
            classes = model.classes_

        confidence_df = pd.DataFrame({
            "Sentiment": classes,
            "Probability": probabilities * 100
        })
        return confidence_df
    except Exception:
        return None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🛒 Amazon Review Sentiment AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze customer reviews using NLP, TF-IDF and Tuned Linear SVM'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("📊 Model Performance")
    if metrics is not None:
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        st.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
        st.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    else:
        st.warning("Metrics file not found.")

    st.markdown("---")
    st.subheader("🤖 Model Info")
    st.write("Algorithm: Tuned Linear SVM")
    st.write("Feature Extraction: TF-IDF")


# ============================================================
# MAIN INPUT SECTION
# ============================================================

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("✍️ Enter Customer Review")
    review = st.text_area("Review", height=150, placeholder="Type your review here...")
    analyze = st.button("🔍 Analyze Sentiment", use_container_width=True)

with col2:
    st.subheader("💡 Examples")
    st.info("😊 Positive\n\nExcellent product!")
    st.error("😞 Negative\n\nWorst experience ever.")
    st.warning("😐 Neutral\n\nIt's okay, average.")


# ============================================================
# ANALYSIS
# ============================================================

if analyze:
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        cleaned_review = clean_text(review)
        prediction = model.predict([cleaned_review])[0]
        sentiment = str(prediction).capitalize()
        
        # Fallback for integer labels (0, 1, 2)
        if sentiment not in ["Positive", "Negative", "Neutral"]:
            mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
            sentiment = mapping.get(prediction, "Neutral")
            prediction = sentiment

        confidence_df = calculate_confidence(model, cleaned_review)
        
        predicted_confidence = 0.0
        if confidence_df is not None:
            row = confidence_df[confidence_df["Sentiment"] == prediction]
            if not row.empty:
                predicted_confidence = row.iloc[0]["Probability"]

        # ----------------------------------------------------
        # RESULT DISPLAY (FIXED SYNTAX)
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader("🎯 Sentiment Prediction Result")

        # Determine the visual text and color class
        if sentiment == "Positive":
            display_text = "😊 Positive"
            css_class = "positive"
        elif sentiment == "Negative":
            display_text = "😞 Negative"
            css_class = "negative"
        else:
            display_text = "😐 Neutral"
            css_class = "neutral"

        st.markdown(
            f"""
            <div class="result-box {css_class}">
                <div class="result-text">
                    {display_text}
                </div>
                <div class="confidence-text">
                    Model Confidence: <b>{predicted_confidence:.2f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # PROBABILITY GRAPH (SHOW ONLY PREDICTED SENTIMENT)
        # ----------------------------------------------------
        if confidence_df is not None
