
import streamlit as st
import joblib
import re

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="🛒",
    layout="centered"
)


# ============================================================
# LOAD MODEL AND TF-IDF VECTORIZER
# ============================================================

@st.cache_resource
def load_files():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return model, vectorizer


model, vectorizer = load_files()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Apply the same preprocessing used during model training.
    """

    text = str(text).lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ============================================================
# SENTIMENT CONVERSION
# ============================================================

def get_sentiment(prediction):
    """
    Convert model prediction into Positive / Negative / Neutral.

    Handles both:
    1. String labels
    2. Numeric labels
    """

    # Convert numpy values to normal Python values
    prediction = prediction.item() if hasattr(prediction, "item") else prediction

    # --------------------------------------------------------
    # STRING LABELS
    # --------------------------------------------------------

    if isinstance(prediction, str):

        label = prediction.lower().strip()

        if label in ["positive", "pos"]:
            return "Positive"

        elif label in ["negative", "neg"]:
            return "Negative"

        elif label in ["neutral", "neu"]:
            return "Neutral"

        else:
            return str(prediction)


    # --------------------------------------------------------
    # NUMERIC LABELS
    # --------------------------------------------------------

    elif isinstance(prediction, (int, float)):

        # Change this mapping ONLY if your training labels
        # use a different encoding.

        if prediction == 0:
            return "Negative"

        elif prediction == 1:
            return "Neutral"

        elif prediction == 2:
            return "Positive"

        else:
            return str(prediction)

    return str(prediction)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🛒 Amazon Review Sentiment Analysis")

st.write(
    "Enter an Amazon product review and the NLP model will "
    "predict whether the sentiment is Positive, Negative, or Neutral."
)


# ============================================================
# USER INPUT
# ============================================================

review = st.text_area(
    "Enter your review:",
    placeholder=(
        "Example: This product is amazing and the quality is excellent!"
    ),
    height=150
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("🔍 Analyze Sentiment"):

    if not review.strip():

        st.warning("⚠️ Please enter a review.")

    else:

        # ----------------------------------------------------
        # STEP 1: CLEAN TEXT
        # ----------------------------------------------------

        cleaned_review = clean_text(review)

        # ----------------------------------------------------
        # STEP 2: TF-IDF TRANSFORMATION
        # ----------------------------------------------------

        review_tfidf = vectorizer.transform([cleaned_review])

        # ----------------------------------------------------
        # STEP 3: MODEL PREDICTION
        # ----------------------------------------------------

        raw_prediction = model.predict(review_tfidf)[0]

        # ----------------------------------------------------
        # STEP 4: CONVERT TO SENTIMENT
        # ----------------------------------------------------

        sentiment = get_sentiment(raw_prediction)

        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.subheader("Prediction Result")

        if sentiment == "Positive":

            st.success("😊 Positive Review")

        elif sentiment == "Negative":

            st.error("😞 Negative Review")

        elif sentiment == "Neutral":

            st.info("😐 Neutral Review")

        else:

            st.warning(f"Prediction: {sentiment}")


# ============================================================
# DEBUG INFORMATION
# ============================================================

with st.expander("🔧 Model Information"):

    st.write("Model:", type(model).__name__)

    st.write("Model classes:", model.classes_)

    st.write("TF-IDF vocabulary size:", len(vectorizer.vocabulary_))


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Built using NLP, TF-IDF and Linear SVM"
)
