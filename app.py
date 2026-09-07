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
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
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
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 10px;
}

.confidence-text {
    font-size: 18px;
    margin-top: 5px;
    opacity: 0.9;
}

.metric-card {
    padding: 18px;
    border-radius: 12px;
    background-color: white;
    text-align: center;
    border: 1px solid #e5e7eb;
}

.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 40px;
}

/* Custom style for probability bars */
.prob-label {
    display: flex;
    justify-content: space-between;
    font-weight: 600;
    margin-bottom: 5px;
    font-size: 14px;
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

# Stop execution if model failed to load
if model is None:
    st.stop()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""
    
    text = str(text).lower()

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep letters and numbers
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# CONFIDENCE CALCULATION (PROBABILITY)
# ============================================================

def calculate_confidence(model, text):
    """
    Calculates probability/confidence for each class.
    Uses decision_function + Softmax for SVMs where predict_proba might not be available.
    """
    try:
        # Try predict_proba first (e.g. for Logistic Regression or calibrated SVM)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            classes = model.classes_
        else:
            # Fallback for Linear SVMs using decision_function
            scores = model.decision_function([text])
            scores = np.asarray(scores).flatten()
            
            # Apply Softmax to get pseudo-probabilities
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / exp_scores.sum()
            classes = model.classes_

        confidence_df = pd.DataFrame({
            "Sentiment": classes,
            "Probability": probabilities * 100  # Convert to percentage
        })

        # Sort by probability descending
        confidence_df = confidence_df.sort_values(by="Probability", ascending=False)
        return confidence_df

    except Exception as e:
        # st.error(f"Error calculating confidence: {e}") # Optional: Hide to keep UI clean
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

        st.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        st.metric(
            "Precision",
            f"{metrics['precision'] * 100:.2f}%"
        )

        st.metric(
            "Recall",
            f"{metrics['recall'] * 100:.2f}%"
        )

        st.metric(
            "Macro F1",
            f"{metrics['macro_f1'] * 100:.2f}%"
        )

        st.metric(
            "Weighted F1",
            f"{metrics['weighted_f1'] * 100:.2f}%"
        )

    else:

        st.warning(
            "Metrics file not found. "
            "Run the model evaluation code first."
        )

    st.markdown("---")

    st.subheader("🤖 Model")

    st.write("Algorithm: Tuned Linear SVM")
    st.write("Feature Extraction: TF-IDF")
    st.write("Task: 3-Class Sentiment Classification")

    st.markdown("---")

    st.info(
        "Model accuracy represents performance on the "
        "test dataset. Confidence represents the model's "
        "strength for the current review."
    )


# ============================================================
# MAIN INPUT SECTION
# ============================================================

col1, col2 = st.columns([1.5, 1])


with col1:

    st.subheader("✍️ Enter Customer Review")

    review = st.text_area(
        "Review",
        placeholder=(
            "Example: The product quality is excellent, "
            "battery life is amazing and I really love it!"
        ),
        height=220,
        label_visibility="collapsed"
    )

    analyze = st.button(
        "🔍 Analyze Sentiment",
        use_container_width=True
    )


with col2:

    st.subheader("💡 Example Reviews")

    st.info(
        "😊 Positive\n\n"
        "This product is excellent. I really love the quality."
    )

    st.error(
        "😞 Negative\n\n"
        "Very poor quality. The product stopped working."
    )

    st.warning(
        "😐 Neutral\n\n"
        "The product is okay. Nothing special."
    )


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if not review.strip():

        st.warning("Please enter a customer review.")

    else:

        # Clean text
        cleaned_review = clean_text(review)

        if not cleaned_review:

            st.warning(
                "The review does not contain usable text."
            )

        else:

            # Prediction
            prediction = model.predict([cleaned_review])[0]
            
            # Normalize prediction to string for display
            # If model returns integers, we might need a map, 
            # but usually scikit-learn returns class labels if trained on strings.
            # We assume prediction is the label (e.g., 'positive' or 'neutral')
            sentiment = str(prediction).capitalize()
            
            # Ensure sentiment matches one of the 3 expected classes for CSS
            if sentiment not in ["Positive", "Negative", "Neutral"]:
                # Fallback if model returns unexpected integers (0,1,2)
                mapping = {0: "Negative", 1: "Neutral", 2: "Positive"} # Common order
                sentiment = mapping.get(prediction, "Neutral")
                prediction = sentiment # Update prediction for lookup

            # Confidence / Probability
            confidence_df = calculate_confidence(
                model,
                cleaned_review
            )

            predicted_confidence = 0.0
            if confidence_df is not None:
                # Find confidence for the predicted class
                # Note: We compare exact string match. Ensure prediction type matches df type.
                row = confidence_df[confidence_df["Sentiment"] == prediction]
                if not row.empty:
                    predicted_confidence = row.iloc[0]["Probability"]
                else:
                    predicted_confidence = None


            # =================================================
            # RESULT
            # =================================================

            st.markdown("---")

            st.subheader("🎯 Prediction Result")

            # Determine CSS class based on sentiment
            css_class = sentiment.lower() if sentiment.lower() in ["positive", "negative", "neutral"] else "neutral"

            # Format confidence score safely
            conf_display = f"{predicted_confidence:.2f}%" if predicted_confidence is not None else "N/A"

            st.markdown(
                f"""
                <div class="result-box {css_class}">
                    <div class="result-text">
                        {{"😊 Positive" if sentiment == "Positive" else "😞 Negative" if sentiment == "Negative" else "😐 Neutral"}}
                    </div>
                    <div class="confidence-text">
                        Model Confidence: 
                        <b>{conf_display}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # PROBABILITY OF ACCURACY (NEW FEATURE)
            # =================================================
            
            if confidence_df is not None:
                st.markdown("---")
                st.subheader("📊 Probability of Accuracy (Breakdown)")
                
                st.write("The model's confidence levels for all possible sentiments:")
                
                # Create a progress bar for each sentiment
                for _, row in confidence_df.iterrows():
                    label = str(row["Sentiment"]).capitalize()
                    prob = row["Probability"]
                    
                    # Color logic for progress bar
                    if label == "Positive":
                        bar_color = "green"
                    elif label == "Negative":
                        bar_color = "red"
                    else:
                        bar_color = "orange"
                        
                    st.markdown(
                        f"""
                        <div class="prob-label">
                            <span>{label}</span>
                            <span>{prob:.2f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.progress(prob / 100)
                    
                # Also show the dataframe for clarity
                with st.expander("View Raw Probability Data"):
                    display_df = confidence_df.copy()
                    display_df["Probability"] = display_df["Probability"].round(2)
                    st.dataframe(display_df, use_container_width=True, hide_index=True)


            # =================================================
            # REVIEW STATISTICS
            # =================================================

            st.markdown("---")

            st.subheader("📝 Review Statistics")

            word_count = len(review.split())
            character_count = len(review)

            stat1, stat2, stat3 = st.columns(3)

            with stat1:
                st.metric("Words", word_count)

            with stat2:
                st.metric("Characters", character_count)

            with stat3:
                st.metric("Clean Words", len(cleaned_review.split()))


            # =================================================
            # CLEANED TEXT
            # =================================================

            with st.expander("🔎 View Preprocessed Text"):
                st.write(cleaned_review)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

with st.expander("⚙️ About This NLP Model"):

    st.write("""
    **Text Preprocessing**
    - Lowercase conversion
    - HTML removal
    - URL removal
    - Special-character cleaning
    - Stop-word removal

    **Feature Engineering**
    - TF-IDF vectorization
    - Unigrams, bigrams and trigrams
    - Sublinear TF

    **Machine Learning**
    - Logistic Regression
    - Multinomial Naive Bayes
    - Random Forest
    - Linear SVM

    **Final Model**
    - Tuned Linear SVM
    - Hyperparameter tuning using GridSearchCV
    - 5-fold Stratified Cross Validation
    - Macro F1 used for optimization
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'Built with Python • NLTK • TF-IDF • Scikit-learn • Streamlit'
    '</div>',
    unsafe_allow_html=True
)
