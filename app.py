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
}

.positive {
    background-color: #e8f7ee;
    border: 2px solid #28a745;
}

.negative {
    background-color: #fdecec;
    border: 2px solid #dc3545;
}

.neutral {
    background-color: #fff8df;
    border: 2px solid #f0ad4e;
}

.result-text {
    font-size: 32px;
    font-weight: 700;
}

.confidence-text {
    font-size: 22px;
    margin-top: 10px;
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

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("sentiment_model.pkl")

    try:
        metrics = joblib.load("sentiment_metrics.pkl")
    except:
        metrics = None

    return model, metrics


model, metrics = load_model()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

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
# CONFIDENCE CALCULATION
# ============================================================

def calculate_confidence(model, text):

    try:

        scores = model.decision_function([text])

        scores = np.asarray(scores).flatten()

        # Softmax transformation
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / exp_scores.sum()

        classes = model.classes_

        confidence_df = pd.DataFrame({
            "Sentiment": classes,
            "Confidence": probabilities * 100
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

            sentiment = str(prediction).capitalize()

            # Confidence
            confidence_df = calculate_confidence(
                model,
                cleaned_review
            )

            if confidence_df is not None:

                predicted_confidence = confidence_df.loc[
                    confidence_df["Sentiment"] == prediction,
                    "Confidence"
                ].iloc[0]

            else:

                predicted_confidence = None


            # =================================================
            # RESULT
            # =================================================

            st.markdown("---")

            st.subheader("🎯 Prediction Result")


            if sentiment == "Positive":

                st.markdown(
                    f"""
                    <div class="result-box positive">
                        <div class="result-text">
                            😊 Positive
                        </div>
                        <div class="confidence-text">
                            Model Confidence:
                            <b>{predicted_confidence:.2f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif sentiment == "Negative":

                st.markdown(
                    f"""
                    <div class="result-box negative">
                        <div class="result-text">
                            😞 Negative
                        </div>
                        <div class="confidence-text">
                            Model Confidence:
                            <b>{predicted_confidence:.2f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-box neutral">
                        <div class="result-text">
                            😐 Neutral
                        </div>
                        <div class="confidence-text">
                            Model Confidence:
                            <b>{predicted_confidence:.2f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # =================================================
            # CONFIDENCE BREAKDOWN
            # =================================================

            if confidence_df is not None:

                st.markdown("---")

                st.subheader(
                    "📈 Sentiment Confidence Breakdown"
                )

                display_df = confidence_df.copy()

                display_df["Confidence"] = display_df[
                    "Confidence"
                ].round(2)

                st.bar_chart(
                    display_df.set_index("Sentiment")
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )


            # =================================================
            # REVIEW STATISTICS
            # =================================================

            st.markdown("---")

            st.subheader("📝 Review Statistics")

            word_count = len(review.split())
            character_count = len(review)

            stat1, stat2, stat3 = st.columns(3)

            with stat1:
                st.metric(
                    "Words",
                    word_count
                )

            with stat2:
                st.metric(
                    "Characters",
                    character_count
                )

            with stat3:
                st.metric(
                    "Clean Words",
                    len(cleaned_review.split())
                )


            # =================================================
            # CLEANED TEXT
            # =================================================

            with st.expander(
                "🔎 View Preprocessed Text"
            ):

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
