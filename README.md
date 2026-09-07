# 🛒 Amazon Review Sentiment Analysis AI
A powerful Machine Learning web application designed to analyze customer reviews from Amazon. It classifies text into three distinct sentiment categories: Positive, Negative, and Neutral. Built using Natural Language Processing (NLP), TF-IDF, and a Tuned Linear SVM model.

# 🌟 Key Features
- 3-Class Sentiment Classification: Accurately predicts if a review is Positive, Negative, or Neutral.
- Real-time Analysis: Instant feedback as you type or paste customer reviews.
-Probability Visualization: Displays the confidence score (probability) of the prediction with an interactive progress bar.
- Text Preprocessing: Automatically cleans HTML tags, URLs, and special characters for better accuracy.
- Model Insights: View model performance metrics (Accuracy, Precision, Recall, F1-Score) directly in the sidebar.

# 🛠️ Tech Stack
This project utilizes the following technologies and libraries:

Language: Python 3.8+
Web Framework: Streamlit
Machine Learning: Scikit-learn (LinearSVC, TfidfVectorizer)
Data Manipulation: Pandas, NumPy
Model Serialization: Joblib
Regular Expressions: Re

# 📂 Project Structure
├── app.py                   # Main Streamlit application file
├── sentiment_model.pkl      # Serialized Tuned Linear SVM Model
├── sentiment_metrics.pkl    # Serialized Model Evaluation Metrics
├── requirements.txt         # Python dependencies
├── P652-Dataset.xlsx        # Raw dataset (Excel)
├── Processed_dataset.csv    # Cleaned dataset (CSV)
└── README.md                # Project documentation

# Deployment Link
https://sentiment-analysis-review-checker.streamlit.app/
The application will automatically open in your default web browser 

# 💻 How to Use
1. Enter Review: Type or paste an Amazon product review into the text area on the left.
2. Analyze: Click the "🔍 Analyze Sentiment" button.
# 3. View Results:
The main result box will show the predicted sentiment (Positive, Negative, or Neutral) with a confidence percentage.
The "Probability Graph" section shows a visual bar representing the accuracy of that specific prediction.
Review statistics (word count, character count) are displayed at the bottom.

## 🤖 Model Methodology
# 1. Text Preprocessing:
-Conversion to lowercase.
-Removal of HTML tags, URLs, and non-alphanumeric characters.
-Whitespace normalization.

# 2.Feature Engineering:
-TF-IDF Vectorization: Converts text into numerical features using Unigrams, Bigrams, and Trigrams with Sublinear TF scaling.

# 3.Classification:
-Algorithm: Linear Support Vector Machine (Linear SVM).
-Optimization: Hyperparameters tuned using GridSearchCV with 5-fold Stratified Cross-Validation.
-Metric Optimization: Model was optimized to maximize the Macro F1-Score.

# 📝 Future Improvements
Integration with live Amazon Product APIs.
Multi-language support (e.g., Hindi, Spanish).
Aspect-based sentiment analysis (analyzing specific features like "Battery" or "Screen").


📄 License
This project is open source and available for educational purposes.
