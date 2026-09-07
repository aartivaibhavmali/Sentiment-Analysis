# 🛒 Amazon Review Sentiment Analysis AI

A Machine Learning web application designed to analyze customer reviews from Amazon and classify them into three sentiment categories:

* 🟢 **Positive**
* 🔴 **Negative**
* 🟡 **Neutral**

The project uses **Natural Language Processing (NLP), TF-IDF Vectorization, and a tuned Linear Support Vector Machine (Linear SVM)** for sentiment classification.

---

## 🌟 Key Features

* **3-Class Sentiment Classification**
  Predicts whether an Amazon review is Positive, Negative, or Neutral.

* **Real-Time Analysis**
  Analyze customer reviews instantly by entering or pasting text.

* **Confidence Visualization**
  Displays a confidence score for the predicted sentiment using an interactive progress bar.

* **Text Preprocessing**
  Automatically cleans HTML tags, URLs, special characters, and unnecessary whitespace.

* **Model Insights**
  Displays model evaluation metrics such as:

  * Accuracy
  * Precision
  * Recall
  * F1-Score

* **Interactive Web Interface**
  Built using Streamlit for easy and user-friendly interaction.

---

## 🛠️ Tech Stack

| Technology          | Purpose                         |
| ------------------- | ------------------------------- |
| **Python 3.8+**     | Programming language            |
| **Streamlit**       | Web application framework       |
| **Scikit-learn**    | Machine Learning and NLP        |
| **LinearSVC**       | Sentiment classification        |
| **TfidfVectorizer** | Text feature extraction         |
| **Pandas**          | Data manipulation               |
| **NumPy**           | Numerical operations            |
| **Joblib**          | Model serialization             |
| **Regex (re)**      | Text cleaning and preprocessing |

---

## 📂 Project Structure

```text
Amazon-Review-Sentiment-Analysis/
│
├── app.py
│   └── Main Streamlit application
│
├── sentiment_model.pkl
│   └── Serialized tuned Linear SVM model
│
├── sentiment_metrics.pkl
│   └── Saved model evaluation metrics
│
├── requirements.txt
│   └── Python dependencies
│
├── P652-Dataset.xlsx
│   └── Raw Amazon review dataset
│
├── Processed_dataset.csv
│   └── Cleaned and processed dataset
│
└── README.md
    └── Project documentation
```

---

## 🚀 Live Deployment

The application is deployed using **Streamlit**.

🔗 **Live Application:**
https://sentiment-analysis-review-checker.streamlit.app/

---

## 💻 How to Use

### 1. Enter Review

Enter or paste an Amazon product review into the text area.

### 2. Analyze Sentiment

Click the:

**🔍 Analyze Sentiment**

button.

### 3. View Results

The application displays:

* Predicted sentiment
* Confidence score
* Probability/confidence visualization
* Review statistics

The review statistics include:

* Word count
* Character count

---

# 🤖 Model Methodology

## 1. Text Preprocessing

The raw review text is cleaned before converting it into machine learning features.

The preprocessing steps include:

* Conversion of text to lowercase
* Removal of HTML tags
* Removal of URLs
* Removal of special characters
* Removal of unnecessary/non-alphanumeric characters
* Whitespace normalization

### Example

```text
Original:
"This product is <b>AMAZING!</b> Visit https://example.com"

After preprocessing:
"this product is amazing visit"
```

---

## 2. Feature Engineering

### TF-IDF Vectorization

**TF-IDF (Term Frequency–Inverse Document Frequency)** is used to convert textual reviews into numerical feature vectors that can be processed by the machine learning model.

The vectorizer uses:

* **Unigrams**
* **Bigrams**
* **Trigrams**
* **Sublinear TF Scaling**

### N-Grams

The model considers combinations of words rather than only individual words.

For example:

```text
Unigram:
"good"

Bigram:
"very good"

Trigram:
"very good product"
```

This helps the model capture more context from customer reviews.

---

## 3. Classification

### Algorithm: Linear Support Vector Machine

A **Linear Support Vector Machine (Linear SVM)** is used as the classification algorithm.

The model learns decision boundaries that separate the three sentiment classes:

```text
                 Review Text
                      ↓
              Text Preprocessing
                      ↓
                TF-IDF Vectorizer
                      ↓
                Linear SVM Model
                      ↓
          ┌───────────┼───────────┐
          ↓           ↓           ↓
       Positive     Neutral     Negative
```

---

## 4. Hyperparameter Tuning

The Linear SVM model was optimized using:

**GridSearchCV**

with:

* **5-Fold Stratified Cross-Validation**
* **Macro F1-Score** as the optimization metric

### Why Macro F1-Score?

Macro F1 calculates the F1-score independently for each class and then takes their average.

This gives equal importance to:

* Positive
* Negative
* Neutral

It is useful for multi-class classification, particularly when class distributions are not perfectly balanced.

---

# 📊 Model Evaluation

The model performance is evaluated using the following metrics:

### Accuracy

Measures the percentage of correctly classified reviews.

### Precision

Measures how many reviews predicted as a particular sentiment actually belong to that sentiment.

### Recall

Measures how many actual reviews of a sentiment class were correctly identified.

### F1-Score

The harmonic mean of Precision and Recall.

The application stores these evaluation metrics in:

```text
sentiment_metrics.pkl
```

---

# 🔄 End-to-End Workflow

```text
Raw Amazon Reviews
        ↓
Data Cleaning
        ↓
Text Preprocessing
        ↓
Exploratory Data Analysis
        ↓
TF-IDF Feature Extraction
        ↓
Train-Test Split
        ↓
Linear SVM
        ↓
GridSearchCV
        ↓
5-Fold Stratified Cross-Validation
        ↓
Macro F1 Optimization
        ↓
Final Model
        ↓
Model Serialization
        ↓
Streamlit Deployment
        ↓
Real-Time Sentiment Prediction
```

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/aartivaibhavmali/Amazon-Review-Sentiment-Analysis.git
```

## 2. Navigate to the Project Directory

```bash
cd Amazon-Review-Sentiment-Analysis
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

# 📈 Example Prediction

### Input

```text
"The product quality is excellent and I am very happy with my purchase."
```

### Output

```text
Sentiment: Positive
```

The application also displays the corresponding confidence visualization.

---

# 🔮 Future Improvements

The project can be further improved by adding:

* 🔗 Integration with live Amazon Product APIs
* 🌎 Multi-language sentiment analysis
* 🇮🇳 Hindi and other regional language support
* 🔍 Aspect-Based Sentiment Analysis
* 📊 Advanced sentiment dashboards
* 🤖 Transformer-based models such as BERT
* 📱 Improved mobile-friendly UI

### Aspect-Based Sentiment Analysis

Instead of predicting only the overall sentiment, the system could identify sentiment related to specific product features.

For example:

```text
Review:
"The battery is excellent but the camera quality is poor."

Battery → Positive
Camera → Negative
```

---

# 📚 Skills Demonstrated

This project demonstrates practical experience in:

* Python
* Natural Language Processing (NLP)
* Text preprocessing
* Feature engineering
* TF-IDF
* N-Gram analysis
* Machine Learning
* Linear SVM
* Multi-class classification
* GridSearchCV
* Cross-validation
* Model evaluation
* Model serialization
* Streamlit
* Machine Learning deployment

---

# ⚠️ Disclaimer

This project is created for **educational and demonstration purposes**. The sentiment predictions are generated by a machine learning model and should not be considered a definitive assessment of customer opinions.

---

# 📄 License

This project is open source and available for **educational purposes**.

---

## 👩‍💻 Author

**Aarti Vaibhav Mali**

Data Science | Machine Learning | Python | SQL | Power BI

GitHub:
https://github.com/aartivaibhavmali
