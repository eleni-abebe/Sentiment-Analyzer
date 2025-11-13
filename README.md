## 🧠 Sentiment Analyzer (Text-Based)

### 📌 Overview

This project is a **Sentiment Analyzer** that predicts whether a given text expresses a **positive** or **negative** sentiment.
It uses **machine learning** techniques from **Scikit-learn** and is trained on the **IMDb Movie Reviews Dataset** from Kaggle.

---

### 🚀 Features

* Classifies text as **positive** or **negative**
* Uses **TF-IDF vectorization** for text representation
* **Logistic Regression** model for classification
* Works interactively in the terminal
* Trains quickly using a lightweight dataset
* Saves trained model and vectorizer for reuse

---

### 📁 Project Structure

```
SentimentAnalyzer/
│
├── venv/                     # Virtual environment
├── IMDB Dataset.csv          # Dataset (from Kaggle)
├── sentiment_analyzer.py     # Main program
├── sentiment_model.pkl       # Saved trained model
├── tfidf_vectorizer.pkl      # Saved TF-IDF vectorizer
└── README.md                 # Project documentation
```

---

### ⚙️ Installation & Setup

#### 1️⃣ Clone or create the project folder

```bash
mkdir SentimentAnalyzer
cd SentimentAnalyzer
```

#### 2️⃣ Set up a virtual environment

```bash
python -m venv venv
```

Activate it:

* **Windows:** `venv\Scripts\activate`
* **Mac/Linux:** `source venv/bin/activate`

#### 3️⃣ Install required libraries

```bash
pip install pandas numpy scikit-learn joblib
```

#### 4️⃣ Download dataset

Go to [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
Download `IMDB Dataset.csv` and place it inside your project folder.

---

### 🧩 Usage

Run the script:

```bash
python sentiment_analyzer.py
```

It will:

* Train the model on a sample of IMDb reviews
* Display accuracy and performance metrics
* Save the trained model and TF-IDF vectorizer
* Let you input sentences to classify sentiment

Example session:

```
Accuracy: 0.89
              precision    recall  f1-score   support
           0       0.88      0.89      0.88      1010
           1       0.89      0.89      0.89       990

Enter a sentence (or 'quit' to exit): I love this movie
Sentiment: positive
Enter a sentence (or 'quit' to exit): This was terrible
Sentiment: negative
```

---

### 💾 Reuse Saved Model

After the first run, the trained model and vectorizer are saved.
You can load them later without retraining:

python
import joblib
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

text = "This movie was awesome!"
text_tfidf = vectorizer.transform([text])
prediction = model.predict(text_tfidf)[0]
print("Sentiment:", "positive" if prediction == 1 else "negative")


---

### 📚 Technologies Used

* **Python 3**
* **Pandas** – data loading and manipulation
* **Scikit-learn** – TF-IDF vectorization and Logistic Regression
* **Joblib** – saving trained models

---

### 🧾 Dataset Info

* **Source:** [Kaggle - IMDb Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Size:** ~6 MB
* **Samples:** 50,000 movie reviews (balanced positive/negative)

---

### 💡 Future Improvements

* Add **Streamlit web interface**
* Include **neutral** sentiment category
* Expand preprocessing (remove emojis, punctuation, etc.)
* Try deep learning models (LSTM, BERT)

