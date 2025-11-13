import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = pd.read_csv('IMDB Dataset.csv')
data['sentiment'] = data['sentiment'].replace({'positive': 1, 'negative': 0})
data = data.sample(10000, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

while True:
    text = input("\nEnter a sentence (or 'quit' to exit): ")
    if text.lower() == 'quit':
        break
    text_tfidf = vectorizer.transform([text])
    pred = model.predict(text_tfidf)[0]
    print("Sentiment:", "positive" if pred == 1 else "negative")
