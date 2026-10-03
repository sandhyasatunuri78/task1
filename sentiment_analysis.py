import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------
# 1. Sample Dataset
# -----------------------------------

data = {
    "text": [
        "I love this product",
        "This is an amazing product",
        "The food was excellent",
        "I am very happy with the service",
        "This is wonderful",
        "I really enjoyed this movie",
        "The product is very good",
        "Amazing experience",

        "I hate this product",
        "This is terrible",
        "The food was awful",
        "I am very disappointed",
        "Worst experience ever",
        "The service was bad",
        "I don't like this product",
        "This movie was boring",

        "The product is okay",
        "The service was average",
        "It is neither good nor bad",
        "The food was normal",
        "The experience was okay",
        "Nothing special about this product",
        "It was fine",
        "The movie was average"
    ],

    "sentiment": [
        "positive", "positive", "positive", "positive",
        "positive", "positive", "positive", "positive",

        "negative", "negative", "negative", "negative",
        "negative", "negative", "negative", "negative",

        "neutral", "neutral", "neutral", "neutral",
        "neutral", "neutral", "neutral", "neutral"
    ]
}


# Convert dataset into DataFrame
df = pd.DataFrame(data)


# -----------------------------------
# 2. Split Dataset
# -----------------------------------

X = df["text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# -----------------------------------
# 3. Convert Text into TF-IDF
# -----------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------------
# 4. Train Machine Learning Model
# -----------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)


# -----------------------------------
# 5. Evaluate Model
# -----------------------------------

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("   SENTIMENT ANALYSIS MODEL")
print("==============================")

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------------
# 6. Predict New Text
# -----------------------------------

def predict_sentiment(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]

    return prediction


# -----------------------------------
# 7. User Input
# -----------------------------------

print("\n------------------------------")
print("Enter a sentence to analyze.")
print("Type 'exit' to stop.")
print("------------------------------")

while True:

    user_text = input("\nEnter text: ")

    if user_text.lower() == "exit":
        print("Thank you!")
        break

    result = predict_sentiment(user_text)

    print("Sentiment:", result.upper())