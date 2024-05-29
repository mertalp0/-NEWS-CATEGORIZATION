import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def train_and_save_model(df, model_file, vectorizer_file, label_encoder_file):
    """
    Trains and saves the model.
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['CLEANED_TEXT'])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['category'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)
    joblib.dump(label_encoder, label_encoder_file)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", report)
