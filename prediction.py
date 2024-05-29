# prediction.py

import joblib
from cleaning import clean_text


def load_model_and_predict(test_texts, model_file, vectorizer_file, label_encoder_file):

    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    label_encoder = joblib.load(label_encoder_file)

    test_texts_cleaned = [clean_text(text) for text in test_texts]
    X_new = vectorizer.transform(test_texts_cleaned)

    predictions = model.predict(X_new)
    predicted_categories = label_encoder.inverse_transform(predictions)

    return predicted_categories
