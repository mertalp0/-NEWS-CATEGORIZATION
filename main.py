import os
from data_loading import load_and_clean_data
from training import train_and_save_model
from prediction import load_model_and_predict

# File paths
model_file = 'models/naive_bayes_model.pkl'
vectorizer_file = 'models/tfidf_vectorizer.pkl'
label_encoder_file = 'models/label_encoder.pkl'
data_file = 'dataset/new_dataset.csv'

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# If model files exist, load and predict; otherwise, train and save model
if os.path.exists(model_file) and os.path.exists(vectorizer_file) and os.path.exists(label_encoder_file):
    print("Loading model and other components...")
else:
    df = load_and_clean_data(data_file)
    train_and_save_model(df, model_file, vectorizer_file, label_encoder_file)
    print("Model and other components trained and saved.")

# Sample test data
test_news = [
    "Ekonomi son dönemdeki politika değişiklikleri nedeniyle zorluklarla karşı karşıya.",
    "Yeni bir teknoloji startup'ı yenilikçi çözümleriyle endüstriyi sarsıyor.",
    "Yerel spor takımı bu yıl da şampiyonluğu kazandı.",
    "Çözümlenmemiş referans 'MultinomialNB",
    "kredi faizlerinde indirim yapıldı",
    "hakan çalhanoğlu 4 asist yaptı",
    "recep tayyip erdoğan samsunda",
]

# Make predictions
predicted_categories = load_model_and_predict(test_news, model_file, vectorizer_file, label_encoder_file)
print("Predicted Categories:", predicted_categories)
