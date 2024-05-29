import requests
from bs4 import BeautifulSoup
import os
from data_loading import load_and_clean_data
from training import train_and_save_model
from prediction import load_model_and_predict

def fetch_latest_news():
    """
    Haberler.com sitesinden en son haber başlıklarını çeker.
    """
    url = "https://www.haberler.com"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p", class_="hbBoxText")
    return [paragraph.get_text(strip=True) for paragraph in paragraphs]

# Dosya yolları
model_file = 'models/naive_bayes_model.pkl'
vectorizer_file = 'models/tfidf_vectorizer.pkl'
label_encoder_file = 'models/label_encoder.pkl'
data_file = 'dataset/new_dataset.csv'

# Modellerin kaydedileceği klasör mevcut değilse oluşturulur
if not os.path.exists('models'):
    os.makedirs('models')

# Eğer model dosyaları mevcutsa, model ve diğer bileşenler yüklenir; aksi halde model eğitilir ve kaydedilir
if os.path.exists(model_file) and os.path.exists(vectorizer_file) and os.path.exists(label_encoder_file):
    print("Model ve diğer bileşenler yükleniyor...")
else:
    df = load_and_clean_data(data_file)
    train_and_save_model(df, model_file, vectorizer_file, label_encoder_file)
    print("Model ve diğer bileşenler eğitildi ve kaydedildi.")

# Haberler.com'dan en son haberleri çekme
test_news = fetch_latest_news()

# Çekilen haber başlıklarını yazdırma
print("\nÇekilen Haber Başlıkları:")
for i, news in enumerate(test_news, 1):
    print(f"{i}. {news}")

# Tahmin yapma
predicted_categories = load_model_and_predict(test_news, model_file, vectorizer_file, label_encoder_file)

# Tahmin edilen kategorileri yazdırma
print("\nTahmin Edilen Kategoriler:")
for i, category in enumerate(predicted_categories, 1):
    print(f"{i}. {category}")

# Haber başlıkları ile tahmin edilen kategorileri eşleştirip yazdırma
print("\nHaber Başlıkları ve Tahmin Edilen Kategoriler:")
for i, (news, category) in enumerate(zip(test_news, predicted_categories), 1):
    print(f"{i}. Haber: {news} - Kategori: {category}")
