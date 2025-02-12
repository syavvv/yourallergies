import joblib
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Memuat model yang telah dilatih
model = joblib.load('model_xgboost.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Memuat dataset uji
# Pastikan Anda memiliki file dataset uji yang sesuai
data_test = pd.read_csv('combined_dataset.csv')

# Menyiapkan fitur dan label uji
X_test = data_test['text']  # Ganti dengan kolom teks pada dataset uji Anda
y_test = data_test['label']  # Ganti dengan kolom label pada dataset uji Anda

# Transformasi data uji menggunakan vectorizer
X_test_tfidf = vectorizer.transform(X_test)

# Melakukan prediksi menggunakan model
y_pred = model.predict(X_test_tfidf)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)

# Menampilkan hasil akurasi
print(f"Akurasi model: {accuracy * 100:.2f}%")
