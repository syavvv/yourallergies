import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Fungsi utama untuk pelatihan model
def train_and_save_model(csv_path, model_output='model_xgboost.pkl', vectorizer_output='tfidf_vectorizer.pkl'):
    # 1. Load dataset
    data = pd.read_csv(csv_path)
    
    if 'text' not in data.columns or 'label' not in data.columns:
        raise ValueError("Dataset harus memiliki kolom 'text' dan 'label'")
    
    X = data['text']
    y = data['label']

    # 2. TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # 4. Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # 5. Evaluation (optional, for terminal feedback)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Akurasi Model: {acc*100:.2f}%")
    print("[INFO] Classification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save model and vectorizer
    joblib.dump(model, model_output)
    joblib.dump(vectorizer, vectorizer_output)
    print(f"[INFO] Model disimpan ke: {model_output}")
    print(f"[INFO] Vectorizer disimpan ke: {vectorizer_output}")

# Jalankan hanya jika file ini dieksekusi langsung
if __name__ == '__main__':
    csv_path = 'combined_dataset.csv'  # Ubah path jika file berada di lokasi lain
    train_and_save_model(csv_path)
