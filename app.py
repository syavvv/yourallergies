import os
from flask import Flask, request, render_template, redirect
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import joblib
import base64
from io import BytesIO

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi direktori unggahan
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Path ke Tesseract OCR di sistem Anda
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load model dan vectorizer (sesuaikan path file jika perlu)
model = joblib.load('model_xgboost.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    # Konversi ke grayscale
    img = image.convert('L')
    # Tingkatkan kontras
    img = ImageEnhance.Contrast(img).enhance(2.0)
    # Terapkan thresholding sederhana
    img = img.filter(ImageFilter.MedianFilter())
    return img

# Fungsi untuk mengekstrak teks menggunakan OCR
def process_image(image):
    if isinstance(image, str):  # Jika path file
        img = Image.open(image)
    else:  # Jika objek BytesIO (dari kamera)
        img = Image.open(image)
    # Preprocessing gambar sebelum OCR
    img = preprocess_image(img)
    extracted_text = pytesseract.image_to_string(img, config='--psm 6')
    return extracted_text

# Fungsi untuk klasifikasi alergen dengan rekomendasi
def classify_allergens(text):
    detected_allergens = []
    recommendations = []

    # Alergen umum dan rekomendasinya
    allergen_recommendations = {
        "milk": {"name": "Susu", "alternative": "Susu kedelai, susu almond", "solution": "Konsumsi antihistamin, konsultasi dokter jika parah"},
        "soy": {"name": "Kedelai", "alternative": "Susu oat, santan kelapa", "solution": "Hindari makanan berbahan kedelai, konsumsi antihistamin"},
        "wheat": {"name": "Gandum", "alternative": "Tepung jagung, tepung almond", "solution": "Hindari produk gandum, konsultasi ahli gizi"},
        "tree nut": {"name": "Kacang Pohon", "alternative": "Biji bunga matahari, biji labu", "solution": "Gunakan epinefrin jika terjadi reaksi anafilaksis"},
        "peanut": {"name": "Kacang", "alternative": "Biji chia, biji rami", "solution": "Gunakan epinefrin atau antihistamin sesuai kebutuhan"},
        "egg": {"name": "Telur", "alternative": "Pengganti telur komersial, saus apel", "solution": "Hindari produk berbahan telur, konsumsi antihistamin"},
        "fish": {"name": "Ikan", "alternative": "Sumber protein nabati seperti tempe", "solution": "Konsumsi antihistamin, konsultasi dokter jika perlu"},
        "shellfish": {"name": "Kerang", "alternative": "Protein nabati lainnya", "solution": "Gunakan epinefrin jika reaksi berat"},
        "sesame": {"name": "Wijen", "alternative": "Biji bunga matahari, minyak kelapa", "solution": "Hindari produk berbahan wijen, konsumsi antihistamin"},
        "sulfite": {"name": "Sulfit", "alternative": "Produk segar tanpa pengawet", "solution": "Hindari makanan kalengan atau anggur tertentu"},
        "gluten": {"name": "Gluten", "alternative": "Produk bebas gluten", "solution": "Ikuti diet bebas gluten"},
        "almond": {"name": "Almond", "alternative": "Biji bunga matahari, kacang mete", "solution": "Hindari produk berbahan almond, konsultasi dokter jika parah"}
    }

    # Deteksi alergen dari teks
    for allergen, info in allergen_recommendations.items():
        if allergen in text.lower():
            detected_allergens.append((info["name"], info["alternative"], info["solution"]))

    # Deteksi kode E
    e_code_mapping = {
    "E120": "Cochineal extract (pewarna dari serangga)",
    "E471": "Mono- dan Digliserida Asam Lemak (emulsifier)",
    "E441": "Gelatin (mungkin berasal dari babi)",
    "E322": "Lesitin (biasanya dari kedelai)",
    "E631": "Disodium Inosinat (penyedap rasa, dari daging atau ikan)",
    "E904": "Shellac (agen pelapis, dari serangga)",
    "E150d": "Caramel IV (pewarna dari gula)",
    "E160a": "Karotenoid (pewarna alami atau sintetis)",
    "E252": "Kalium Nitrat (pengawet, digunakan pada daging olahan)",
    "E100": "Kurkumin (pewarna alami dari kunyit)",
    "E129": "Allura Red AC (pewarna sintetik merah)",
    "E133": "Brilliant Blue FCF (pewarna sintetik biru)",
    "E110": "Sunset Yellow FCF (pewarna sintetik kuning)",
    "E491": "Sorbitan Monostearate (emulsifier, digunakan pada makanan panggang)"
}

    # Deteksi kode E dalam teks
    e_codes = []
    normalized_text = text.lower().replace(" ", "").replace("\n", "")
    for code, description in e_code_mapping.items():
        if code.lower() in normalized_text:
            e_codes.append((code, description))

    return detected_allergens, e_codes

# Rute utama
@app.route('/')
def home():
    return render_template('index.html')

# Rute untuk kontak dan media sosial
@app.route('/twitter')
def twitter():
    return redirect("https://twitter.com/yourprofile")

@app.route('/facebook')
def facebook():
    return redirect("https://facebook.com/yourprofile")

@app.route('/instagram')
def instagram():
    return redirect("https://instagram.com/yourprofile")

@app.route('/linkedin')
def linkedin():
    return redirect("https://linkedin.com/in/yourprofile")

@app.route('/contact')
def contact():
    return render_template('contact.html')

# Rute untuk deteksi alergen
@app.route('/detect', methods=['GET', 'POST'])
def detect():
    result = None
    if request.method == 'POST':
        if 'file' in request.files:
            # Simpan file yang diunggah
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
                uploaded_file.save(image_path)
                extracted_text = process_image(image_path)
                print(f"Hasil OCR dari file: {extracted_text}")  # Debugging
                allergens, e_codes = classify_allergens(extracted_text)
                result = (allergens, e_codes)

        elif 'camera_image' in request.form:
            # Proses gambar dari kamera
            camera_data = request.form['camera_image']
            if ',' in camera_data:
                camera_data = camera_data.split(',')[1]
            image = BytesIO(base64.b64decode(camera_data))
            extracted_text = process_image(image)
            print(f"Hasil OCR dari kamera: {extracted_text}")  # Debugging
            allergens, e_codes = classify_allergens(extracted_text)
            result = (allergens, e_codes)

    return render_template('detect.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
