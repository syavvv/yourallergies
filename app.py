import os
from flask import Flask, request, render_template, redirect
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import joblib
import base64
from io import BytesIO  

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # Maks upload 25MB

# Load variabel lingkungan
port = int(os.environ.get("PORT", 5000))  
secret_key = os.environ.get("SECRET_KEY", "punyapinaa")

# Untuk Linux (Railway), tidak perlu set path manual ke tesseract
# pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_PATH", r'C:\Program Files\Tesseract-OCR\tesseract.exe')

app.secret_key = secret_key

# Konfigurasi direktori unggahan
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model dan vectorizer
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "model_xgboost.pkl")
vectorizer_path = os.path.join(base_dir, "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Fungsi preprocessing gambar
def preprocess_image(image):
    img = image.convert('L')  # Grayscale
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.MedianFilter())
    return img

# Fungsi ekstraksi teks dari gambar
def process_image(image):
    img = Image.open(image) if isinstance(image, str) else Image.open(image)
    img = preprocess_image(img)
    return pytesseract.image_to_string(img, config='--psm 6')

# Fungsi klasifikasi alergen
def classify_allergens(text):
    detected_allergens = []
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

    for allergen, info in allergen_recommendations.items():
        if allergen in text.lower():
            detected_allergens.append((info["name"], info["alternative"], info["solution"]))

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
        "E491": "Sorbitan Monostearate (emulsifier, digunakan pada makanan panggang)",
        "E319": "TBHQ (antioksidan sintetis)",
        "E621": "Monosodium Glutamate (penyedap rasa)",
        "E307": "Alpha Tocopherol (vitamin E, antioksidan)"
    }

    e_codes = []
    normalized_text = text.lower().replace(" ", "").replace("\n", "")
    for code, description in e_code_mapping.items():
        if code.lower() in normalized_text:
            e_codes.append((code, description))

    return detected_allergens, e_codes

# Routing
@app.route('/')
def home():
    return render_template('index.html')

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

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    result = None
    if request.method == 'POST':
        if 'file' in request.files:
            uploaded_file = request.files['file']
            print("File diterima dari kamera:", uploaded_file.filename)

            if uploaded_file.filename != '':
                image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
                uploaded_file.save(image_path)
                print("Disimpan di:", image_path)

                extracted_text = process_image(image_path)
                allergens, e_codes = classify_allergens(extracted_text)
                result = (allergens, e_codes)
        else:
            print("Tidak ada file dalam request")

    return render_template('detect.html', result=result)

# Menjalankan Flask (khusus lokal)
if __name__ == '__main__':
    app.run()