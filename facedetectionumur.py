import cv2
import numpy as np
from keras.models import load_model
import os
from PIL import Image, ImageDraw, ImageFont

# Inisialisasi detektor wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path ke model
age_model_path = '/home/fs/Dokumen//machine-learning/AgePredictionModel.h5'
gender_model_path = '/home/fs/Dokumen/machine-learning/gender.keras'
expression_model_path = '/home/fs/Dokumen/machine-learning/ekspresi.h5'

# Cek apakah model ada
for model_path in [age_model_path, gender_model_path, expression_model_path]:
    if not os.path.exists(model_path):
        print(f"Model file tidak ditemukan di {model_path}")
        exit()

# Muat model prediksi
try:
    age_model = load_model(age_model_path)
    gender_model = load_model(gender_model_path)
    expression_model = load_model(expression_model_path)
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# Fungsi untuk memperkirakan umur
def predict_age(face_img):
    face_img = cv2.resize(face_img, (200, 200))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    age_pred = age_model.predict(face_img)
    predicted_class = np.argmax(age_pred)
    age_mapping = {i: 20 + i for i in range(31)}
    return age_mapping.get(predicted_class, "Tidak Diketahui")

# Fungsi untuk memperkirakan gender
def predict_gender(face_img):
    face_img = cv2.resize(face_img, (200, 200))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    gender_pred = gender_model.predict(face_img)
    predicted_class = np.argmax(gender_pred)
    return 'Laki-laki' if predicted_class == 0 else 'Perempuan'

# Fungsi untuk memperkirakan ekspresi
def predict_expression(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=-1)
    face_img = np.expand_dims(face_img, axis=0)
    
    expression_pred = expression_model.predict(face_img)
    predicted_class = np.argmax(expression_pred)

    # Logging output probabilitas
    print(f"Probabilitas ekspresi: {expression_pred}")

    expression_mapping = {
        0: 'Marah',
        1: 'Jijik',
        2: 'Takut',
        3: 'Senang',
        4: 'Biasa',
        5: 'Sedih',
        6: 'Terkejut'
    }
    
    return expression_mapping.get(predicted_class, "Tidak Diketahui")

# Fungsi untuk menambahkan teks dengan border menggunakan Pillow
def put_text_with_border(frame, text, position, font, border_color, text_color):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    
    # Gambar border teks
    x, y = position
    # Offset untuk border
    offset = 1
    draw.text((x - offset, y - offset), text, font=font, fill=border_color)
    draw.text((x + offset, y - offset), text, font=font, fill=border_color)
    draw.text((x - offset, y + offset), text, font=font, fill=border_color)
    draw.text((x + offset, y + offset), text, font=font, fill=border_color)
    
    # Gambar teks utama
    draw.text(position, text, font=font, fill=text_color)
    
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Mulai capture video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Tidak dapat mengakses kamera. Pastikan kamera terhubung dan tidak digunakan oleh aplikasi lain.")
    exit()

# Ukuran font dinamis
font_size = 17
font = ImageFont.truetype('/home/fs/Dokumen/font/AnonymousPro-Regular.ttf', font_size)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari kamera. Keluar...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        try:
            age = predict_age(face_img)
            gender = predict_gender(face_img)
            expression = predict_expression(face_img)

            # Menggambar persegi di wajah
            cv2.rectangle(frame, (x, y), (x+w, y+h), (36, 255, 12), 2)

            # Menentukan posisi untuk menampilkan umur, gender, dan ekspresi
            text_offset = 10  # Jarak dari batas atas wajah
            border_color = (0, 0, 0)  # Warna border (hitam)
            text_color = (36, 255, 12)  # Warna teks (hijau)

            frame = put_text_with_border(frame, f'Umur: {age}', (x, y - 50 - text_offset), font, border_color, text_color)
            frame = put_text_with_border(frame, f'Gender: {gender}', (x, y - 35 - text_offset), font, border_color, text_color)
            frame = put_text_with_border(frame, f'Ekspresi: {expression}', (x, y - 20 - text_offset), font, border_color, text_color)
        except Exception as e:
            print(f"Error saat memprediksi: {e}")

    # Menampilkan hasil deteksi wajah dan estimasi umur, gender, serta ekspresi
    cv2.imshow('Face Detection, Age, Gender, and Expression Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
