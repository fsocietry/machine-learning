import cv2
import numpy as np
from keras.models import load_model

# Path ke model ekspresi
expression_model_path = '/home/fs/Dokumen/ekspresi.h5'

# Muat model ekspresi
try:
    expression_model = load_model(expression_model_path)
    print("Model ekspresi berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model ekspresi: {e}")
    exit()

# Fungsi untuk memperkirakan ekspresi
def predict_expression(face_img):
    # Ubah gambar menjadi grayscale
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))  # Ukuran input model ekspresi
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=-1)  # Tambahkan dimensi channel
    face_img = np.expand_dims(face_img, axis=0)  # Tambahkan dimensi batch

    expression_pred = expression_model.predict(face_img)
    predicted_class = np.argmax(expression_pred)

    expression_mapping = {0: 'Marah', 1: 'Senang', 2: 'Sedih', 3: 'Takut', 4: 'Terkejut', 5: 'Biasa', 6: 'Jijik'}
    return expression_mapping.get(predicted_class, "Tidak Diketahui")

# Uji dengan gambar statis
test_img_path = '/home/fs/Dokumen/images/train/angry/0.jpg'  # Path gambar yang ingin diuji
test_img = cv2.imread(test_img_path)

if test_img is not None:
    expression = predict_expression(test_img)
    print(f"Ekspresi yang diprediksi: {expression}")
else:
    print("Gambar tidak ditemukan atau tidak bisa dibaca.")
