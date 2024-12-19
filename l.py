import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path ke folder dataset
train_data_path = '/home/fs/Dokumen/MMAFEDB/train/'  # Ganti dengan path folder train kamu
validation_data_path = '/home/fs/Dokumen/MMAFEDB/valid/'  # Ganti dengan path folder validasi kamu

# Pra-pemrosesan data dengan ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalisasi untuk data pelatihan
validation_datagen = ImageDataGenerator(rescale=1./255)  # Normalisasi untuk data validasi

# Menghasilkan data pelatihan
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(48, 48),  # Ukuran gambar
    color_mode='grayscale', # Gambar grayscale
    class_mode='categorical', # Menggunakan one-hot encoding untuk kelas
    batch_size=64
)

# Menghasilkan data validasi
validation_generator = validation_datagen.flow_from_directory(
    validation_data_path,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=64
)

# Membangun model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 kelas untuk ekspresi

# Kompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
model.fit(train_generator, epochs=30, validation_data=validation_generator)

# Menyimpan model
model.save('ekspresi.h5')

print("Model ekspresi telah dilatih dan disimpan.")
