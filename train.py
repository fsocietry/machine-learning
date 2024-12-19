import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Perbarui impor

# Parameter
batch_size = 32
img_height, img_width = 200, 200
epochs = 10

# Path ke dataset
train_data_dir = '/home/fs/Unduhan/20-50/train'  # Ubah ke path dataset kamu
validation_data_dir = '/home/fs/Unduhan/20-50/test'  # Ubah ke path dataset validasi

# Image Data Generator untuk preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Membaca data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse'  # Menggunakan sparse karena label adalah integer
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse'
)

# Membangun model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(31, activation='softmax')  # Sesuaikan dengan jumlah kelas
])

# Mengkompilasi model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Melatih model
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Menyimpan model
model.save('AgePredictionModel.h5')

