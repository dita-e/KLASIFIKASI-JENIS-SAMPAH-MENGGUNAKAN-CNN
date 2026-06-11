import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. PATH DATASET (Sesuaikan dengan folder Anda)
DATASET_DIR = r"C:\Users\dita\Documents\KULIAH\dataset_uas" 
IMG_SIZE = (150, 150)
BATCH_SIZE = 16  # Angka batch kecil (16) sangat bagus untuk dataset berukuran 300 gambar

# 2. DATA AUGMENTATION (Kunci sukses untuk dataset mini agar akurasi tinggi)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,       # Otomatis memotong 20% untuk data testing
    rotation_range=30,          # Memutar gambar acak hingga 30 derajat
    width_shift_range=0.2,      # Menggeser gambar secara horizontal
    height_shift_range=0.2,     # Menggeser gambar secara vertikal
    shear_range=0.2,            # Meregangkan gambar
    zoom_range=0.2,             # Memperbesar/memperkecil gambar secara acak
    horizontal_flip=True,       # Membalik gambar secara horizontal
    fill_mode='nearest'
)

# Load Data Training (Akan muncul: Found 240 images belonging to 3 classes)
train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Load Data Testing (Akan muncul: Found 60 images belonging to 3 classes)
test_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 3. ARSITEKTUR CNN SEDERHANA (Efektif untuk data kecil)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5), # Mencegah overfitting tambahan
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. TRAINING (Gunakan 15-20 Epoch karena datanya sedikit)
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20
)

# =====================================================================
# 5. OUTPUT EVALUASI (Sesuai Ketentuan No. 4 & 5 di Soal)
# =====================================================================

# Generate Confusion Matrix
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Greens')
plt.xlabel('Prediksi Model')
plt.ylabel('Data Asli')
plt.title('Confusion Matrix - UAS AI Kelompok C')
plt.savefig('confusion_matrix.png') # File ini yang nanti diupload ke Github
plt.show()

# Simpan Model untuk bahan Demo Video (.h5)
model.save('model_uas_cnn.h5')
print("Model berhasil disimpan dengan nama 'model_uas_cnn.h5'")