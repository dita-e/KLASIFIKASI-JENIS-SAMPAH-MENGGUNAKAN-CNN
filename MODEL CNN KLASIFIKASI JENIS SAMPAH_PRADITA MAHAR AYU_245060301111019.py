import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

DATASET_DIR = r"C:\Users\dita\Documents\KULIAH\dataset_uas" 
IMG_SIZE = (150, 150)
BATCH_SIZE = 16  

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,      
    rotation_range=30,      
    width_shift_range=0.2,      
    height_shift_range=0.2,     
    shear_range=0.2,          
    zoom_range=0.2,             
    horizontal_flip=True,       
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

test_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20
)

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
plt.savefig('confusion_matrix.png') 
plt.show()

model.save('model_uas_cnn.h5')
print("Model berhasil disimpan dengan nama 'model_uas_cnn.h5'")