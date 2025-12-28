import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters

IMG_SIZE = 96
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS_STAGE1 = 20   # Initial training
EPOCHS_STAGE2 = 15   # Fine-tuning 1
EPOCHS_STAGE3 = 20   # Fine-tuning 2
EPOCHS_STAGE4 = 15   # Final fine-tuning

TRAIN_DIR = r"F:\NadaOnTop\3\New folder\archive (1)\split_dataset\train"
VAL_DIR   = r"F:\NadaOnTop\3\New folder\archive (1)\split_dataset\val"

# Data Generators (Strong Augmentation)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    shear_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Class Indices:", train_data.class_indices)

# Compute Class Weights

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# Build Base Model (Transfer Learning)

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Stage 1: Initial Training

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nStage 1: Initial Training")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE1
)

# Stage 2: Fine-tuning (Unfreeze last 50 layers)

for layer in model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nStage 2: Fine-tuning (Last 50 layers)")
history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE2
)

# Stage 3: Fine-tuning (Unfreeze last 100 layers + class weights)

for layer in model.layers[-100:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nStage 3: Fine-tuning (Last 100 layers + Class Weights)")
history3 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE3,
    class_weight=class_weights
)

# Stage 4: Final Fine-tuning

print("\nStage 4: Final Fine-tuning")
history4 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE4,
    class_weight=class_weights
)

# Save Final Model

model.save("emotion_model_final.h5")
print("\nFinal model saved as emotion_model_final.h5")

# Plot Final Accuracy & Loss

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history4.history['accuracy'], label='Train')
plt.plot(history4.history['val_accuracy'], label='Validation')
plt.title("Final Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history4.history['loss'], label='Train')
plt.plot(history4.history['val_loss'], label='Validation')
plt.title("Final Loss")
plt.legend()

plt.show()
