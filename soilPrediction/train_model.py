# ============================================================
# SmartAgro — train_model.py
# 
# WHAT THIS DOES:
#   Trains a CNN image classifier on soil images using
#   Transfer Learning with MobileNetV2 (pretrained on ImageNet).
#   Saves the final model as  soil_model.h5
#
# HOW TO RUN:
#   python train_model.py
#
# DATASET FOLDER STRUCTURE EXPECTED:
#   dataset/
#     train/
#       Clay/       ← put ~200+ clay soil images here
#       Sandy/
#       Loam/
#       Black/
#     val/
#       Clay/
#       Sandy/
#       Loam/
#       Black/
#
# WHERE TO GET DATA:
#   https://www.kaggle.com/datasets/jayaprakashpondy/soil-image-dataset
# ============================================================

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CONFIG  ← easy to tweak
# ─────────────────────────────────────────
IMG_SIZE    = (224, 224)   # MobileNetV2 expects 224x224
BATCH_SIZE  = 32
EPOCHS      = 20           # increase for better accuracy
DATASET_DIR = "dataset"    # folder with train/ and val/
MODEL_OUT   = "soil_model.h5"
CLASS_NAMES = ["Black", "Clay", "Loam", "Sandy"]   # must match folder names A-Z


# ─────────────────────────────────────────
# STEP 1 — Data Augmentation
# Augmentation creates extra training variety by
# randomly flipping, rotating, and zooming images.
# This prevents overfitting on small datasets.
# ─────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,          # normalize pixel values 0-1
    validation_split=0.2,       # Use 20% of data for validation
    rotation_range=20,          # rotate images up to 20 degrees
    width_shift_range=0.1,      # shift image left/right
    height_shift_range=0.1,
    zoom_range=0.2,             # zoom in/out
    horizontal_flip=True,       # mirror the image
    brightness_range=[0.8, 1.2] # simulate different lighting
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255           # only normalize, NO augmentation for validation
)

# Load images from folders
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",   # multiple classes
    subset='training'           # Set this to 'training'
)

val_generator = val_datagen.flow_from_directory(
    'dataset/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset='validation'         # Set this to 'validation'
)

print("\n✅ Classes found:", train_generator.class_indices)
print("   Total training images:", train_generator.samples)
print("   Total validation images:", val_generator.samples)


# ─────────────────────────────────────────
# STEP 2 — Build the CNN Model
#
# We use TRANSFER LEARNING:
# MobileNetV2 was already trained on 1.2 million ImageNet images,
# so it already knows how to detect edges, textures, and shapes.
# We freeze those learned weights and only train our new top layers
# to recognize the 4 soil types.
#
# This means we need MUCH less data and training time!
# ─────────────────────────────────────────

# Load MobileNetV2 WITHOUT its top classification layer
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),   # 224 x 224 x 3 (RGB)
    include_top=False,            # remove ImageNet's 1000-class head
    weights="imagenet"            # use pretrained weights
)

# Freeze the base model — we do NOT retrain these layers
base_model.trainable = False

# Build our custom classifier on top
num_classes = len(CLASS_NAMES)

model = models.Sequential([
    base_model,

    # Flatten 7x7x1280 feature map to a vector
    layers.GlobalAveragePooling2D(),

    # Fully connected hidden layer
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),        # stabilizes training
    layers.Dropout(0.4),               # dropout prevents overfitting

    # Output layer — one neuron per soil class
    layers.Dense(num_classes, activation="softmax")
])

model.summary()


# ─────────────────────────────────────────
# STEP 3 — Compile the Model
# ─────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ─────────────────────────────────────────
# STEP 4 — Train!
# ─────────────────────────────────────────

# Early stopping — stops training if accuracy stops improving
# (prevents wasted time + overfitting)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,             # stop if no improvement for 5 epochs
    restore_best_weights=True
)

# Auto-reduce learning rate if stuck
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)

print("\n🚀 Starting training...\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop, reduce_lr]
)


# ─────────────────────────────────────────
# STEP 5 — Fine-tune (optional but improves accuracy)
# Unfreeze the last 30 layers of MobileNetV2 and
# retrain at a very low learning rate.
# ─────────────────────────────────────────
print("\n🔧 Fine-tuning top layers of base model...\n")

base_model.trainable = True
# Only train last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # very low LR
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stop, reduce_lr]
)


# ─────────────────────────────────────────
# STEP 6 — Save the Model
# ─────────────────────────────────────────
model.save(MODEL_OUT)
print(f"\n✅ Model saved to: {MODEL_OUT}")


# ─────────────────────────────────────────
# STEP 7 — Plot Training Accuracy Graph
# ─────────────────────────────────────────
def plot_history(h1, h2):
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.title("Model Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.title("Model Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_graph.png")
    print("📊 Training graph saved to training_graph.png")
    plt.show()

plot_history(history, history_fine)