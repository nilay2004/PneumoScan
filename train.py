import os
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import shutil
import zipfile
import glob


# Download dataset using kagglehub
def download_dataset():
    dataset_name = "paultimothymooney/chest-xray-pneumonia"
    dataset_dir = "dataset/chest_xray"
    if not os.path.exists(dataset_dir):
        print("Downloading dataset...")
        path = kagglehub.dataset_download(dataset_name)
        print("Path to dataset files:", path)

        # Create dataset directory
        os.makedirs("dataset", exist_ok=True)
        target_dir = "dataset/chest_xray"

        # Check if the path contains the expected dataset structure
        if os.path.exists(os.path.join(path, "chest_xray")):
            # If chest_xray directory exists, move it to target_dir
            shutil.move(os.path.join(path, "chest_xray"), target_dir)
        else:
            # Look for a zip file in the downloaded path
            zip_files = glob.glob(os.path.join(path, "*.zip"))
            if zip_files:
                zip_file = zip_files[0]
                print(f"Extracting {zip_file}...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall("dataset")
            else:
                # If no zip file, check for extracted dataset
                for dir_name in os.listdir(path):
                    temp_dir = os.path.join(path, dir_name)
                    if os.path.isdir(temp_dir) and "train" in os.listdir(temp_dir):
                        shutil.move(temp_dir, target_dir)
                        break
                else:
                    raise FileNotFoundError("Could not find dataset structure in downloaded files.")

        # Verify directory structure
        for subdir in ["train", "val", "test"]:
            subdir_path = os.path.join(target_dir, subdir)
            if not os.path.exists(subdir_path):
                raise FileNotFoundError(f"Expected directory not found: {subdir_path}")

        print(f"Dataset successfully saved to {target_dir}")
    else:
        print("Dataset already exists at", dataset_dir)


# Data preparation
def prepare_data():
    train_dir = "dataset/chest_xray/train"
    val_dir = "dataset/chest_xray/val"
    test_dir = "dataset/chest_xray/test"

    # Verify directories exist
    for dir_path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
    )
    test_generator = val_datagen.flow_from_directory(
        test_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
    )
    return train_generator, val_generator, test_generator


# Custom CNN
def build_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# VGG16 Transfer Learning
def build_vgg16():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Lightweight ResNet
def build_resnet():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# MobileNetV2
def build_mobilenet():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Train models
def train_models():
    train_generator, val_generator, _ = prepare_data()
    os.makedirs("models", exist_ok=True)

    models = {
        "cnn": build_cnn(),
        "vgg16": build_vgg16(),
        "resnet": build_resnet(),
        "mobilenet": build_mobilenet()
    }

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    for name, model in models.items():
        print(f"Training {name} model...")
        history = model.fit(
            train_generator,
            epochs=20,
            validation_data=val_generator,
            callbacks=[early_stopping]
        )
        model.save(f"models/{name}_model.keras")
        print(f"Saved {name}_model.keras")


if __name__ == "__main__":
    download_dataset()
    train_models()
