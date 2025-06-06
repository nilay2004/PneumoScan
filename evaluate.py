import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def load_data():
    test_dir = "dataset/chest_xray/test"
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False
    )
    return test_generator


def evaluate_model(model_path, test_generator):
    model = tf.keras.models.load_model(model_path)
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Model: {model_path}, Accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")

    # Predictions
    y_pred = model.predict(test_generator)
    y_pred = (y_pred > 0.5).astype(int)
    y_true = test_generator.classes

    # Classification report
    print(f"\nClassification Report for {model_path}:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
    plt.title(f'Confusion Matrix: {os.path.basename(model_path)}')
    plt.savefig(f"visualizations/{os.path.basename(model_path).replace('.keras', '')}_cm.png")
    plt.close()

    return test_accuracy, test_loss


def plot_metrics():
    models = ["cnn", "vgg16", "resnet", "mobilenet"]
    accuracies = []
    losses = []

    test_generator = load_data()

    for model_name in models:
        model_path = f"models/{model_name}_model.keras"
        acc, loss = evaluate_model(model_path, test_generator)
        accuracies.append(acc)
        losses.append(loss)

    # Plot accuracy
    plt.figure(figsize=(8, 5))
    plt.bar(models, accuracies, color='skyblue')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.savefig("visualizations/model_accuracy.png")
    plt.close()

    # Plot loss
    plt.figure(figsize=(8, 5))
    plt.bar(models, losses, color='salmon')
    plt.title('Model Loss Comparison')
    plt.ylabel('Loss')
    plt.ylim(0, max(losses) + 0.1)
    for i, v in enumerate(losses):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.savefig("visualizations/model_loss.png")
    plt.close()


if __name__ == "__main__":
    os.makedirs("visualizations", exist_ok=True)
    plot_metrics()