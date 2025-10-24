

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model

from data_preprocessing import PlantDiseaseDataLoader


def load_config(config_path='models/training_config.json'):
    """Load training configuration."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='models'):
    """
    Plot and save confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Directory to save plot
    """
    cm = confusion_matrix(y_true, y_pred)

    # Plot full confusion matrix
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(20, 18))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    plt.title('Normalized Confusion Matrix', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix_normalized.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrices saved to {save_path}")


def plot_per_class_accuracy(y_true, y_pred, class_names, save_path='models'):
    """
    Plot per-class accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Directory to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    # Sort by accuracy
    sorted_indices = np.argsort(per_class_accuracy)
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_accuracies = per_class_accuracy[sorted_indices]

    plt.figure(figsize=(12, 14))
    colors = ['red' if acc < 0.8 else 'orange' if acc < 0.9 else 'green'
              for acc in sorted_accuracies]
    plt.barh(range(len(sorted_classes)), sorted_accuracies, color=colors, alpha=0.7)
    plt.yticks(range(len(sorted_classes)), sorted_classes)
    plt.xlabel('Accuracy', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14, pad=20)
    plt.axvline(x=0.9, color='blue', linestyle='--', alpha=0.5, label='90% threshold')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Per-class accuracy plot saved to {save_path}")

    return per_class_accuracy


def plot_sample_predictions(model, generator, class_names, num_samples=16, save_path='models'):
    """
    Plot sample predictions with images.

    Args:
        model: Trained model
        generator: Data generator
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Directory to save plot
    """
    # Get a batch of images
    images, labels = next(generator)
    predictions = model.predict(images)

    # Select random samples
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)

    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()

    for idx, i in enumerate(indices):
        if idx >= 16:
            break

        axes[idx].imshow(images[i])

        true_label = class_names[np.argmax(labels[i])]
        pred_label = class_names[np.argmax(predictions[i])]
        confidence = np.max(predictions[i]) * 100

        color = 'green' if true_label == pred_label else 'red'
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                            fontsize=8, color=color)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Sample predictions plot saved to {save_path}")


def evaluate_model(model_path='models/best_model.h5',
                   data_dir='data/PlantVillage',
                   config_path='models/training_config.json'):
    """
    Evaluate trained model and generate comprehensive reports.

    Args:
        model_path: Path to saved model
        data_dir: Path to dataset
        config_path: Path to training configuration
    """
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # Load configuration
    print("\n[1/6] Loading configuration...")
    config_data = load_config(config_path)
    class_names = config_data['class_names']
    img_size = config_data['config']['img_size']
    batch_size = config_data['config']['batch_size']

    print(f"Number of classes: {len(class_names)}")
    print(f"Image size: {img_size}")

    # Load model
    print("\n[2/6] Loading trained model...")
    model = load_model(model_path)
    print(f"Model loaded from: {model_path}")

    # Load validation data
    print("\n[3/6] Loading validation data...")
    data_loader = PlantDiseaseDataLoader(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        validation_split=config_data['config']['validation_split']
    )
    _, val_generator, _ = data_loader.create_data_generators()

    # Evaluate on validation set
    print("\n[4/6] Evaluating model...")
    results = model.evaluate(val_generator, verbose=1)

    print(f"\nValidation Loss: {results[0]:.4f}")
    print(f"Validation Accuracy: {results[1]:.4f}")
    if len(results) > 2:
        print(f"Top-3 Accuracy: {results[2]:.4f}")

    # Get predictions
    print("\n[5/6] Generating predictions...")
    y_pred_probs = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_generator.classes

    # Classification report
    print("\n[6/6] Generating evaluation reports...")
    print("\nClassification Report:")
    print("=" * 70)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # Save classification report
    report_path = os.path.join('models', 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Validation Loss: {results[0]:.4f}\n")
        f.write(f"Validation Accuracy: {results[1]:.4f}\n")
        if len(results) > 2:
            f.write(f"Top-3 Accuracy: {results[2]:.4f}\n")
        f.write("\n" + "=" * 70 + "\n\n")
        f.write(report)
    print(f"\nClassification report saved to: {report_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_true, y_pred, class_names)
    per_class_acc = plot_per_class_accuracy(y_true, y_pred, class_names)
    plot_sample_predictions(model, val_generator, class_names)

    # Identify best and worst performing classes
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    worst_5_idx = np.argsort(per_class_acc)[:5]
    best_5_idx = np.argsort(per_class_acc)[-5:]

    print("\nWorst Performing Classes:")
    for idx in worst_5_idx:
        print(f"  {class_names[idx]}: {per_class_acc[idx] * 100:.2f}%")

    print("\nBest Performing Classes:")
    for idx in best_5_idx[::-1]:
        print(f"  {class_names[idx]}: {per_class_acc[idx] * 100:.2f}%")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - confusion_matrix_normalized.png")
    print("  - per_class_accuracy.png")
    print("  - sample_predictions.png")
    print("  - classification_report.txt")
    print("\nNext step: Run 'python app.py' to launch the web application")


if __name__ == "__main__":
    evaluate_model()