import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import PlantDiseaseDataLoader


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='models'):
    """
    Plot and save confusion matrices (raw counts + normalized).
    """
    cm = confusion_matrix(y_true, y_pred)

    # 1. Raw confusion matrix
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, 'confusion_matrix.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # 2. Normalized confusion matrix (per-row accuracy)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion'}
    )
    plt.title('Normalized Confusion Matrix', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, 'confusion_matrix_normalized.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    print(f"✓ Confusion matrices saved to {save_path}")


def plot_per_class_accuracy(y_true, y_pred, class_names, save_path='models'):
    """
    Plot per-class accuracy and return per-class accuracy array.
    """
    cm = confusion_matrix(y_true, y_pred)  # shape [num_classes, num_classes]
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    # Sort classes by accuracy (lowest first)
    sorted_indices = np.argsort(per_class_accuracy)
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_acc = per_class_accuracy[sorted_indices]

    plt.figure(figsize=(12, 14))
    colors = [
        ('red' if acc < 0.8 else ('orange' if acc < 0.9 else 'green'))
        for acc in sorted_acc
    ]
    plt.barh(
        range(len(sorted_classes)),
        sorted_acc,
        color=colors,
        alpha=0.7
    )
    plt.yticks(range(len(sorted_classes)), sorted_classes)
    plt.xlabel('Accuracy', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14, pad=20)
    plt.axvline(
        x=0.9,
        color='blue',
        linestyle='--',
        alpha=0.5,
        label='90% threshold'
    )
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, 'per_class_accuracy.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    print(f"✓ Per-class accuracy plot saved to {save_path}")
    return per_class_accuracy


def plot_sample_predictions(model, generator, class_names,
                            num_samples=16, save_path='models'):
    """
    Show a grid of model predictions on a batch from the generator.
    Color title green if correct, red if wrong.
    """
    # Get one batch
    images, labels = next(generator)
    preds = model.predict(images)

    # Random subset
    n = min(num_samples, len(images), 16)
    idxs = np.random.choice(len(images), n, replace=False)

    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    axes = axes.ravel()

    for plot_i, img_i in enumerate(idxs):
        ax = axes[plot_i]

        true_idx = np.argmax(labels[img_i])
        pred_idx = np.argmax(preds[img_i])
        true_label = class_names[true_idx]
        pred_label = class_names[pred_idx]
        confidence = float(np.max(preds[img_i])) * 100.0

        ax.imshow(images[img_i] / 255.0)  # generator may have preprocessed images already;
                                          # if it's already in EfficientNet space you might skip `/255.0`
        title_color = 'green' if (true_idx == pred_idx) else 'red'
        ax.set_title(
            f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%",
            fontsize=8,
            color=title_color
        )
        ax.axis('off')

    # Hide any unused subplots if n < 16
    for j in range(plot_i + 1, rows * cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, 'sample_predictions.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    print(f"✓ Sample predictions plot saved to {save_path}")


def evaluate_model(
    model_path='models/final_model.h5',
    class_names_path='models/class_names.json',
    data_dir='data/PlantVillage',
    img_size=224,
    batch_size=32,
    validation_split=0.2,
    save_path='models'
):
    """
    Full evaluation:
    - loads the trained model
    - rebuilds the validation generator (same preprocessing)
    - reports accuracy / top-3 accuracy
    - saves confusion matrix, per-class accuracy, and sample predictions
    - writes a classification_report.txt
    """

    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # Ensure output dir
    os.makedirs(save_path, exist_ok=True)

    # 1. Load class names
    print("\n[1/6] Loading class names...")
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"✓ Loaded {num_classes} classes")

    # 2. Load trained model
    print("\n[2/6] Loading trained model...")
    model = tf.keras.models.load_model(model_path)
    print(f"✓ Model loaded from: {model_path}")

    # 3. Build validation generator using same preprocessing
    print("\n[3/6] Building validation generator...")
    loader = PlantDiseaseDataLoader(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        validation_split=validation_split
    )
    # We only care about validation split here
    _, val_generator, _ = loader.create_data_generators()

    # 4. Evaluate model quantitatively
    print("\n[4/6] Evaluating model on validation data...")
    results = model.evaluate(val_generator, verbose=1)

    # results: [loss, acc, top_3_acc] if compiled with those metrics
    val_loss = results[0]
    val_acc = results[1]
    top3_acc = results[2] if len(results) > 2 else None

    print(f"\nValidation Loss:      {val_loss:.4f}")
    print(f"Validation Accuracy:  {val_acc:.4f} ({val_acc*100:.2f}%)")
    if top3_acc is not None:
        print(f"Top-3 Accuracy:       {top3_acc:.4f} ({top3_acc*100:.2f}%)")

    # 5. Predictions for report + confusion matrix
    print("\n[5/6] Generating predictions...")
    y_pred_probs = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)  # predicted class idx
    y_true = val_generator.classes            # true class idxs from generator

    # Classification report
    print("\nClassification Report:")
    print("=" * 70)
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print(report_text)

    # Save classification report to disk
    report_path = os.path.join(save_path, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Validation Loss:     {val_loss:.4f}\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        if top3_acc is not None:
            f.write(f"Top-3 Accuracy:     {top3_acc:.4f}\n")
        f.write("\n" + "=" * 70 + "\n\n")
        f.write(report_text)
    print(f"\n✓ Classification report saved to: {report_path}")

    # 6. Visual diagnostics
    print("\n[6/6] Creating visualizations...")

    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=save_path
    )

    per_class_acc = plot_per_class_accuracy(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=save_path
    )

    plot_sample_predictions(
        model=model,
        generator=val_generator,
        class_names=class_names,
        num_samples=16,
        save_path=save_path
    )

    # best / worst classes
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
    # These defaults match what train.py used.
    evaluate_model(
        model_path='models/final_model.h5',
        class_names_path='models/class_names.json',
        data_dir='data/PlantVillage',
        img_size=224,
        batch_size=32,
        validation_split=0.2,
        save_path='models'
    )
