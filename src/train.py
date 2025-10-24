"""
Training script for plant disease classification model.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Import custom modules
from data_preprocessing import PlantDiseaseDataLoader
from model import (
    create_and_compile_model,
    get_callbacks,
    unfreeze_model,
    compile_model,
    print_model_summary
)

# Configuration
CONFIG = {
    'data_dir': 'data/PlantVillage',
    'model_save_dir': 'models',
    'img_size': 224,
    'batch_size': 32,
    'epochs_phase1': 20,  # Initial training with frozen base
    'epochs_phase2': 30,  # Fine-tuning with unfrozen layers
    'learning_rate_phase1': 0.001,
    'learning_rate_phase2': 0.0001,
    'validation_split': 0.2,
    'patience': 10,
    'unfreeze_layers': 20
}


def plot_training_history(history, phase_name, save_path='models'):
    """
    Plot and save training history.

    Args:
        history: Training history object
        phase_name: Name of training phase (for plot title)
        save_path: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title(f'{phase_name} - Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title(f'{phase_name} - Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'training_history_{phase_name.lower().replace(" ", "_")}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training plots saved to {save_path}")


def save_training_config(config, class_names, save_path='models'):
    """
    Save training configuration and class names.

    Args:
        config: Configuration dictionary
        class_names: List of class names
        save_path: Directory to save configuration
    """
    training_info = {
        'config': config,
        'class_names': class_names,
        'num_classes': len(class_names),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    config_path = os.path.join(save_path, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(training_info, f, indent=4)

    print(f"Training configuration saved to {config_path}")


def train_model():
    """
    Main training function with two-phase training strategy.

    Phase 1: Train with frozen base model
    Phase 2: Fine-tune with unfrozen top layers
    """
    print("=" * 70)
    print("PLANT DISEASE CLASSIFICATION - MODEL TRAINING")
    print("=" * 70)

    # Create model save directory
    os.makedirs(CONFIG['model_save_dir'], exist_ok=True)

    # Load data
    print("\n[1/6] Loading and preprocessing data...")
    data_loader = PlantDiseaseDataLoader(
        data_dir=CONFIG['data_dir'],
        img_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        validation_split=CONFIG['validation_split']
    )

    train_generator, val_generator, class_names = data_loader.create_data_generators()

    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Batch size: {CONFIG['batch_size']}")

    # Calculate class weights for handling imbalanced data
    print("\n[2/6] Calculating class weights...")
    class_weights = data_loader.get_class_weights(train_generator)

    # Create model
    print("\n[3/6] Creating model...")
    model = create_and_compile_model(
        num_classes=len(class_names),
        img_size=CONFIG['img_size'],
        learning_rate=CONFIG['learning_rate_phase1']
    )
    print_model_summary(model)

    # Phase 1: Initial training with frozen base
    print("\n[4/6] Phase 1: Training with frozen base model...")
    print(f"Epochs: {CONFIG['epochs_phase1']}")
    print(f"Learning rate: {CONFIG['learning_rate_phase1']}")

    callbacks_phase1 = get_callbacks(
        model_save_path=os.path.join(CONFIG['model_save_dir'], 'best_model_phase1.h5'),
        patience=CONFIG['patience']
    )

    history_phase1 = model.fit(
        train_generator,
        epochs=CONFIG['epochs_phase1'],
        validation_data=val_generator,
        callbacks=callbacks_phase1,
        class_weight=class_weights,
        verbose=1
    )

    # Plot Phase 1 results
    plot_training_history(history_phase1, 'Phase 1 - Transfer Learning', CONFIG['model_save_dir'])

    # Phase 2: Fine-tuning
    print("\n[5/6] Phase 2: Fine-tuning with unfrozen layers...")
    print(f"Unfreezing top {CONFIG['unfreeze_layers']} layers")
    print(f"Epochs: {CONFIG['epochs_phase2']}")
    print(f"Learning rate: {CONFIG['learning_rate_phase2']}")

    model = unfreeze_model(model, num_layers_to_unfreeze=CONFIG['unfreeze_layers'])
    model = compile_model(model, learning_rate=CONFIG['learning_rate_phase2'])

    callbacks_phase2 = get_callbacks(
        model_save_path=os.path.join(CONFIG['model_save_dir'], 'best_model.h5'),
        patience=CONFIG['patience']
    )

    history_phase2 = model.fit(
        train_generator,
        epochs=CONFIG['epochs_phase2'],
        validation_data=val_generator,
        callbacks=callbacks_phase2,
        class_weight=class_weights,
        verbose=1
    )

    # Plot Phase 2 results
    plot_training_history(history_phase2, 'Phase 2 - Fine Tuning', CONFIG['model_save_dir'])

    # Save configuration
    print("\n[6/6] Saving training configuration...")
    save_training_config(CONFIG, class_names, CONFIG['model_save_dir'])

    # Final evaluation
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print("\nPhase 1 Results:")
    print(f"  Best Training Accuracy: {max(history_phase1.history['accuracy']):.4f}")
    print(f"  Best Validation Accuracy: {max(history_phase1.history['val_accuracy']):.4f}")

    print("\nPhase 2 Results:")
    print(f"  Best Training Accuracy: {max(history_phase2.history['accuracy']):.4f}")
    print(f"  Best Validation Accuracy: {max(history_phase2.history['val_accuracy']):.4f}")

    print(f"\nBest model saved to: {os.path.join(CONFIG['model_save_dir'], 'best_model.h5')}")
    print(f"Training plots saved to: {CONFIG['model_save_dir']}")
    print("\nNext steps:")
    print("  1. Run 'python src/evaluate.py' to evaluate the model")
    print("  2. Run 'python app.py' to launch the web application")

    return model, history_phase1, history_phase2


if __name__ == "__main__":
    train_model()