import os
import json
import tensorflow as tf
from data_loader import PlantDiseaseDataLoader, get_dataset_statistics
from model import (
    create_and_compile_model,
    unfreeze_model,
    compile_model,
    get_callbacks,
    print_model_summary,
    train_model_two_phase,
)


def main():
    print("\n" + "="*70)
    print("PLANT DISEASE CLASSIFICATION TRAINING")
    print("="*70)

    # === CONFIG ===
    DATA_DIR = "data/PlantVillage"   # <-- make sure this exists in Colab after you unzip
    IMG_SIZE = 224
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    EPOCHS_PHASE1 = 10
    EPOCHS_PHASE2 = 20

    print("\nConfiguration:")
    print(f"  Data directory:   {DATA_DIR}")
    print(f"  Image size:       {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"  Validation split: {VALIDATION_SPLIT}")
    print(f"  Phase 1 epochs:   {EPOCHS_PHASE1}")
    print(f"  Phase 2 epochs:   {EPOCHS_PHASE2}")

    # === DATASET STATS ===
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)

    stats = get_dataset_statistics(DATA_DIR)
    print(f"\nDataset contains:")
    print(f"  Total images:     {stats['total_images']}")
    print(f"  Number of classes:{stats['num_classes']}")
    if stats['num_classes'] == 0:
        print(f"\n❌ ERROR: No classes found in {DATA_DIR}")
        return

    # === LOAD DATA ===
    loader = PlantDiseaseDataLoader(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT
    )

    train_generator, val_generator, class_names = loader.create_data_generators()

    class_weights = loader.get_class_weights(train_generator)

    # === MODEL ===
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)

    num_classes = len(class_names)
    print(f"\nBuilding model for {num_classes} classes...")

    model = create_and_compile_model(
        num_classes=num_classes,
        img_size=IMG_SIZE,
        learning_rate=0.001
    )

    print_model_summary(model)

    # === TRAIN TWO PHASES ===
    print("\n" + "="*70)
    print("STARTING TWO-PHASE TRAINING")
    print("="*70)

    histories, model = train_model_two_phase(
        model=model,
        train_generator=train_generator,
        val_generator=val_generator,
        epochs_phase1=EPOCHS_PHASE1,
        epochs_phase2=EPOCHS_PHASE2,
        class_weights=class_weights
    )

    # === SAVE MODEL + CLASS NAMES ===
    print("\n" + "="*70)
    print("SAVING FINAL MODEL")
    print("="*70)

    os.makedirs('models', exist_ok=True)
    final_model_path = 'models/final_model.h5'
    model.save(final_model_path)
    print(f"✓ Final model saved to: {final_model_path}")

    with open('models/class_names.json', 'w') as f:
        json.dump(class_names, f)
    print(f"✓ Class names saved to: models/class_names.json")

    # === FINAL EVAL ===
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    print("\nEvaluating on validation set...")
    val_loss, val_accuracy, val_top3_accuracy = model.evaluate(val_generator, verbose=0)

    print(f"\nFinal Validation Metrics:")
    print(f"  Loss:         {val_loss:.4f}")
    print(f"  Accuracy:     {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"  Top-3 Acc.:   {val_top3_accuracy:.4f} ({val_top3_accuracy*100:.2f}%)")
    print("="*70 + "\n")

    print("✓ Training complete!")


if __name__ == "__main__":
    # GPU check
    print("\n" + "="*70)
    print("SYSTEM CHECK")
    print("="*70)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU available: {len(gpus)} GPU(s) detected:")
        for gpu in gpus:
            print("  -", gpu)
    else:
        print("⚠ No GPU detected - training will be slower")
        print("  (In Colab: Runtime -> Change runtime type -> GPU)")
    print(f"\nTensorFlow version: {tf.__version__}")
    print("="*70)

    main()
