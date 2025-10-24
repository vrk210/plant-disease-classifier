

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def create_model(num_classes, img_size=224, base_model_name='EfficientNetB0', trainable_layers=20):
    """
    Create a transfer learning model for plant disease classification.

    Args:
        num_classes: Number of output classes
        img_size: Input image size (default: 224)
        base_model_name: Name of base model (default: 'EfficientNetB0')
        trainable_layers: Number of layers to unfreeze for fine-tuning (default: 20)

    Returns:
        Compiled Keras model
    """
    # Select base model
    if base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(img_size, img_size, 3)
        )
    elif base_model_name == 'EfficientNetB3':
        base_model = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=(img_size, img_size, 3)
        )
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")

    # Freeze base model initially
    base_model.trainable = False

    # Build model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer and loss function.

    Args:
        model: Keras model to compile
        learning_rate: Initial learning rate (default: 0.001)

    Returns:
        Compiled model
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )

    return model


def get_callbacks(model_save_path='models/best_model.h5', patience=10):
    """
    Create training callbacks.

    Args:
        model_save_path: Path to save best model (default: 'models/best_model.h5')
        patience: Patience for early stopping (default: 10)

    Returns:
        List of callbacks
    """
    callbacks = [
        # Save best model
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),

        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    return callbacks


def unfreeze_model(model, num_layers_to_unfreeze=20):
    """
    Unfreeze top layers of base model for fine-tuning.

    Args:
        model: Keras model
        num_layers_to_unfreeze: Number of layers to unfreeze from the top

    Returns:
        Model with unfrozen layers
    """
    # Get base model (first layer)
    base_model = model.layers[0]

    # Unfreeze top layers
    base_model.trainable = True

    # Freeze all layers except the last num_layers_to_unfreeze
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    print(f"Unfroze {num_layers_to_unfreeze} layers for fine-tuning")
    print(f"Total trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights])}")

    return model


def create_and_compile_model(num_classes, img_size=224, learning_rate=0.001):
    """
    Create and compile model in one step.

    Args:
        num_classes: Number of output classes
        img_size: Input image size
        learning_rate: Initial learning rate

    Returns:
        Compiled model ready for training
    """
    model = create_model(num_classes, img_size)
    model = compile_model(model, learning_rate)

    return model


def print_model_summary(model):
    """
    Print detailed model summary.

    Args:
        model: Keras model
    """
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)
    model.summary()

    print("\n" + "=" * 70)
    print("TRAINABLE PARAMETERS")
    print("=" * 70)

    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_count = sum([tf.size(w).numpy() for w in model.non_trainable_weights])

    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")
    print(f"Total parameters: {trainable_count + non_trainable_count:,}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Example: Create and display model
    num_classes = 38
    model = create_and_compile_model(num_classes)
    print_model_summary(model)

    # Example: Fine-tuning setup
    print("\nSetting up for fine-tuning...")
    model = unfreeze_model(model, num_layers_to_unfreeze=20)
    model = compile_model(model, learning_rate=0.0001)
    print_model_summary(model)