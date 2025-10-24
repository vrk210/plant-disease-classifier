import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def create_model(num_classes, img_size=224, base_model_name='EfficientNetB0'):
    """
    Build transfer learning model with EfficientNet backbone.
    """
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

    # Freeze base model for phase 1
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile for categorical classification of N plant disease classes.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=3, name='top_3_accuracy'
            ),
        ],
    )
    return model


def get_callbacks(model_save_path='best_model.h5', patience=10):
    """
    Checkpoint, early stopping, LR scheduler.
    """
    return [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]


def unfreeze_model(model, num_layers_to_unfreeze=20):
    """
    Phase 2 fine-tuning: unfreeze top N layers of EfficientNet.
    """
    base_model = model.layers[0]  # The EfficientNet backbone
    base_model.trainable = True

    # Freeze all but last N layers
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    print("\n" + "=" * 70)
    print("FINE-TUNING CONFIGURATION")
    print("=" * 70)
    print(f"Unfroze {num_layers_to_unfreeze} layers for fine-tuning")

    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params

    print(f"Trainable parameters:     {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters:         {total_params:,}")
    print("=" * 70 + "\n")

    return model


def print_model_summary(model):
    """
    Nice pretty summary with param counts.
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
    total = trainable_count + non_trainable_count

    print(f"Trainable parameters:     {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")
    print(f"Total parameters:         {total:,}")
    print("=" * 70 + "\n")


def create_and_compile_model(num_classes, img_size=224, learning_rate=0.001):
    """
    Convenience: build then compile.
    """
    model = create_model(num_classes=num_classes, img_size=img_size)
    model = compile_model(model, learning_rate=learning_rate)
    return model


def train_model_two_phase(
    model,
    train_generator,
    val_generator,
    epochs_phase1=10,
    epochs_phase2=20,
    class_weights=None,
):
    """
    Phase 1: train top classifier head (backbone frozen).
    Phase 2: unfreeze last N backbone layers and fine-tune at lower LR.
    """

    callbacks = get_callbacks(patience=10)

    print("\n===== PHASE 1: TRANSFER LEARNING (frozen base) =====")
    history1 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs_phase1,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Phase 2: unfreeze some of the EfficientNet base and fine-tune
    print("\n===== PHASE 2: FINE-TUNING (unfreeze top layers) =====")
    model = unfreeze_model(model, num_layers_to_unfreeze=20)

    # lower LR for fine-tune
    model = compile_model(model, learning_rate=1e-4)

    callbacks_finetune = get_callbacks(
        model_save_path='best_model_finetuned.h5', patience=7
    )

    history2 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs_phase2,
        callbacks=callbacks_finetune,
        class_weight=class_weights,
        verbose=1,
    )

    return (history1, history2), model
