import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split


class PlantDiseaseDataLoader:
    """Handles loading and preprocessing of plant disease images."""

    def __init__(self, data_dir, img_size=224, batch_size=32, validation_split=0.2):
        """
        Initialize data loader.

        Args:
            data_dir: Path to dataset directory
            img_size: Target image size (default: 224)
            batch_size: Batch size for training (default: 32)
            validation_split: Fraction of data for validation (default: 0.2)
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.class_names = None

    def create_data_generators(self):
        """
        Create training and validation data generators with augmentation.
        Uses EfficientNet preprocessing.
        """

        # TRAIN: augmentation + EfficientNet preprocessing
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=self.validation_split,
        )

        # VAL: ONLY preprocessing (no augmentation)
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=self.validation_split,
        )

        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42,
        )

        val_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42,
        )

        self.class_names = list(train_generator.class_indices.keys())

        print(f"\n{'=' * 70}")
        print("DATA LOADING SUMMARY")
        print(f"{'=' * 70}")
        print(f"Training samples:    {train_generator.samples}")
        print(f"Validation samples:  {val_generator.samples}")
        print(f"Number of classes:   {len(self.class_names)}")
        print(f"Image size:          {self.img_size}x{self.img_size}")
        print(f"Batch size:          {self.batch_size}")
        print("Preprocessing:       EfficientNet (ImageNet normalization)")
        print(f"{'=' * 70}\n")

        return train_generator, val_generator, self.class_names

    def create_test_generator(self, test_dir):
        """
        Create test generator with proper preprocessing only.
        """
        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        return test_generator

    def get_class_weights(self, train_generator):
        """
        Compute class weights so rare classes get up-weighted.
        """
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(train_generator.classes)

        class_weights_values = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=train_generator.classes
        )

        class_weights = dict(zip(classes, class_weights_values))

        print(f"\nClass weights computed for {len(classes)} classes")
        print("Top 5 most weighted classes:")
        sorted_weights = sorted(
            class_weights.items(), key=lambda x: x[1], reverse=True
        )[:5]
        for class_idx, weight in sorted_weights:
            class_name = list(train_generator.class_indices.keys())[class_idx]
            print(f"  {class_name}: {weight:.3f}")

        return class_weights


def preprocess_single_image(img_path, img_size=224):
    """
    Load a single image and preprocess it for EfficientNet inference.
    """
    from tensorflow.keras.preprocessing import image

    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_dataset_statistics(data_dir):
    """
    Count classes and images per class so we can sanity check the dataset.
    """
    stats = {
        'total_images': 0,
        'num_classes': 0,
        'class_distribution': {},
        'classes': []
    }

    if not os.path.exists(data_dir):
        print(f"Warning: Directory {data_dir} does not exist!")
        return stats

    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    stats['num_classes'] = len(classes)
    stats['classes'] = classes

    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        num_images = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        stats['class_distribution'][class_name] = num_images
        stats['total_images'] += num_images

    return stats
