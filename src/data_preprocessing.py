
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf


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

        Returns:
            train_generator: Training data generator
            val_generator: Validation data generator
            class_names: List of class names
        """
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=self.validation_split
        )

        # Validation data (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=self.validation_split
        )

        # Create training generator
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )

        # Create validation generator
        val_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )

        self.class_names = list(train_generator.class_indices.keys())

        return train_generator, val_generator, self.class_names

    def create_test_generator(self, test_dir):
        """
        Create test data generator.

        Args:
            test_dir: Path to test dataset directory

        Returns:
            test_generator: Test data generator
        """
        test_datagen = ImageDataGenerator(rescale=1. / 255)

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
        Calculate class weights for imbalanced datasets.

        Args:
            train_generator: Training data generator

        Returns:
            class_weights: Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight

        # Get class labels
        classes = np.unique(train_generator.classes)

        # Compute class weights
        class_weights_values = compute_class_weight(
            'balanced',
            classes=classes,
            y=train_generator.classes
        )

        class_weights = dict(zip(classes, class_weights_values))

        return class_weights


def preprocess_single_image(img_path, img_size=224):
    """
    Preprocess a single image for prediction.

    Args:
        img_path: Path to image file
        img_size: Target image size

    Returns:
        Preprocessed image array
    """
    from tensorflow.keras.preprocessing import image

    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def get_dataset_statistics(data_dir):
    """
    Get statistics about the dataset.

    Args:
        data_dir: Path to dataset directory

    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_images': 0,
        'num_classes': 0,
        'class_distribution': {},
        'classes': []
    }

    if not os.path.exists(data_dir):
        return stats

    classes = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))])

    stats['num_classes'] = len(classes)
    stats['classes'] = classes

    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        num_images = len([f for f in os.listdir(class_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        stats['class_distribution'][class_name] = num_images
        stats['total_images'] += num_images

    return stats


if __name__ == "__main__":
    # Example usage
    data_dir = "data/PlantVillage"

    # Get dataset statistics
    stats = get_dataset_statistics(data_dir)
    print(f"Dataset Statistics:")
    print(f"Total Images: {stats['total_images']}")
    print(f"Number of Classes: {stats['num_classes']}")
    print(f"\nClass Distribution:")
    for class_name, count in sorted(stats['class_distribution'].items()):
        print(f"  {class_name}: {count}")

    # Create data loaders
    loader = PlantDiseaseDataLoader(data_dir)
    train_gen, val_gen, class_names = loader.create_data_generators()

    print(f"\nTraining samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Classes: {len(class_names)}")