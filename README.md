# Plant Disease Classifier

An end-to-end deep learning application that identifies diseases in plant leaves using computer vision.

## Project Overview

The classifier uses transfer learning with a pre-trained EfficientNetB0 model to classify 38 different classes of plant diseases and healthy leaves across various crops including tomato, potato, pepper, and more.

### Key Features
- **38 Disease Classes**: Covers multiple crops and disease types
- **Transfer Learning**: Uses EfficientNetB0 pre-trained on ImageNet
- **Interactive Web App**: Easy-to-use Gradio interface for real-time predictions
- **Comprehensive Analysis**: Includes EDA, training metrics, and error analysis
- **Production Ready**: Model optimization and deployment considerations

## Dataset

**PlantVillage Dataset**
- **Source**: [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
- **Size**: ~54,000 images
- **Classes**: 38 (various plant diseases + healthy)
- **Image Format**: RGB images of plant leaves

### Downloading the Dataset

1. Install Kaggle CLI: `pip install kaggle`
2. Set up Kaggle API credentials (place `kaggle.json` in `~/.kaggle/`)
3. Download dataset:
```bash
   kaggle datasets download -d arjuntejaswi/plant-village
   unzip plant-village.zip -d data/
```

Alternatively, download manually from Kaggle and extract to `data/` folder.

### Installation

1. Clone the repository:
```bash
   git clone https://github.com/yourusername/plant-disease-classifier.git
   cd plant-disease-classifier
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Download the dataset (see Dataset section above)

### Project Structure
```
plant-disease-classifier/
├── data/                          # Dataset directory
│   └── PlantVillage/             # Extracted dataset
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   └── 02_model_training.ipynb   # Model training and evaluation
├── src/
│   ├── data_preprocessing.py      # Data loading and augmentation
│   ├── model.py                   # Model architecture
│   ├── train.py                   # Training script
│   └── evaluate.py                # Evaluation and visualization
├── models/
│   └── best_model.h5              # Saved trained model
├── app.py                         # Gradio web application
├── requirements.txt
└── README.md
```

## Model Training

Run the complete training pipeline:
```bash
python src/train.py
```

This will:
- Load and preprocess the data
- Apply data augmentation
- Train the model with early stopping
- Save the best model to `models/`
- Generate training plots

### Training Configuration
- **Base Model**: EfficientNetB0 (pre-trained on ImageNet)
- **Image Size**: 224x224
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Optimizer**: Adam with learning rate scheduling
- **Loss**: Categorical Crossentropy

### Data Augmentation
- Random rotation (±20°)
- Width/height shift (±20%)
- Shear transformation
- Zoom (±20%)
- Horizontal flip

## Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | ~99% |
| Validation Accuracy | ~95-97% |
| Test Accuracy | ~95-97% |

*Actual results will vary based on training run*

## 🌐 Web Application

Launch the interactive web app:
```bash
python app.py
```

Then open your browser to `http://localhost:7860`

### Features
- Upload plant leaf images
- Get instant disease predictions
- View confidence scores
- See top 3 predictions
- Example images included

##  Notebooks

### 1. Exploratory Data Analysis (`01_eda.ipynb`)
- Dataset overview and statistics
- Class distribution analysis
- Sample image visualization
- Image size and quality checks

### 2. Model Training (`02_model_training.ipynb`)
- Complete training pipeline
- Model architecture visualization
- Training/validation curves
- Confusion matrix
- Per-class performance metrics
- Error analysis

## 🔧 Advanced Usage

### Custom Training
Modify training parameters in `src/train.py`:
```python
BATCH_SIZE = 32
EPOCHS = 50
IMG_SIZE = 224
LEARNING_RATE = 0.001
```

### Using the Trained Model
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model('models/best_model.h5')

# Prepare image
img = image.load_img('path/to/leaf.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]
```

## Model Evaluation

Run evaluation on test set:
```bash
python src/evaluate.py
```

This generates:
- Confusion matrix
- Classification report
- Per-class accuracy
- Misclassification analysis
- Sample predictions visualization
