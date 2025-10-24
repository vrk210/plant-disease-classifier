# Quick Start Guide - Plant Disease Classifier


## Step 1: Clone and Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/plant-disease-classifier.git
cd plant-disease-classifier

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Download Dataset

### Option A: Using Kaggle API (Recommended)
```bash
# Install Kaggle CLI
pip install kaggle

# Set up Kaggle API credentials
# 1. Go to https://www.kaggle.com/account
# 2. Create new API token (downloads kaggle.json)
# 3. Place kaggle.json in ~/.kaggle/ (or C:\Users\YourUsername\.kaggle\ on Windows)

# Download dataset
kaggle datasets download -d arjuntejaswi/plant-village
unzip plant-village.zip -d data/
```

### Option B: Manual Download

1. Go to [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
2. Click "Download" button
3. Extract the zip file to the `data/` folder
4. Your structure should be: `data/PlantVillage/[class_folders]/`

## Step 3: Explore the Data (Optional)
```bash
# Launch Jupyter notebook for EDA
jupyter notebook notebooks/01_eda.ipynb
```

Or run the data preprocessing script to see statistics:
```bash
python src/data_preprocessing.py
```

## Step 4: Train the Model

### Quick Training (Script)
```bash
python src/train.py
```

This will:
- Load and preprocess data
- Train in two phases (transfer learning + fine-tuning)
- Save the best model to `models/best_model.h5`
- Generate training plots
- Take approximately 2-4 hours on GPU, longer on CPU

### Interactive Training (Notebook)
```bash
jupyter notebook notebooks/02_model_training.ipynb
```

Follow the notebook cells to train step by step with visualizations.

### Expected Results
- Training Accuracy: ~99%
- Validation Accuracy: ~95-97%

## Step 5: Evaluate the Model
```bash
python src/evaluate.py
```

This generates:
- Confusion matrix
- Per-class accuracy plot
- Sample predictions
- Classification report

## Step 6: Launch Web Application
```bash
python app.py
```

Then open your browser to: `http://localhost:7860`

### Using the Web App:
1. Upload an image of a plant leaf
2. Click "Analyze Image" or wait for automatic analysis
3. View prediction results and confidence scores
4. Get disease information and recommendations



## Common Issues and Solutions

### Issue: "No module named 'tensorflow'"
**Solution:** Make sure you've activated your virtual environment and installed dependencies:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "CUDA out of memory" during training
**Solution:** Reduce batch size in the training script or config:
- Open `src/train.py`
- Change `'batch_size': 32` to `'batch_size': 16` or `8`

### Issue: Dataset not found
**Solution:** Ensure your data structure is correct:
```
plant-disease-classifier/
└── data/
    └── PlantVillage/
        ├── Apple___Apple_scab/
        ├── Apple___Black_rot/
        └── ... (other class folders)
```

## Alternative: Use Pre-trained Model

If you don't want to train from scratch, you can use a pre-trained model:

1. Download pre-trained model (if available) and place in `models/`
2. Skip training step
3. Go directly to evaluation or web app
