

import os
import json
import numpy as np
import gradio as gr
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Global variables
MODEL = None
CLASS_NAMES = None
IMG_SIZE = 224


def load_model_and_config():
    """Load the trained model and configuration."""
    global MODEL, CLASS_NAMES, IMG_SIZE

    model_path = 'models/best_model.h5'
    config_path = 'models/training_config.json'

    # Load model
    print("Loading model...")
    MODEL = load_model(model_path)
    print("Model loaded successfully!")

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    CLASS_NAMES = config['class_names']
    IMG_SIZE = config['config']['img_size']

    print(f"Loaded {len(CLASS_NAMES)} classes")


def preprocess_image(img):
    """
    Preprocess image for model prediction.

    Args:
        img: PIL Image or numpy array

    Returns:
        Preprocessed image array
    """
    # Convert to PIL if numpy array
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype('uint8'))

    # Resize
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_disease(img):
    """
    Predict plant disease from image.

    Args:
        img: Input image (PIL Image or numpy array)

    Returns:
        Dictionary with class names as keys and confidence scores as values
    """
    if img is None:
        return None

    # Preprocess image
    processed_img = preprocess_image(img)

    # Make prediction
    predictions = MODEL.predict(processed_img, verbose=0)

    # Get top 5 predictions
    top_5_idx = np.argsort(predictions[0])[-5:][::-1]

    # Create results dictionary
    results = {}
    for idx in top_5_idx:
        class_name = CLASS_NAMES[idx]
        confidence = float(predictions[0][idx])
        results[class_name] = confidence

    return results


def format_class_name(class_name):
    """Format class name for better readability."""
    # Replace underscores with spaces and title case
    formatted = class_name.replace('_', ' ').title()
    return formatted


def get_disease_info(class_name):
    """
    Get additional information about the predicted disease.

    Args:
        class_name: Name of the disease class

    Returns:
        Information string about the disease
    """
    # This is a simplified version. In production, you'd have a database
    if 'healthy' in class_name.lower():
        return "✅ The plant appears to be healthy. No disease detected."
    else:
        return f"⚠️ Disease detected: {format_class_name(class_name)}\n\nRecommendation: Consult with an agricultural expert for proper treatment."


def predict_and_explain(img):
    """
    Predict disease and provide explanation.

    Args:
        img: Input image

    Returns:
        Tuple of (prediction results, explanation text)
    """
    if img is None:
        return None, "Please upload an image to get started."

    # Get predictions
    results = predict_disease(img)

    if results is None:
        return None, "Error processing image. Please try again."

    # Get top prediction
    top_class = list(results.keys())[0]
    top_confidence = results[top_class]

    # Generate explanation
    explanation = f"## 🔍 Analysis Results\n\n"
    explanation += f"**Top Prediction:** {format_class_name(top_class)}\n\n"
    explanation += f"**Confidence:** {top_confidence * 100:.2f}%\n\n"
    explanation += get_disease_info(top_class)
    explanation += f"\n\n### 📊 Top 5 Predictions:\n\n"

    for i, (class_name, confidence) in enumerate(list(results.items())[:5], 1):
        explanation += f"{i}. **{format_class_name(class_name)}** - {confidence * 100:.2f}%\n"

    return results, explanation


# Create Gradio interface
def create_interface():
    """Create and configure Gradio interface."""

    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .output-markdown {
        font-size: 16px;
    }
    """

    # Example images (if you have them in an examples folder)
    examples = []
    if os.path.exists('images'):
        example_images = [os.path.join('images', f) for f in os.listdir('images')
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:5]
        examples = [[img] for img in example_images]

    # Create interface
    with gr.Blocks(css=custom_css, title="Plant Disease Classifier") as demo:
        gr.Markdown(
            """
            # 🌿 Plant Disease Classifier

            Upload an image of a plant leaf to identify potential diseases. This AI model can recognize 38 different plant diseases and healthy leaves.

            **Supported plants:** Tomato, Potato, Pepper, Corn, Apple, Cherry, Peach, Grape, and more.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Plant Leaf Image",
                    type="pil",
                    height=400
                )

                predict_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")

                gr.Markdown("### 📸 Tips for best results:")
                gr.Markdown(
                    """
                    - Use clear, well-lit images
                    - Focus on the affected leaf area
                    - Avoid blurry or distant shots
                    - Single leaf works best
                    """
                )

            with gr.Column(scale=1):
                output_label = gr.Label(
                    label="Prediction Results",
                    num_top_classes=5
                )

                output_explanation = gr.Markdown(
                    label="Detailed Analysis",
                    value="Upload an image and click 'Analyze Image' to get started."
                )

        # Add examples if available
        if examples:
            gr.Examples(
                examples=examples,
                inputs=input_image,
                label="Example Images"
            )

        gr.Markdown(
            """
            ---
            ### ⚠️ Disclaimer
            This tool is for educational and informational purposes only. For accurate diagnosis and treatment, 
            please consult with agricultural experts or plant pathologists.

            ### 🤖 About the Model
            - **Architecture:** EfficientNetB0 with transfer learning
            - **Training Data:** PlantVillage Dataset (~54,000 images)
            - **Accuracy:** ~95-97% on validation set
            """
        )

        # Connect button to function
        predict_btn.click(
            fn=predict_and_explain,
            inputs=input_image,
            outputs=[output_label, output_explanation]
        )

        # Also trigger on image upload
        input_image.change(
            fn=predict_and_explain,
            inputs=input_image,
            outputs=[output_label, output_explanation]
        )

    return demo


def main():
    """Main function to run the application."""
    print("=" * 70)
    print("PLANT DISEASE CLASSIFIER - WEB APPLICATION")
    print("=" * 70)

    # Load model and configuration
    try:
        load_model_and_config()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nPlease ensure:")
        print("  1. Model file exists at: models/best_model.h5")
        print("  2. Configuration file exists at: models/training_config.json")
        print("  3. Run 'python src/train.py' to train the model first")
        return

    # Create and launch interface
    print("\nCreating web interface...")
    demo = create_interface()

    print("\nLaunching application...")
    print("=" * 70)

    # Launch with share=True to create public link (optional)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set to True to create a public link
    )


if __name__ == "__main__":
    main()