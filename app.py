from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import requests
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import os

# Flask app initialization
app = Flask(__name__)

# Constants
IMAGE_SIZE = (224, 224)
SIMILARITY_THRESHOLD = 0.8  # Default threshold

# Ensure static folder exists
STATIC_FOLDER = "static"
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

# Load the pre-trained model once to optimize performance
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# ---- IMAGE PROCESSING FUNCTIONS ----
def download_image(image_url):
    """Download image from URL and convert to PIL format."""
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        image = Image.open(response.raw).convert("RGB")
        return image
    else:
        raise ValueError("Failed to download image")

def preprocess_image(image):
    """Resize and normalize image."""
    image = image.resize(IMAGE_SIZE)
    image = img_to_array(image) / 255.0  # Normalize pixels
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def extract_features(image):
    """Extract deep learning features using VGG16."""
    return model.predict(image).flatten()

def find_most_similar(input_features, comparison_features):
    """Find similar images using cosine similarity."""
    similarities = cosine_similarity([input_features], comparison_features)
    return [i for i, score in enumerate(similarities[0]) if score >= SIMILARITY_THRESHOLD]

# ---- FLASK ROUTES ----
@app.route("/", methods=["GET"])
def home():
    """Simple test route to check if Flask is running."""
    return "Hello, Flask is running!"

@app.route("/upload", methods=["POST"])
def upload():
    """Upload endpoint to receive user and reference photos."""
    if "user_photo" not in request.files or "reference_photo" not in request.files:
        return jsonify({"error": "Missing files"}), 400

    user_photo = request.files["user_photo"]
    reference_photo = request.files["reference_photo"]

    user_photo.save(os.path.join(STATIC_FOLDER, "user_uploaded.jpg"))
    reference_photo.save(os.path.join(STATIC_FOLDER, "reference_uploaded.jpg"))

    return jsonify({"message": "Images received and saved"}), 200

@app.route("/compare", methods=["POST"])
def compare():
    """Compare input image with reference images."""
    global SIMILARITY_THRESHOLD

    input_image_url = request.json.get("input_image_url")
    comparison_image_urls = request.json.get("comparison_image_urls")
    
    if not input_image_url or not comparison_image_urls:
        return jsonify({"error": "Missing input_image_url or comparison_image_urls"}), 400
    
    try:
        SIMILARITY_THRESHOLD = float(request.json.get("similarity_threshold", 0.8))
        
        # Download and process input image
        input_image = download_image(input_image_url)
        input_image = preprocess_image(input_image)
        input_features = extract_features(input_image)
        
        comparison_features = []
        image_names = []
        
        for i, img_url in enumerate(comparison_image_urls):
            try:
                img = download_image(img_url)
                processed_img = preprocess_image(img)
                comparison_features.append(extract_features(processed_img))
                image_names.append(img_url)
            except Exception as e:
                print(f"Error processing {img_url}: {e}")
                continue
        
        most_similar_indices = find_most_similar(input_features, comparison_features)
        similar_images = [image_names[idx] for idx in most_similar_indices]

        return jsonify({"similar_images": similar_images}), 200

    except Exception as e:
        print(f"Error during comparison: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)