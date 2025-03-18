from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
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
def resize_image(image, size=IMAGE_SIZE):
    """Resize image to the required input size."""
    return image.resize(size)

def normalize_image(image):
    """Normalize image pixels between 0 and 1."""
    return np.array(image) / 255.0

def extract_features(image):
    """Extract deep learning features from an image using VGG16."""
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return model.predict(image).flatten()

def find_most_similar(input_features, image_features):
    """Find most similar images based on cosine similarity."""
    similarities = cosine_similarity([input_features], image_features)
    similar_indices = [i for i, score in enumerate(similarities[0]) if score >= SIMILARITY_THRESHOLD]
    return similar_indices

def compare_images(input_image, comparison_images):
    """Compare input image to reference images and return similar ones."""
    input_image = resize_image(input_image)
    input_image = normalize_image(input_image)
    input_features = extract_features(input_image)

    comparison_features = []
    image_names = []

    for i, image in enumerate(comparison_images):
        image_name = f"image_{i}.png"
        image_path = os.path.join(STATIC_FOLDER, image_name)

        # Save the image
        img = resize_image(image)
        img.save(image_path)

        # Extract features
        img_array = normalize_image(img)
        comparison_features.append(extract_features(img_array))
        image_names.append(image_name)

    most_similar_indices = find_most_similar(input_features, comparison_features)
    similar_images = [image_names[index] for index in most_similar_indices]

    return similar_images

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

    # Save files
    user_photo.save(os.path.join(STATIC_FOLDER, "user_uploaded.jpg"))
    reference_photo.save(os.path.join(STATIC_FOLDER, "reference_uploaded.jpg"))

    return jsonify({"message": "Images received and saved"}), 200

@app.route("/compare", methods=["POST"])
def compare():
    """Compare input image with reference images."""
    global SIMILARITY_THRESHOLD

    try:
        # Get similarity threshold
        SIMILARITY_THRESHOLD = float(request.form.get("similarity_threshold", 0.8))

        # Get uploaded images
        input_image = request.files["input_image"]
        comparison_images = request.files.getlist("comparison_images")

        if input_image.filename == "":
            return jsonify({"error": "No selected file for input image"}), 400

        input_image = Image.open(input_image.stream)
        comparison_images_pil = [Image.open(img.stream) for img in comparison_images]

        similar_images = compare_images(input_image, comparison_images_pil)

        return jsonify({"similar_images": similar_images}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred during comparison"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
