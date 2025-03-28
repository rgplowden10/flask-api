from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import os
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity

# Flask app initialization
app = Flask(__name__)

# Constants
IMAGE_SIZE = (224, 224)
SIMILARITY_THRESHOLD = 0.8  # Default threshold

# Ensure static folder exists for storing images
STATIC_FOLDER = "static"
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

# Load the pre-trained model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# ---- IMAGE PROCESSING FUNCTIONS ----
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
    """Serve the homepage with the form."""
    return render_template("index.html")

@app.route("/compare", methods=["POST"])
def compare():
    """Compare input image with reference images."""
    global SIMILARITY_THRESHOLD

    try:
        # Check if the required files are in the request
        if "user_photo" not in request.files or "reference_photos" not in request.files:
            return jsonify({"error": "Missing files"}), 400

        # Get the uploaded images
        user_photo = request.files["user_photo"]
        reference_photos = request.files.getlist("reference_photos")

        # Save the user photo
        user_photo_path = os.path.join(STATIC_FOLDER, "user_uploaded.jpg")
        user_photo.save(user_photo_path)

        # Process user photo
        user_image = Image.open(user_photo_path)
        user_image = preprocess_image(user_image)
        user_features = extract_features(user_image)

        # Process reference photos
        comparison_features = []
        image_names = []
        for ref_photo in reference_photos:
            ref_photo_path = os.path.join(STATIC_FOLDER, ref_photo.filename)
            ref_photo.save(ref_photo_path)
            ref_image = Image.open(ref_photo_path)
            ref_image = preprocess_image(ref_image)
            comparison_features.append(extract_features(ref_image))
            image_names.append(ref_photo.filename)

        # Compare the features
        most_similar_indices = find_most_similar(user_features, comparison_features)
        similar_images = [image_names[idx] for idx in most_similar_indices]

        # Return results as JSON
        response_data = {
            "input_image_url": f"/static/user_uploaded.jpg",
            "similar_images": [f"/static/{name}" for name in similar_images],
            "confidence_score": round(np.mean([cosine_similarity([user_features], [comparison_features[idx]])[0][0] for idx in most_similar_indices]), 2) if most_similar_indices else 0
        }

        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error: {str(e)}")  # Log error in terminal
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
