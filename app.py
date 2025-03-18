from PIL import Image
import numpy as np
from tensorflow.keras.applications import VGG16  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template
import os

# Constants
IMAGE_SIZE = (224, 224)
SIMILARITY_THRESHOLD = 0.8  # Default similarity threshold

app = Flask(__name__)

# Ensure static folder exists for saving images
STATIC_FOLDER = "static"
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

# Model initialization
def load_model():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    return Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

model = load_model()

def resize_image(image, size=IMAGE_SIZE):
    return image.resize(size)

def normalize_image(image):
    return np.array(image) / 255.0

def extract_features(image):
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return model.predict(image).flatten()

def find_most_similar(input_features, image_features):
    similarities = cosine_similarity([input_features], image_features)
    similar_indices = [i for i, score in enumerate(similarities[0]) if score >= SIMILARITY_THRESHOLD]
    return similar_indices

def compare_images(input_image, comparison_images):
    input_image = resize_image(input_image)
    input_image = normalize_image(input_image)
    input_features = extract_features(input_image)

    comparison_features = []
    image_names = []

    for i, image in enumerate(comparison_images):
        image_name = f"image_{i}.png"
        image_path = os.path.join(STATIC_FOLDER, image_name)

        # Save image
        img = resize_image(image)
        img.save(image_path)

        # Extract features
        img_array = normalize_image(img)
        comparison_features.append(extract_features(img_array))
        image_names.append(image_name)

    most_similar_indices = find_most_similar(input_features, comparison_features)
    similar_images = [image_names[index] for index in most_similar_indices]

    return similar_images

@app.route("/", methods=["GET"])
def home():
    return "Hello, Flask is running!"

@app.route("/upload", methods=["POST"])
def upload():
    if "user_photo" not in request.files or "reference_photo" not in request.files:
        return jsonify({"error": "Missing files"}), 400

    user_photo = request.files["user_photo"]
    reference_photo = request.files["reference_photo"]

    user_photo_path = os.path.join(STATIC_FOLDER, "user_uploaded.jpg")
    reference_photo_path = os.path.join(STATIC_FOLDER, "reference_uploaded.jpg")

    user_photo.save(user_photo_path)
    reference_photo.save(reference_photo_path)

    # Open the images
    user_image = Image.open(user_photo_path)
    reference_image = Image.open(reference_photo_path)

    # Compare images
    similar_images = compare_images(user_image, [reference_image])

    return jsonify({"message": "Images received and processed", "similar_images": similar_images}), 200

@app.route("/compare", methods=["POST"])
def compare():
    global SIMILARITY_THRESHOLD

    try:
        SIMILARITY_THRESHOLD = float(request.form.get("similarity_threshold", 0.8))

        input_image = request.files["input_image"]
        comparison_images = request.files.getlist("comparison_im
