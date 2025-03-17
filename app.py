from PIL import Image
import numpy as np
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template
import os

# Constants
IMAGE_SIZE = (224, 224)
SIMILARITY_THRESHOLD = 0.8  # Default value

def resize_image(image, size=IMAGE_SIZE):
    return image.resize(size)

def normalize_image(image):
    return np.array(image) / 255.0

def extract_features(image):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
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
    
    static_folder = os.path.join('static')
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)

    for i, image in enumerate(comparison_images):
        image_name = f"image_{i}.png"
        image_path = os.path.join(static_folder, image_name)

        # Save the image using PIL Image object
        img = resize_image(image)
        img.save(image_path)
        
        img_array = normalize_image(img)
        comparison_features.append(extract_features(img_array))
        image_names.append(image_name)
    
    most_similar_indices = find_most_similar(input_features, comparison_features)
    similar_images = [image_names[index] for index in most_similar_indices]
    
    return similar_images

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global SIMILARITY_THRESHOLD

    if request.method == 'POST':
        try:
            # Read similarity threshold from form
            SIMILARITY_THRESHOLD = float(request.form.get('similarity_threshold', 0.8))

            input_image = request.files['input_image']
            comparison_images = request.files.getlist('comparison_images')

            # Ensure input image is not empty
            if input_image.filename == '':
                return "No selected file for input image."

            # Read and process input image
            input_image = Image.open(input_image.stream)

            # Process comparison images
            comparison_images_pil = [Image.open(img.stream) for img in comparison_images]

            similar_images = compare_images(input_image, comparison_images_pil)

            return render_template('result.html', images=similar_images)

        except Exception as e:
            # Print the error details for debugging
            print(f"Error: {e}")
            return "An error occurred. Please try again."

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
