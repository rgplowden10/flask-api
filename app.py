from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

# Ensure static folder exists for storing images
STATIC_FOLDER = "static"
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

@app.route("/compare", methods=["GET"])
def home():
    """Serve the homepage with the form."""
    return render_template("index.html")

@app.route("/compare", methods=["POST"])
def compare():
    """Compare input image with reference images."""
    try:
        # Ensure files are provided
        if "user_photo" not in request.files or "reference_photos" not in request.files:
            return jsonify({"error": "Missing required files"}), 400

        # Save user photo
        user_photo = request.files["user_photo"]
        user_photo_path = os.path.join(STATIC_FOLDER, "user_uploaded.jpg")
        user_photo.save(user_photo_path)

        # Save reference photos
        reference_photos = request.files.getlist("reference_photos")
        reference_paths = []
        for ref_photo in reference_photos:
            ref_path = os.path.join(STATIC_FOLDER, ref_photo.filename)
            ref_photo.save(ref_path)
            reference_paths.append(ref_path)

        # Debugging: Log paths of saved files
        print("User Photo Path:", user_photo_path)
        print("Reference Image Paths:", reference_paths)

        # Dummy similarity logic (Replace this with actual comparison logic)
        similar_images = reference_paths[:2]  # Mock response

        # Create response
        response_data = {
            "input_image_url": f"/static/user_uploaded.jpg",
            "similar_images": [f"/static/{os.path.basename(path)}" for path in similar_images],
            "confidence_score": 0.85  # Example value
        }

        print("Response Data:", response_data)  # Debugging

        return jsonify(response_data), 200

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# -------------------------------
# NEW: ADDING /verify-signature ROUTE
# -------------------------------
@app.route("/verify-signature", methods=["POST"])
def verify_signature():
    """Verify a user's signature against reference signatures."""
    try:
        # Ensure required files are provided
        if "user_image" not in request.files or "reference_image" not in request.files:
            return jsonify({"error": "Missing image files"}), 400

        # Save uploaded files
        user_image = request.files["user_image"]
        user_image_path = os.path.join(STATIC_FOLDER, "user_signature.jpg")
        user_image.save(user_image_path)

        reference_image = request.files["reference_image"]
        reference_image_path = os.path.join(STATIC_FOLDER, "reference_signature.jpg")
        reference_image.save(reference_image_path)

        # Debugging: Print saved file paths
        print("User Signature Path:", user_image_path)
        print("Reference Signature Path:", reference_image_path)

        # Dummy verification logic (Replace with real signature analysis)
        verification_result = {
            "input_image_url": f"/static/user_signature.jpg",
            "reference_image_url": f"/static/reference_signature.jpg",
            "match_confidence": 0.92  # Example confidence score
        }

        print("Verification Result:", verification_result)  # Debugging

        return jsonify(verification_result), 200

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)