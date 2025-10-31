from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

# ------------------------------
# Load model
# ------------------------------
MODEL_PATH = "black_pepper_model.h5"   # your trained model path
model = tf.keras.models.load_model(MODEL_PATH)

# Classes must match your training dataset folders
CLASS_NAMES = ["black_pepper_healthy", "black_pepper_leaf_blight", "black_pepper_yellow_mottle_virus"]

IMG_SIZE = (224, 224)  # width, height (RGB handled separately)

# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        img = None

        # Case 1: base64 image from camera capture
        if request.is_json:
            data = request.get_json()
            if "image" not in data:
                return jsonify({"error": "No image provided"}), 400
            image_data = data["image"].split(",")[1]
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Case 2: uploaded file
        elif "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400
            img = Image.open(file).convert("RGB")

        if img is None:
            return jsonify({"error": "No valid image received"}), 400

        # ---------------- Preprocess ----------------
        img = img.resize(IMG_SIZE)           # only width & height
        arr = np.array(img) / 255.0          # normalize
        arr = np.expand_dims(arr, axis=0)    # add batch dimension: (1, 224, 224, 3)

        # ---------------- Predict ----------------
        preds = model.predict(arr)
        idx = np.argmax(preds[0])
        confidence = float(np.max(preds[0]))

        return jsonify({
            "class": CLASS_NAMES[idx],
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
