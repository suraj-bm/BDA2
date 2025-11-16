from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Load your trained YOLO model
model = YOLO("best.pt")   # <-- replace with your best.pt path if needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    # Save uploaded file
    file = request.files['image']
    unique_filename = str(uuid.uuid4()) + ".jpg"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(image_path)

    # Run YOLO prediction
    results = model.predict(source=image_path, save=False)

    # Open the uploaded image for drawing boxes
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Load a font for labels
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    detailed_detections = []
    class_counts = {}

    # Process detections
    for r in results:
        for box in r.boxes:
            cls_idx = int(box.cls[0])
            cls = r.names[cls_idx]       # original YOLO name
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            # -----------------------------------------------------
            # 🔄 SWAP LOGIC (Healthy ↔ Pepper_blight)
            # -----------------------------------------------------
            normalized = cls.lower().replace("_", "").replace(" ", "")

            if normalized == "healthy":
                cls = "Pepper_blight"
            elif normalized == "pepperblight":
                cls = "Healthy"
            # -----------------------------------------------------

            # Count classes
            class_counts[cls] = class_counts.get(cls, 0) + 1

            # Save for frontend list
            detailed_detections.append({
                "class": cls,
                "confidence": round(conf, 2)
            })

            # Draw bounding box
            x1, y1, x2, y2 = map(int, xyxy)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Draw label
            label = f"{cls} {conf*100:.1f}%"
            draw.text((x1, y1 - 20), label, fill="red", font=font)

    # Save rendered output image
    output_dir = "static/runs/predict"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/{unique_filename}"
    img.save(output_path)

    # Render frontend with all results
    return render_template(
        "index.html",
        result_image=output_path,
        detections=detailed_detections,
        class_counts=class_counts
    )


if __name__ == "__main__":
    app.run(debug=True)
