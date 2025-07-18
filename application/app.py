# To run this file, open your terminal and type and change working directory to 'application' subfolder
# then run 'python app.py' and then click on the link provided 'http:/127.0.0.1:5000' or something similar

# this file is for the YOLO and flask implementation of the app.
from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)
model = YOLO("crack_classifier_yolov8.pt")
class_names = ['Low', 'Medium', 'High']
OUTPUT_FOLDER = 'static/results'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/static/results/<filename>")
def serve_result(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route("/detect_upload", methods=["POST"])
def detect_upload():
    file = request.files["image"]
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    original = image.copy()

    results = model(image)[0]
    image_url = None

    if results.boxes:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{class_names[cls]}: {conf:.2f}"
            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        filename = f"{uuid.uuid4().hex}.jpg"
        path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(path, original)
        image_url = f"/static/results/{filename}"
    else:
        # Save a version without bounding boxes but still return it for user reference
        filename = f"{uuid.uuid4().hex}_nocrack.jpg"
        path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(path, original)
        image_url = f"/static/results/{filename}"

    class_id = int(results.probs.top1)
    confidence = float(results.probs.top1conf)
    return jsonify({
        "class": class_names[class_id],
        "confidence": round(confidence, 2),
        "image_url": image_url
    })

if __name__ == "__main__":
    app.run(debug=True)

