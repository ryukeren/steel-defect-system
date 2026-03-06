from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io
import base64
import time
import logging
import logging
import json
from datetime import datetime
import os
from app.nms import apply_nms
from app.visualization import draw_boxes
from .inference import run_inference
from app.heatmap import generate_heatmap
from app.logger import save_log
from app.analytics_engine import compute_analytics


app = FastAPI(title="Steel Defect Detection API", version="1.0")

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

os.makedirs("results", exist_ok=True)
# Logging setup
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load ONNX model once
session = ort.InferenceSession("models/best.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


def preprocess(image):
    img = cv2.resize(image, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def apply_nms(detections):
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    final = []

    while detections:
        best = detections.pop(0)
        final.append(best)

        detections = [
            d for d in detections
            if compute_iou(best["bbox"], d["bbox"]) < IOU_THRESHOLD
        ]

    return final


@app.get("/")
def health():
    return {"status": "API running", "model": "YOLOv8 ONNX"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    start_time = time.time()

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = np.array(image)

        original_h, original_w = image.shape[:2]

        input_tensor = preprocess(image)

        outputs = session.run(None, {input_name: input_tensor})
        predictions = outputs[0][0].transpose(1, 0)

        boxes = predictions[:, :4]
        scores = predictions[:, 4:]

        detections = []

        for i in range(len(boxes)):

            class_id = np.argmax(scores[i])
            confidence = scores[i][class_id]

            if confidence < CONF_THRESHOLD:
                continue

            x, y, w, h = boxes[i]

            x1 = int((x - w / 2) * original_w / 640)
            y1 = int((y - h / 2) * original_h / 640)
            x2 = int((x + w / 2) * original_w / 640)
            y2 = int((y + h / 2) * original_h / 640)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_w, x2)
            y2 = min(original_h, y2)

            area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            ratio = area / (original_w * original_h)

            if ratio < 0.05:
                severity = "Minor"
            elif ratio < 0.15:
                severity = "Moderate"
            else:
                severity = "Severe"

            detections.append({
                "class": CLASS_NAMES[class_id],
                "confidence": float(confidence),
                "bbox": [x1, y1, x2, y2],
                "severity": severity
            })

        detections = apply_nms(detections)

        severity_summary = {"Minor": 0, "Moderate": 0, "Severe": 0}

        for d in detections:
            severity_summary[d["severity"]] += 1

        annotated = draw_boxes(image, detections)
        heatmap_img = generate_heatmap(image, detections)


        _, buffer = cv2.imencode(".jpg", annotated)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
  
        _, heatmap_buffer = cv2.imencode(".jpg", heatmap_img)
        heatmap_base64 = base64.b64encode(heatmap_buffer).decode("utf-8")

        processing_time = round((time.time() - start_time) * 1000, 2)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        result_data = {
            "timestamp": timestamp,
            "total_detections": len(detections),
            "severity_summary": severity_summary
        }
        save_log(result_data)
        with open(f"results/{timestamp}.json", "w") as f:
            json.dump(result_data, f)

    

        logging.info(
            f"Detections={len(detections)}, Severity={severity_summary}, Time={processing_time}ms"
        )

        return {
            "status": "success",
            "processing_time_ms": processing_time,
            "total_detections": len(detections),
            "severity_summary": severity_summary,
            "detections": detections,
            "annotated_image_base64": img_base64,
            "heatmap_image_base64": heatmap_base64
        }

    except Exception as e:

        logging.error(f"Prediction error: {str(e)}")

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )
@app.get("/analytics")
def analytics():

    stats = compute_analytics()

    return stats

