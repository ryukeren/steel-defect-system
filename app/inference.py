
import onnxruntime as ort
import numpy as np
import cv2
from .config import MODEL_PATH

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name


def preprocess(image):
    img = cv2.resize(image, (640, 640))
    img = img.astype("float32") / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    return img


def run_inference(image):

    input_tensor = preprocess(image)

    outputs = session.run(None, {input_name: input_tensor})

    return outputs