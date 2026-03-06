import numpy as np
import cv2

def generate_heatmap(image, detections):

    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]

        heatmap[y1:y2, x1:x2] += 1

    heatmap = cv2.GaussianBlur(heatmap, (51,51), 0)

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    heatmap = heatmap.astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

    return overlay