def compute_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2-x1) * max(0, y2-y1)

    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])

    union = box1_area + box2_area - inter_area

    return inter_area/union if union > 0 else 0


def apply_nms(detections, iou_threshold=0.4):

    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

    final = []

    while detections:

        best = detections.pop(0)
        final.append(best)

        detections = [
            d for d in detections
            if compute_iou(best["bbox"], d["bbox"]) < iou_threshold
        ]

    return final