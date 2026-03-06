import cv2

def draw_boxes(image, detections):

    annotated = image.copy()

    for d in detections:

        x1,y1,x2,y2 = d["bbox"]

        label = f"{d['class']} {d['confidence']:.2f}"

        cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(
            annotated,
            label,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

    return annotated