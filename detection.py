import cv2
from ultralytics import YOLO

from config import MODEL_PATH

model = YOLO(MODEL_PATH)

def detect_people(frame):
    results = model(frame)
    boxes = results[0].boxes
    people_coords = []
    bboxes = []

    for box in boxes:
        if int(box.cls[0]) == 0:  # Class 0 is 'person'
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            people_coords.append((cx, cy))
            bboxes.append((x1, y1, x2, y2))
    return people_coords, bboxes
