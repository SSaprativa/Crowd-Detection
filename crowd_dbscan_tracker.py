# import and setup
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

# Configuration
VIDEO_PATH = 'dataset_video.mp4'
MODEL_PATH = 'yolov8n.pt'
OUTPUT_VIDEO = 'output_crowd_detection.mp4'
OUTPUT_CSV = 'crowd_detection_log.csv'

# Parameters
# max distance --> more close
DISTANCE_THRESHOLD = 80   
# min no of people to define the crowd
MIN_PEOPLE_IN_GROUP = 3    
# no of frames such that a group can be considered as crowd
FRAMES_REQUIRED = 10  
# centroid movement tolerence across frame         
FUZZY_MATCH_RADIUS = 30        

# Load model and video
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Tracking state
frame_no = 0
group_id_counter = 0
active_groups = {}  # {group_id: {'centroid': (x,y), 'frames': count, 'logged': bool}}
crowd_log_entries = []

def cluster_groups(coords):
    if len(coords) < MIN_PEOPLE_IN_GROUP:
        return []
    db = DBSCAN(eps=DISTANCE_THRESHOLD, min_samples=MIN_PEOPLE_IN_GROUP).fit(np.array(coords))
    labels = db.labels_
    groups = []
    for label in set(labels):
        if label == -1: continue
        group = [i for i, l in enumerate(labels) if l == label]
        if len(group) >= MIN_PEOPLE_IN_GROUP:
            groups.append(group)
    return groups

def centroid(points):
    xs, ys = zip(*points)
    return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_no += 1

    # Detect people
    results = model(frame)
    boxes = results[0].boxes
    people_coords = []
    bboxes = []

    for box in boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            people_coords.append((cx, cy))
            bboxes.append((x1, y1, x2, y2))

    # Cluster people into groups
    groups = cluster_groups(people_coords)
    new_active_groups = {}
    matched_ids = set()

    for group in groups:
        group_coords = [people_coords[i] for i in group]
        group_centroid = centroid(group_coords)

        matched_id = None
        for gid, data in active_groups.items():
            dist = np.linalg.norm(np.array(group_centroid) - np.array(data['centroid']))
            if dist <= FUZZY_MATCH_RADIUS:
                matched_id = gid
                break

        if matched_id is not None:
            frames_count = active_groups[matched_id]['frames'] + 1
            logged = active_groups[matched_id]['logged']
            if frames_count == FRAMES_REQUIRED and not logged:
                crowd_log_entries.append((frame_no, len(group)))
                print(f"✅ Crowd confirmed at Frame {frame_no}: {len(group)} people")
                logged = True
            new_active_groups[matched_id] = {'centroid': group_centroid, 'frames': frames_count, 'logged': logged}
            matched_ids.add(matched_id)
        else:
            group_id_counter += 1
            new_active_groups[group_id_counter] = {'centroid': group_centroid, 'frames': 1, 'logged': False}
            matched_ids.add(group_id_counter)

    active_groups = {gid: data for gid, data in new_active_groups.items() if gid in matched_ids}

    # Draw boxes
    all_crowd_indices = set(idx for group in groups for idx in group)
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        color = (0, 0, 255) if i in all_crowd_indices else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.putText(frame, f"Frame: {frame_no}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    out.write(frame)

cap.release()
out.release()

# Save CSV
df = pd.DataFrame(crowd_log_entries, columns=["Frame Number", "Person Count in Crowd"])
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n🎥 Output saved to: {OUTPUT_VIDEO}")
print(f"📄 Crowd log saved to: {OUTPUT_CSV} ({len(df)} events)")
