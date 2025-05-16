import cv2
import pandas as pd
from config import *
from detection import detect_people
from utils import cluster_groups
from tracker import update_group_tracking

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_no = 0
group_id_counter = 0
active_groups = {}
crowd_log_entries = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_no += 1

    people_coords, bboxes = detect_people(frame)
    groups = cluster_groups(people_coords)
    active_groups, group_id_counter = update_group_tracking(
        groups, people_coords, active_groups,
        group_id_counter, frame_no, crowd_log_entries
    )

    all_crowd_indices = set(idx for group in groups for idx in group)
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        color = (0, 0, 255) if i in all_crowd_indices else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.putText(frame, f"Frame: {frame_no}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()

# Save to CSV
df = pd.DataFrame(crowd_log_entries, columns=["Frame Number", "Person Count in Crowd"])
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n Output saved to: {OUTPUT_VIDEO}")
print(f" Crowd log saved to: {OUTPUT_CSV} ({len(df)} events)")
