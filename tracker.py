# tracker.py
from utils import centroid
import numpy as np
from config import FUZZY_MATCH_RADIUS, FRAMES_REQUIRED

def update_group_tracking(groups, people_coords, active_groups, group_id_counter, frame_no, crowd_log_entries):
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
                print(f"=======Crowd confirmed at Frame {frame_no}: {len(group)} people========")
                logged = True
            new_active_groups[matched_id] = {'centroid': group_centroid, 'frames': frames_count, 'logged': logged}
            matched_ids.add(matched_id)
        else:
            group_id_counter += 1
            new_active_groups[group_id_counter] = {'centroid': group_centroid, 'frames': 1, 'logged': False}
            matched_ids.add(group_id_counter)

    active_groups = {gid: data for gid, data in new_active_groups.items() if gid in matched_ids}
    return active_groups, group_id_counter
