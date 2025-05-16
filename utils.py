import numpy as np
from sklearn.cluster import DBSCAN
from config import DISTANCE_THRESHOLD, MIN_PEOPLE_IN_GROUP

# DBSCAN clustering
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

# Calculate the centroid of a group of points
def centroid(points):
    xs, ys = zip(*points)
    return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))