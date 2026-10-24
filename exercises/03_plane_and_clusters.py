"""Exercise 03: segmentation - RANSAC plane fitting and DBSCAN clustering.

Classic robotics trick: find the dominant plane (floor / road), remove it,
then cluster what is left into separate objects.

Run:  python exercises/03_plane_and_clusters.py [--show]
"""
import numpy as np
import open3d as o3d

from common import label_colors, save_views, show

o3d.utility.random.seed(0)  # make RANSAC repeatable
pcd = o3d.io.read_point_cloud(o3d.data.PCDPointCloud().path).voxel_down_sample(0.02)
print("points:", len(pcd.points))

# 1. RANSAC: repeatedly fit a plane to 3 random points, keep the best one -----
planes = []
rest = pcd
for i in range(3):  # peel off the 3 largest planes
    model, inliers = rest.segment_plane(distance_threshold=0.02, ransac_n=3,
                                        num_iterations=1000)
    a, b, c, d = model
    plane = rest.select_by_index(inliers)
    plane.paint_uniform_color(np.eye(3)[i])
    planes.append(plane)
    print(f"plane {i}: {a:+.2f}x {b:+.2f}y {c:+.2f}z {d:+.2f} = 0 | {len(inliers)} pts")
    rest = rest.select_by_index(inliers, invert=True)

rest.paint_uniform_color([0.5, 0.5, 0.5])
save_views(planes + [rest], "03_planes.png", "3 largest planes (R, G, B)", point_size=1)

# 2. DBSCAN on what is left: points within eps of each other join a cluster ---
labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=10))
n_clusters = labels.max() + 1
print(f"DBSCAN found {n_clusters} clusters, {np.sum(labels < 0)} noise points")
colors, _ = label_colors(labels)
rest.colors = o3d.utility.Vector3dVector(colors)

# 3. A bounding box per cluster (what a detector would output) -----------------
boxes = []
sizes = np.bincount(labels[labels >= 0])
for k in np.argsort(sizes)[::-1][:5]:
    obj = rest.select_by_index(np.where(labels == k)[0])
    box = obj.get_axis_aligned_bounding_box()
    box.color = (1, 0, 0)
    boxes.append(box)
    print(f"  cluster {k}: {sizes[k]} pts, box size {box.get_extent().round(2)} m")

save_views([rest], "03_clusters.png", f"{n_clusters} DBSCAN clusters after removing planes",
           point_size=1)
show([rest, *planes, *boxes], "03 planes + clusters")

# TRY IT:
#  - Increase distance_threshold to 0.05. Do planes "eat" nearby objects?
#  - DBSCAN eps controls merging: try 0.02 and 0.1.
