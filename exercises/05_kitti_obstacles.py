"""Exercise 05: a real automotive lidar scan - ground removal + obstacle detection.

KITTI's Velodyne HDL-64E spins 10x per second and returns ~120k points per scan.
This is the classic "non-learning" perception pipeline used on many robots:

  crop region of interest -> downsample -> remove ground (RANSAC)
  -> cluster (DBSCAN) -> filter clusters by size -> bounding boxes

Data: run `python tools/download_kitti.py` first.
Run:  python exercises/05_kitti_obstacles.py [--frame 0] [--show]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from common import OUT_DIR, label_colors, load_kitti_scan, show

ap = argparse.ArgumentParser()
ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--show", action="store_true")
args = ap.parse_args()
o3d.utility.random.seed(0)

# 1. Load and inspect ---------------------------------------------------------
xyz, refl = load_kitti_scan(args.frame)
dist = np.linalg.norm(xyz[:, :2], axis=1)
print(f"scan {args.frame}: {len(xyz)} points, range {dist.min():.1f}-{dist.max():.1f} m, "
      f"reflectance {refl.min():.2f}-{refl.max():.2f}")

# 2. Region of interest: 40 m ahead/behind, 20 m to each side, ignore the car itself
roi = (np.abs(xyz[:, 0]) < 40) & (np.abs(xyz[:, 1]) < 20) & (dist > 3) & (xyz[:, 2] < 2)
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[roi]))
pcd = pcd.voxel_down_sample(0.1)
print(f"ROI + 10 cm voxels: {len(pcd.points)} points")

# 3. Ground removal: the road is the biggest plane. Sensor is 1.73 m above it.
model, ground_idx = pcd.segment_plane(distance_threshold=0.2, ransac_n=3,
                                      num_iterations=200)
print("ground plane: {:+.3f}x {:+.3f}y {:+.3f}z {:+.3f} = 0".format(*model))
print(f"  -> sensor height above ground = {abs(model[3]) / np.linalg.norm(model[:3]):.2f} m")

# One plane rarely fits a whole street (slopes, sidewalks, curbs). Refine locally:
# in each 2 m x 2 m cell, points within 0.25 m of the cell's lowest point are
# ground, as long as that lowest point is near the global plane.
pts_all = np.asarray(pcd.points)
height = (pts_all @ np.array(model[:3]) + model[3]) / np.linalg.norm(model[:3])
cell = np.floor(pts_all[:, :2] / 2.0).astype(int)
_, cell_id = np.unique(cell, axis=0, return_inverse=True)
cell_id = cell_id.ravel()
cell_min = np.full(cell_id.max() + 1, np.inf)
np.minimum.at(cell_min, cell_id, height)
is_ground = (height - cell_min[cell_id] < 0.25) & (cell_min[cell_id] < 0.6)
is_ground[ground_idx] |= np.abs(height[ground_idx]) < 0.2
print(f"single plane: {len(ground_idx)} ground pts | with grid refinement: {is_ground.sum()}")

ground = pcd.select_by_index(np.where(is_ground)[0])
objects = pcd.select_by_index(np.where(~is_ground)[0])
print(f"ground {len(ground.points)} pts, non-ground {len(objects.points)} pts")

# 4. Cluster the non-ground points ---------------------------------------------
labels = np.array(objects.cluster_dbscan(eps=0.5, min_points=10))
print(f"DBSCAN: {labels.max() + 1} clusters")

# 5. Keep clusters with a car/pedestrian/cyclist-like size ---------------------
pts = np.asarray(objects.points)
boxes = []
for k in range(labels.max() + 1):
    c = pts[labels == k]
    ext = c.max(0) - c.min(0)
    if 0.3 < ext[2] < 3.0 and max(ext[0], ext[1]) < 8 and len(c) > 20:
        box = o3d.geometry.AxisAlignedBoundingBox(c.min(0), c.max(0))
        box.color = (1, 0, 0)
        boxes.append(box)
print(f"{len(boxes)} obstacle candidates after size filtering")
for b in sorted(boxes, key=lambda b: np.linalg.norm(b.get_center()[:2]))[:5]:
    cx, cy, _ = b.get_center()
    print(f"  at x={cx:+5.1f} y={cy:+5.1f} m, size {b.get_extent().round(1)}")

# 6. Bird's-eye view image (the most common way to look at lidar data) ----------
colors, _ = label_colors(labels)
fig, ax = plt.subplots(figsize=(11, 6))
g = np.asarray(ground.points)
ax.scatter(g[:, 0], g[:, 1], s=0.3, c="lightgray")
ax.scatter(pts[:, 0], pts[:, 1], s=0.6, c=colors)
for b in boxes:
    (x0, y0, _), (x1, y1, _) = b.get_min_bound(), b.get_max_bound()
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=1))
ax.plot(0, 0, "k^", ms=10)
ax.set(aspect="equal", xlabel="x forward (m)", ylabel="y left (m)",
       title=f"KITTI frame {args.frame}: ground (gray), clusters, obstacle boxes (red)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/05_kitti_bev.png", dpi=110)
print("  saved output/05_kitti_bev.png")

if args.show:
    ground.paint_uniform_color([0.7, 0.7, 0.7])
    objects.colors = o3d.utility.Vector3dVector(colors)
    show([ground, objects, *boxes], "05 KITTI obstacles")

# TRY IT:
#  - Run on --frame 50 and --frame 100 (download all frames with --frames 108).
#  - Color the raw scan by reflectance: license plates and road paint light up.
#  - Replace AxisAlignedBoundingBox with get_oriented_bounding_box() per cluster.
#  - Comment out the grid refinement and compare the BEV image: what goes wrong?
