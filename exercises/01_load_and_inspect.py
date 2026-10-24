"""Exercise 01: load a point cloud and look at it as plain numbers.

Goal: understand that a point cloud is just an (N, 3) array of x, y, z
(plus optional colors / normals), and get comfortable converting between
Open3D objects and NumPy arrays.

Data: Open3D sample "fragment" scan of a room (downloads automatically, ~2 MB).
Run:  python exercises/01_load_and_inspect.py [--show]
"""
import numpy as np
import open3d as o3d

from common import OUT_DIR, save_views, show

# 1. Load ---------------------------------------------------------------
pcd = o3d.io.read_point_cloud(o3d.data.PCDPointCloud().path)
print(pcd)  # PointCloud with N points.

# 2. Open3D <-> NumPy -------------------------------------------------------
pts = np.asarray(pcd.points)  # (N, 3) float64, a view (no copy)
print("shape:", pts.shape, "dtype:", pts.dtype)
print("has colors:", pcd.has_colors(), "| has normals:", pcd.has_normals())
print("first 3 points:\n", pts[:3])

# 3. Basic statistics -------------------------------------------------------
print("min xyz :", pts.min(axis=0).round(3))
print("max xyz :", pts.max(axis=0).round(3))
print("centroid:", pts.mean(axis=0).round(3))
aabb = pcd.get_axis_aligned_bounding_box()
print("bounding box size (m):", aabb.get_extent().round(3))

# 4. Build a point cloud from your own NumPy array --------------------------
# Here: keep only points above the centroid height, and color them by height.
z = pts[:, 2]
mask = z > z.mean()
upper = o3d.geometry.PointCloud()
upper.points = o3d.utility.Vector3dVector(pts[mask])
t = (z[mask] - z[mask].min()) / np.ptp(z[mask])
upper.colors = o3d.utility.Vector3dVector(np.c_[t, 0.2 * np.ones_like(t), 1 - t])
print(f"kept {mask.sum()} / {len(pts)} points above mean height")

# 5. Save and view -----------------------------------------------------------
o3d.io.write_point_cloud(f"{OUT_DIR}/01_upper_half.ply", upper)
save_views([pcd], "01_fragment.png", "Open3D sample fragment")
save_views([upper], "01_upper_half.png", "points above mean height, colored by z")
show([pcd], "01 fragment")

# TRY IT:
#  - Crop with pcd.crop(aabb) using a smaller box you create yourself.
#  - Translate / rotate the cloud: pcd.translate(...), pcd.rotate(R, center=...)
#    with R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi / 4)).
