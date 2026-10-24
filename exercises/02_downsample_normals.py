"""Exercise 02: clean up a cloud - downsampling, outlier removal, normals.

Raw sensor clouds are dense and noisy. Almost every pipeline starts with:
  voxel downsample -> remove outliers -> estimate normals.

Run:  python exercises/02_downsample_normals.py [--show]
"""
import time

import numpy as np
import open3d as o3d

from common import save_views, show

pcd = o3d.io.read_point_cloud(o3d.data.PCDPointCloud().path)
print("raw:", len(pcd.points), "points")

# 1. Voxel downsampling: one point per voxel (cube) of the given size ---------
for voxel in [0.01, 0.02, 0.05]:
    t0 = time.perf_counter()
    down = pcd.voxel_down_sample(voxel_size=voxel)
    print(f"voxel {voxel:.2f} m -> {len(down.points):6d} points "
          f"({(time.perf_counter() - t0) * 1000:.1f} ms)")
down = pcd.voxel_down_sample(0.02)

# 2. Add fake noise so outlier removal has something to do --------------------
rng = np.random.default_rng(42)
pts = np.asarray(down.points)
lo, hi = pts.min(0), pts.max(0)
noise = rng.uniform(lo, hi, size=(500, 3))  # random "dust" points
noisy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([pts, noise])))
print("with 500 fake outliers:", len(noisy.points))

# 3. Statistical outlier removal: drop points far from their neighbours -------
clean, kept_idx = noisy.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
removed = len(noisy.points) - len(kept_idx)
print(f"statistical outlier removal dropped {removed} points")
# Radius alternative: noisy.remove_radius_outlier(nb_points=16, radius=0.05)

inliers = noisy.select_by_index(kept_idx)
outliers = noisy.select_by_index(kept_idx, invert=True)
inliers.paint_uniform_color([0.6, 0.6, 0.6])
outliers.paint_uniform_color([1.0, 0.0, 0.0])
save_views([inliers, outliers], "02_outliers.png", "red = removed as outliers", point_size=1)

# 4. Normals: fit a small plane to each point's neighbourhood -----------------
clean.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
clean.orient_normals_towards_camera_location(np.zeros(3))  # consistent sign
n = np.asarray(clean.normals)
print("normals shape:", n.shape, "| unit length:", np.allclose(np.linalg.norm(n, axis=1), 1))

# Color by normal direction: |nz| ~ 1 means horizontal surface (floor/table).
clean.colors = o3d.utility.Vector3dVector(np.abs(n))
horizontal = np.abs(n[:, 2]) > 0.9
print(f"{horizontal.mean() * 100:.1f}% of points lie on near-horizontal surfaces")
save_views([clean], "02_normals.png", "colored by |normal| (blue = horizontal)", point_size=1)
show([clean], "02 normals (press N to draw them)")

# TRY IT:
#  - Change std_ratio to 1.0 and 3.0. What gets removed?
#  - Try max_nn=5 for normals. Why do the colors get noisy?
