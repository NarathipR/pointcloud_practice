"""Exercise 07: depth camera (ToF / structured light) -> point cloud.

A depth image stores one distance per pixel. With the pinhole camera model
you can turn every pixel (u, v, depth) into a 3D point:

    z = depth / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

ToF sensors (Azure Kinect, PMD, iPhone lidar, robot-vacuum sensors) produce
exactly this kind of image, plus characteristic noise you will clean up here.

Data: one frame from the TUM RGB-D benchmark (Kinect v1), via Open3D (~1 MB).
Run:  python exercises/07_depth_to_pointcloud.py [--show]
"""
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from common import OUT_DIR, save_views, show

data = o3d.data.SampleTUMRGBDImage()
color = np.asarray(o3d.io.read_image(data.color_path))
depth_raw = np.asarray(o3d.io.read_image(data.depth_path))  # uint16

# TUM stores depth in units of 1/5000 m; 0 means "no measurement".
DEPTH_SCALE = 5000.0
fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5  # default Kinect intrinsics used by TUM
h, w = depth_raw.shape
depth = depth_raw / DEPTH_SCALE
print(f"depth image {w}x{h}, valid {np.mean(depth > 0) * 100:.0f}% of pixels, "
      f"range {depth[depth > 0].min():.2f}-{depth.max():.2f} m")

# 1. Back-project by hand with NumPy -------------------------------------------
u, v = np.meshgrid(np.arange(w), np.arange(h))
valid = (depth > 0) & (depth < 4.0)
z = depth[valid]
x = (u[valid] - cx) * z / fx
y = (v[valid] - cy) * z / fy
pts_manual = np.c_[x, y, z]  # camera frame: x right, y down, z forward
print("manual back-projection:", pts_manual.shape)

# 2. Same thing with Open3D ------------------------------------------------------
intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
rgbd = o3d.geometry.RGBDImage.create_from_tum_format(
    o3d.io.read_image(data.color_path), o3d.io.read_image(data.depth_path),
    convert_rgb_to_intensity=False)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
# create_from_tum_format truncates at 4 m by default, same as our mask above
print("open3d back-projection:", len(pcd.points), "points",
      "| match:", np.allclose(np.sort(pts_manual[:, 2]), np.sort(np.asarray(pcd.points)[:, 2]),
                              atol=1e-3))

# Camera frame (x right, y down, z forward) -> robot frame (x forward, y left, z up),
# the ROS convention (REP 103). Same rotation every robot applies to its cameras.
CAM_TO_ROBOT = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
pcd.transform(CAM_TO_ROBOT)
save_views([pcd], "07_rgbd_cloud.png", "TUM frame as colored point cloud", point_size=0.3)

# 3. Simulate ToF artefacts ----------------------------------------------------------
rng = np.random.default_rng(0)
noisy = depth.copy()
# (a) noise grows with distance (roughly sigma = 1.5 mm + 0.25% of range)
noisy[valid] += rng.normal(0, 0.0015 + 0.0025 * depth[valid])
# (b) "flying pixels": at depth edges a ToF pixel mixes foreground and background
#     and reports a distance somewhere in between
gy, gx = np.gradient(depth)
edges = valid & (np.hypot(gx, gy) > 0.05)
mix = rng.uniform(0, 1, edges.sum())
noisy[edges] = depth[edges] + mix * 0.5 * np.sign(rng.normal(size=edges.sum()))
print(f"simulated noise + {edges.sum()} flying pixels")


def to_cloud(d):
    m = (d > 0) & (d < 4.0)
    zz = d[m]
    p = np.c_[(u[m] - cx) * zz / fx, (v[m] - cy) * zz / fy, zz]
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p)).transform(CAM_TO_ROBOT)


noisy_pcd = to_cloud(noisy)

# 4. Clean it up: drop pixels at depth discontinuities, then statistical outliers
gy, gx = np.gradient(noisy)
filtered = np.where(np.hypot(gx, gy) > 0.05, 0, noisy)  # jump-edge filter
clean_pcd = to_cloud(filtered).voxel_down_sample(0.01)
clean_pcd, _ = clean_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"noisy cloud {len(noisy_pcd.points)} pts -> cleaned {len(clean_pcd.points)} pts")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(color)
axes[0].set_title("RGB")
im = axes[1].imshow(np.where(depth > 0, depth, np.nan), cmap="turbo")
axes[1].set_title("depth (m), white = no return")
fig.colorbar(im, ax=axes[1], fraction=0.035)
axes[2].imshow(edges, cmap="gray")
axes[2].set_title("depth edges = where flying pixels appear")
for a in axes:
    a.axis("off")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/07_depth_images.png", dpi=100)
print("  saved output/07_depth_images.png")

noisy_pcd.paint_uniform_color([0.85, 0.2, 0.2])
clean_pcd.paint_uniform_color([0.1, 0.4, 0.8])
save_views([noisy_pcd], "07_noisy.png", "simulated ToF noise + flying pixels", point_size=0.2)
save_views([clean_pcd], "07_cleaned.png", "after jump-edge + outlier filtering", point_size=0.3)
show([pcd], "07 RGB-D point cloud")

# TRY IT:
#  - Change fx/fy by 10%. How does the room geometry distort?
#  - Fit the floor/table plane with segment_plane and measure its noise (rmse).
#  - Download a full TUM sequence (e.g. freiburg1_xyz) and chain frames with
#    o3d.pipelines.odometry.compute_rgbd_odometry - the depth-camera version of 06.
