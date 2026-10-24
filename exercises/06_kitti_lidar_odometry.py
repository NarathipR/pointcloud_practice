"""Exercise 06: lidar odometry - estimate how the car moved, using only lidar.

Chain ICP between consecutive scans: T_world_k = T_world_(k-1) @ T_(k-1)_k.
Then compare against the GPS/IMU (OXTS) ground truth that ships with KITTI.
Small per-frame errors add up: that accumulated "drift" is why SLAM adds
loop closure and why robots fuse lidar with IMU / wheel odometry.

Data: run `python tools/download_kitti.py` first (30 frames = 3 seconds of driving).
Run:  python exercises/06_kitti_lidar_odometry.py [--show]
"""
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from common import KITTI_DIR, OUT_DIR, kitti_num_scans, load_kitti_scan, show

VOXEL = 0.3
reg = o3d.pipelines.registration


def prep(i):
    xyz, _ = load_kitti_scan(i)
    keep = np.linalg.norm(xyz[:, :2], axis=1) > 3  # drop hits on our own car
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[keep]))
    pcd = pcd.voxel_down_sample(VOXEL)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    return pcd


def oxts_positions(n):
    """GPS lat/lon/alt + yaw -> metric x, y in the first frame's heading."""
    rows = np.array([np.loadtxt(f"{KITTI_DIR}/oxts/data/{i:010d}.txt") for i in range(n)])
    lat, lon, alt, yaw = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 5]
    r_earth = 6378137.0
    scale = np.cos(np.radians(lat[0]))  # Mercator projection, as in the KITTI devkit
    x = scale * r_earth * np.radians(lon)
    y = scale * r_earth * np.log(np.tan(np.radians(90 + lat) / 2))
    xy = np.c_[x - x[0], y - y[0]]
    c, s = np.cos(-yaw[0]), np.sin(-yaw[0])  # rotate so the car starts facing +x
    xy = xy @ np.array([[c, -s], [s, c]]).T
    return np.c_[xy, alt - alt[0]]


n = kitti_num_scans()
print(f"{n} scans available")
poses = [np.eye(4)]
T_guess = np.eye(4)  # constant-velocity model: assume the same motion as last time
prev = prep(0)
for i in range(1, n):
    cur = prep(i)
    result = reg.registration_icp(cur, prev, 1.0, T_guess,
                                  reg.TransformationEstimationPointToPlane(),
                                  reg.ICPConvergenceCriteria(max_iteration=50))
    T_guess = result.transformation
    poses.append(poses[-1] @ result.transformation)
    step = np.linalg.norm(result.transformation[:3, 3])
    if i % 5 == 0 or i == 1:
        print(f"  {i - 1:3d}->{i:3d}: moved {step:.2f} m ({step * 10 * 3.6:.0f} km/h), "
              f"fitness {result.fitness:.2f}, rmse {result.inlier_rmse:.3f}")
    prev = cur

est = np.array([P[:3, 3] for P in poses])
gt = oxts_positions(n)
err = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
dist = np.sum(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1))
print(f"distance driven (GPS): {dist:.1f} m | lidar odometry: "
      f"{np.sum(np.linalg.norm(np.diff(est[:, :2], axis=0), axis=1)):.1f} m")
print(f"final position error: {err[-1]:.2f} m ({100 * err[-1] / max(dist, 1e-6):.1f}% of distance)")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(gt[:, 0], gt[:, 1], "k-", lw=2, label="GPS/IMU (OXTS)")
ax.plot(est[:, 0], est[:, 1], "r.--", label="lidar ICP odometry")
ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title=f"{n} frames of KITTI drive 0001")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/06_trajectory.png", dpi=110)
print("  saved output/06_trajectory.png")

# Stitch every 5th scan into one map using the estimated poses
world = o3d.geometry.PointCloud()
for i in range(0, n, 5):
    world += prep(i).transform(poses[i])
world = world.voxel_down_sample(VOXEL)
o3d.io.write_point_cloud(f"{OUT_DIR}/06_map.ply", world)
print(f"  saved output/06_map.ply ({len(world.points)} points)")
show([world], "06 stitched map")

# TRY IT:
#  - Set T_guess = np.eye(4) every frame (no motion model). Does ICP still work?
#  - Use point-to-point ICP or VOXEL = 1.0. How does drift change?
#  - Download all 108 frames (--frames 108) and rerun.
#  - Note: GPS is in the IMU frame and lidar is ~0.8 m away from it, so a
#    perfect match is not expected; KITTI ships calib files for the exact offset.
