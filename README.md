# Point cloud practice (lidar + depth cameras, no hardware needed)

Seven short Python exercises that take you from "what is a point cloud" to
lidar obstacle detection, lidar odometry and depth-camera processing, using
only free data that downloads automatically.

## Setup (once)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python tools/download_kitti.py    # ~60 MB: 30 KITTI lidar scans + GPS
```

Python 3.10 to 3.12 is the safest choice for Open3D wheels. On a headless Linux
box, Open3D also needs `libegl1` and `libgl1` (`sudo apt install libegl1 libgl1`).

Every exercise prints what it is doing and saves pictures to `output/`
(`sample_output/` shows what you should get).
Add `--show` to open the interactive 3D viewer (needs a display).
Each file ends with a **TRY IT** list: those small changes are where most of
the learning happens.

## Exercises

| # | File | You learn | Data |
|---|------|-----------|------|
| 01 | `01_load_and_inspect.py` | Point cloud = N x 3 array, Open3D <-> NumPy, bounds, save | Open3D sample (auto) |
| 02 | `02_downsample_normals.py` | Voxel downsampling, outlier removal, normals | Open3D sample (auto) |
| 03 | `03_plane_and_clusters.py` | RANSAC plane fitting, DBSCAN clustering, boxes | Open3D sample (auto) |
| 04 | `04_icp_registration.py` | ICP, FPFH features, global registration | Open3D demo pair (auto) |
| 05 | `05_kitti_obstacles.py` | Real lidar: ROI, ground removal, obstacle boxes, bird's-eye view | KITTI |
| 06 | `06_kitti_lidar_odometry.py` | Chained ICP odometry vs GPS ground truth, drift, map stitching | KITTI |
| 07 | `07_depth_to_pointcloud.py` | Pinhole back-projection, RGB-D, ToF noise and flying pixels | TUM RGB-D frame (auto) |

Run them all: `for f in exercises/0*.py; do python "$f"; done`

## Learning roadmap

**Week 1: fundamentals (exercises 01 to 04).** Get fluent with NumPy and
Open3D. Make sure you can explain voxel size, k-nearest neighbours, what a
normal is, how RANSAC picks a plane, and why ICP needs a good initial guess.

**Week 2: real lidar (05, 06).** Work through every TRY IT. Download all 108
frames (`--frames 108`) and other KITTI raw drives (same URL pattern on the
[KITTI raw page](https://www.cvlibs.net/datasets/kitti/raw_data.php)). Learn
the KITTI calibration files (lidar -> camera -> IMU transforms) and project
lidar points onto the camera image.

**Week 3: depth cameras (07).** Download a full TUM RGB-D sequence
([freiburg1_xyz](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download), ~470 MB)
and do RGB-D odometry with `o3d.pipelines.odometry`, then fuse frames into a
mesh with `o3d.pipelines.integration.ScalableTSDFVolume`. Compare against the
ground-truth trajectory with the TUM tools.

**Week 4+: robotics stack.**
- **ROS 2** (Humble/Jazzy): play public rosbags, view `sensor_msgs/PointCloud2`
  in RViz, write a small node that does exercise 05 on live messages.
- **Simulation:** Gazebo or CARLA give you simulated lidar and depth cameras
  with perfect ground truth, still free.
- **Bigger datasets:** nuScenes mini (~4 GB, free sign-up), KITTI odometry
  benchmark, Waymo Open, Argoverse 2.
- **Next algorithms:** KISS-ICP / LOAM-style lidar odometry, scan context loop
  closure, and learned 3D detection (PointPillars via OpenPCDet).

## Data licenses

- Open3D sample data: downloaded by `open3d.data`, MIT-licensed project.
- KITTI: CC BY-NC-SA 3.0, non-commercial use. Geiger et al., IJRR 2013.
- TUM RGB-D: CC BY 4.0. Sturm et al., IROS 2012.
