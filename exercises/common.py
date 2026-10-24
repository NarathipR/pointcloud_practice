"""Small helpers shared by the exercises.

Everything saves PNG images to ./output so the scripts also work on machines
without a display (WSL, SSH, cloud). Pass --show to any exercise to open the
interactive Open3D viewer as well.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

SHOW = "--show" in sys.argv


def show(geoms, title="Open3D"):
    """Open the interactive viewer only when --show was passed."""
    if SHOW:
        o3d.visualization.draw_geometries(geoms, window_name=title)


def save_views(pcd_list, filename, title="", max_points=60000, point_size=0.5):
    """Save a top view (X-Y) and a side view (X-Z) of one or more point clouds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for pcd in pcd_list:
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors) if pcd.has_colors() else None
        if len(pts) > max_points:
            idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
            pts = pts[idx]
            cols = cols[idx] if cols is not None else None
        c = cols if cols is not None else "tab:blue"
        axes[0].scatter(pts[:, 0], pts[:, 1], s=point_size, c=c)
        axes[1].scatter(pts[:, 0], pts[:, 2], s=point_size, c=c)
    axes[0].set(title="top view (x-y)", xlabel="x", ylabel="y", aspect="equal")
    axes[1].set(title="side view (x-z)", xlabel="x", ylabel="z", aspect="equal")
    fig.suptitle(title)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  saved {os.path.relpath(path, ROOT)}")


def label_colors(labels):
    """Map integer cluster labels to RGB colors (noise = -1 becomes black)."""
    labels = np.asarray(labels)
    n = max(labels.max() + 1, 1)
    cmap = plt.get_cmap("tab20")
    colors = cmap((labels % 20) / 20)[:, :3]
    colors[labels < 0] = 0
    return colors, n


KITTI_DIR = os.path.join(DATA_DIR, "kitti", "2011_09_26_drive_0001")


def load_kitti_scan(i):
    """Return (N, 3) xyz and (N,) reflectance for KITTI Velodyne scan i.

    Each .bin file is a flat float32 array: x, y, z, reflectance per point.
    Frame: x forward, y left, z up, origin at the sensor (1.73 m above road).
    """
    path = os.path.join(KITTI_DIR, "velodyne_points", "data", f"{i:010d}.bin")
    if not os.path.exists(path):
        sys.exit(f"Missing {path}\nRun first:  python tools/download_kitti.py")
    scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return scan[:, :3].astype(np.float64), scan[:, 3]


def kitti_num_scans():
    d = os.path.join(KITTI_DIR, "velodyne_points", "data")
    return len(os.listdir(d)) if os.path.isdir(d) else 0
