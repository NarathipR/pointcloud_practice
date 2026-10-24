"""Exercise 04: registration - aligning two scans with ICP.

Registration = find the rigid transform (rotation + translation) that lines up
a "source" cloud with a "target" cloud. It is the core of lidar odometry,
mapping and SLAM.

  * ICP needs a decent initial guess and refines it (local).
  * FPFH features + RANSAC give that guess without any prior (global).

Data: Open3D DemoICPPointClouds (two overlapping scans, downloads automatically).
Run:  python exercises/04_icp_registration.py [--show]
"""
import copy

import numpy as np
import open3d as o3d

from common import save_views, show

o3d.utility.random.seed(0)  # make RANSAC repeatable
demo = o3d.data.DemoICPPointClouds()
source = o3d.io.read_point_cloud(demo.paths[0])
target = o3d.io.read_point_cloud(demo.paths[1])
VOXEL = 0.05


def colored_pair(src, tgt, T):
    s, t = copy.deepcopy(src), copy.deepcopy(tgt)
    s.paint_uniform_color([1, 0.7, 0])  # yellow = source
    t.paint_uniform_color([0, 0.65, 0.93])  # blue = target
    s.transform(T)
    return [s, t]


def prep(pcd):
    down = pcd.voxel_down_sample(VOXEL)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 5, max_nn=100))
    return down, fpfh


src_down, src_fpfh = prep(source)
tgt_down, tgt_fpfh = prep(target)
reg = o3d.pipelines.registration

# 1. Before alignment --------------------------------------------------------
eval0 = reg.evaluate_registration(src_down, tgt_down, VOXEL * 1.5, np.eye(4))
print(f"identity        : fitness={eval0.fitness:.3f} rmse={eval0.inlier_rmse:.4f}")
save_views(colored_pair(src_down, tgt_down, np.eye(4)), "04_before.png", "before alignment")

# 2. ICP from identity (bad initial guess) -----------------------------------
icp_bad = reg.registration_icp(src_down, tgt_down, VOXEL * 1.5, np.eye(4),
                               reg.TransformationEstimationPointToPoint())
print(f"ICP from identity: fitness={icp_bad.fitness:.3f} rmse={icp_bad.inlier_rmse:.4f}")

# 3. Global registration: match FPFH features with RANSAC --------------------
ransac = reg.registration_ransac_based_on_feature_matching(
    src_down, tgt_down, src_fpfh, tgt_fpfh, True, VOXEL * 1.5,
    reg.TransformationEstimationPointToPoint(False), 3,
    [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
     reg.CorrespondenceCheckerBasedOnDistance(VOXEL * 1.5)],
    reg.RANSACConvergenceCriteria(100000, 0.999))
print(f"RANSAC (global)  : fitness={ransac.fitness:.3f} rmse={ransac.inlier_rmse:.4f}")

# 4. Refine with point-to-plane ICP on the full-resolution clouds -------------
target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
icp = reg.registration_icp(source, target, VOXEL * 0.4, ransac.transformation,
                           reg.TransformationEstimationPointToPlane())
print(f"RANSAC + ICP     : fitness={icp.fitness:.3f} rmse={icp.inlier_rmse:.4f}")
np.set_printoptions(precision=3, suppress=True)
print("final transform:\n", icp.transformation)

save_views(colored_pair(src_down, tgt_down, icp.transformation), "04_after.png",
           "after RANSAC + point-to-plane ICP")
show(colored_pair(source, target, icp.transformation), "04 aligned")

# TRY IT:
#  - Compare TransformationEstimationPointToPoint vs PointToPlane in step 4
#    (fitness, rmse, iterations via icp_criteria=reg.ICPConvergenceCriteria(...)).
#  - Rotate the source by 90 degrees first. Does plain ICP still converge? RANSAC?
