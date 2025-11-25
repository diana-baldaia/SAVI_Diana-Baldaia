#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
from functools import partial
import glob
from random import randint
import copy
from matplotlib import pyplot as plt
import numpy as np
import argparse
import open3d as o3d


view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 1.5787856578826904, 1.3218379700742449, 3.9939999580383301 ],
			"boundingbox_min" : [ -3.4785298648675282, -1.1435666402180991, -0.53179652584938775 ],
			"field_of_view" : 60.0,
			"front" : [ 0.78322856019974751, 0.1218461163517448, -0.60967741176578605 ],
			"lookat" : [ -0.96943976578983448, 0.13875294945645733, 1.6967955427681571 ],
			"up" : [ -0.13086180058148053, -0.92633358459353121, -0.35324393724828762 ],
			"zoom" : 0.69999999999999996
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}



def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_down

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, trans_init, threshold, criteria=None):
    # Pode alternar entre PointToPoint e PointToPlane aqui
    if criteria is None:
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)

    # Para PointToPlane, as normais são necessárias
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))

    reg_fine = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(), # Geralmente melhor que PointToPoint
        criteria)
    return reg_fine


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'],
                                    zoom=view['trajectory'][0]['zoom'])


def main():
    # ... (carregamento de imagens e criação de pcd1, pcd2) ...

    filename_rgb1 = 'tum_dataset/rgb/1.png'
    rgb1 = o3d.io.read_image(filename_rgb1)
    filename_depth1 = 'tum_dataset/depth/1.png'
    depth1 = o3d.io.read_image(filename_depth1)
    rgbd1 = o3d.geometry.RGBDImage.create_from_tum_format(rgb1, depth1)
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd1, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    filename_rgb2 = 'tum_dataset/rgb/2.png'
    rgb2 = o3d.io.read_image(filename_rgb2)
    filename_depth2 = 'tum_dataset/depth/2.png'
    depth2 = o3d.io.read_image(filename_depth2)
    rgbd2 = o3d.geometry.RGBDImage.create_from_tum_format(rgb2, depth2)
    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd2, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    # Definir parâmetros
    voxel_size_global = 0.05 # Para downsampling e calculo de features globais
    threshold_fine = 0.02    # Para o ICP fino

    # 1. Registo Global (Coarse Registration)
    print("Performing global registration...")
    source_down_global = preprocess_point_cloud(pcd1, voxel_size_global)
    target_down_global = preprocess_point_cloud(pcd2, voxel_size_global)

    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down_global, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_global * 5, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down_global, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_global * 5, max_nn=100))

    result_global = execute_global_registration(source_down_global, target_down_global,
                                                source_fpfh, target_fpfh,
                                                voxel_size_global)
    trans_init_for_icp = result_global.transformation
    print("Global registration result (initial transformation for ICP):", result_global)
    draw_registration_result(pcd1, pcd2, trans_init_for_icp)

    # Se o registo global falhar (fitness muito baixo ou inlier_rmse muito alto), pode ser preciso ajustar
    # os parâmetros do FPFH/RANSAC ou usar uma estratégia diferente.
    # Por exemplo, se result_global.fitness < 0.1, pode indicar que a transformação inicial não é boa.

    # 2. Registo Fino com ICP (Refine Registration)
    print("Performing fine registration with ICP (Point-to-Plane)...")
    reg_fine = refine_registration(pcd1, pcd2, trans_init_for_icp, threshold_fine)

    print("Fine registration result:", reg_fine)
    print("Final transformation is:")
    print(reg_fine.transformation)
    draw_registration_result(pcd1, pcd2, reg_fine.transformation)

if __name__ == '__main__':
    main()