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


def main():


    # ------------------------------------
    # Visualize the point cloud
    # ------------------------------------
    filename_rgb1 = 'tum_dataset/rgb/1.png'
    rgb1 = o3d.io.read_image(filename_rgb1)

    filename_depth1 = 'tum_dataset/depth/1.png'
    depth1 = o3d.io.read_image(filename_depth1)

    # Create the rgbd image
    rgbd1 = o3d.geometry.RGBDImage.create_from_tum_format(rgb1, depth1)
    print(rgbd1)

    filename_rgb2 = 'tum_dataset/rgb/2.png'
    rgb2 = o3d.io.read_image(filename_rgb2)

    filename_depth2 = 'tum_dataset/depth/2.png'
    depth2 = o3d.io.read_image(filename_depth2)

    # Create the rgbd image
    rgbd2 = o3d.geometry.RGBDImage.create_from_tum_format(rgb2, depth2)
    print(rgbd2)

    # Show the images using matplotlib
    # plt.subplot(1, 2, 1)
    # plt.title('TUM grayscale image')
    # plt.imshow(rgbd1.color)
    # plt.subplot(1, 2, 2)
    # plt.title('TUM depth image')
    # plt.imshow(rgbd1.depth)
    # plt.show()

    # Obtain the point cloud from the rgbd image
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd1, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd2, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    # Flip it, otherwise the pointcloud will be upside down
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd], zoom=0.35)




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


    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    # pcd1 and pcd2 are already open3d.geometry.PointCloud objects
    # created from the RGB-D images above. read_point_cloud expects a
    # filename (os.PathLike). Pass the PointClouds directly instead.
    source = pcd1
    target = pcd2
    threshold = 0.02
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                            [-0.139, 0.967, -0.215, 0.7],
                            [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    draw_registration_result(source, target, trans_init)

    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)


    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(source, target, reg_p2p.transformation)


    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(source, target, reg_p2p.transformation)

if __name__ == '__main__':
    main()
