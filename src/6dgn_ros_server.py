#!/usr/bin/env python

import rospy
import ros_numpy
import cv2, cv_bridge
import numpy as np
import message_filters

from sensor_msgs.msg import Image, CameraInfo, PointCloud2

from easy_tcp_python2_3 import socket_utils as su
from open3d_ros_helper import open3d_ros_helper as orh
import open3d

class GraspNet():

    def __init__(self):
        
        rospy.init_node("graspnet")
        rospy.loginfo("Starting 6dof_graspnet node")
        self.cv_bridge = cv_bridge.CvBridge()

        # initialize dnn server 
        self.sock, add = su.initialize_server('localhost', 7777)
        
        self.camera_info = rospy.wait_for_message("/azure1/rgb/camera_info", CameraInfo)
        self.data = {}
        self.data["intrinsics_matrix"] = np.array(self.camera_info.K).reshape(3, 3)
        
        rgb_sub = message_filters.Subscriber("/azure1/rgb/image_raw", Image, buff_size=2048*1536*3)
        depth_sub = message_filters.Subscriber("/azure1/depth_to_rgb/image_raw", Image, buff_size=2048*1536)
        point_sub = message_filters.Subscriber("/azure1/points2", PointCloud2, buff_size=2048*1536*3)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, point_sub], queue_size=1, slop=1)
        self.ts.registerCallback(self.inference)



        self.result_pub = rospy.Publisher("/classification_result", Image, queue_size=1)

    def inference(self, rgb, depth, pcl_msg):
        
        # convert ros imgmsg to opencv img
        rgb = self.cv_bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        depth = self.cv_bridge.imgmsg_to_cv2(depth, desired_encoding='32FC1')
        

        o3dpc = orh.rospc_to_o3dpc(pcl_msg, remove_nans=True) 
        o3dpc = o3dpc.voxel_down_sample(0.001)
        o3dpc_ds = orh.apply_pass_through_filter(o3dpc, [-0.3, 0.3], [-0.3, 0.3], [0, 1.0])
        plane_model, inliers = o3dpc_ds.segment_plane(distance_threshold=0.01,
                                         ransac_n=10,
                                         num_iterations=100)
        inlier_cloud = o3dpc_ds.select_down_sample(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = o3dpc_ds.select_down_sample(inliers, invert=True)
        outlier_cloud.paint_uniform_color([0, 1, 0])
        idx = np.random.choice(np.asarray(outlier_cloud.points).shape[0], 10000, replace=False)
        outlier_cloud = outlier_cloud.select_down_sample(idx, invert=True)
        open3d.visualization.draw_geometries([outlier_cloud])
        rospy.loginfo_once("Visualizing tabletop segmentation")
        cloud_npy = np.asarray(outlier_cloud.points)
        # cloud_npy = cloud_npy[idx, :]

        self.data["image"] = rgb
        self.data["depth"] = depth 
        self.data["smoothed_object_pc"] = cloud_npy

        # send image to dnn client
        rospy.loginfo_once("Sending rgb, depth, pcl to 6 dof graspnet client")
        su.sendall_pickle(self.sock, self.data)
        grasp = su.recvall_pickle(self.sock)
        print(grasp)

    
if __name__ == '__main__':

    grasp_net = GraspNet()
    rospy.spin()