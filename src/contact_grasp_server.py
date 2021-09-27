#!/usr/bin/env python

from math import degrees
import rospy
import ros_numpy
import cv2, cv_bridge
import numpy as np
import message_filters

import tf2_ros
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from easy_tcp_python2_3 import socket_utils as su
from open3d_ros_helper import open3d_ros_helper as orh
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from deep_grasping_ros.msg import Grasp
from deep_grasping_ros.srv import GetTargetContactGraspSegm, GetTargetContactGraspSegmResponse
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib import cm


class ContactGraspNet():

    def __init__(self):
        
        rospy.init_node("contact_graspnet")
        rospy.loginfo("Starting contact_graspnet node")

        # initialize dnn server 
        self.grasp_sock, _ = su.initialize_server('localhost', 3333)
        rospy.loginfo("Wating for camera info")
        self.camera_info = rospy.wait_for_message("/azure1/rgb/camera_info", CameraInfo)
        self.data = {}
        self.data["intrinsics_matrix"] = np.array(self.camera_info.K).reshape(3, 3)
        rospy.loginfo("Successfully got camera info")

        self.bridge = cv_bridge.CvBridge()
        self.grasp_srv = rospy.Service('/get_target_grasp_pose', GetTargetContactGraspSegm, self.get_target_grasp_pose)


        # tf publisher
        self.br = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1.0))
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        # marker
        self.marker_pub = rospy.Publisher("target_grasp", MarkerArray, queue_size = 1)
        self.marker_id = 0
        self.cmap = plt.get_cmap("YlGn")
        control_points = np.load("/home/demo/catkin_ws/src/deep_grasping_ros/src/contact_graspnet/gripper_control_points/panda.npy")[:, :3]
        control_points = [[0, 0, 0], control_points[0, :], control_points[1, :], control_points[-2, :], control_points[-1, :]]
        control_points = np.asarray(control_points, dtype=np.float32)
        control_points[1:3, 2] = 0.0584
        control_points = np.tile(np.expand_dims(control_points, 0), [1, 1, 1]).squeeze()
        mid_point = 0.5*(control_points[1, :] + control_points[2, :])
        self.grasp_line_plot = np.array([np.zeros((3,)), mid_point, control_points[1], control_points[3], 
                                    control_points[1], control_points[2], control_points[4]])
        
        self.uoais_vm_pub = rospy.Publisher("uoais/visible_mask", Image, queue_size=1)
        self.uoais_am_pub = rospy.Publisher("uoais/amodal_mask", Image, queue_size=1)


    def get_target_grasp_pose(self, msg):
        
        while True:
            try:
                H_base_to_cam = self.tf_buffer.lookup_transform("panda_link0", "azure1_rgb_camera_link", rospy.Time(), rospy.Duration(5.0))
                H_base_to_hand = self.tf_buffer.lookup_transform("panda_link0", "panda_hand", rospy.Time(), rospy.Duration(5.0))
                H_cam_to_hand = self.tf_buffer.lookup_transform("azure1_rgb_camera_link", "panda_hand", rospy.Time(), rospy.Duration(5.0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                print(e)
                rospy.sleep(0.5)
            else:
                H_base_to_cam = orh.msg_to_se3(H_base_to_cam)
                H_base_to_hand = orh.msg_to_se3(H_base_to_hand)
                H_cam_to_hand = orh.msg_to_se3(H_cam_to_hand)
                break

        rgb = rospy.wait_for_message("/azure1/rgb/image_raw", Image)
        depth = rospy.wait_for_message("/azure1/depth_to_rgb/image_raw", Image)
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth, desired_encoding='32FC1')
        self.data["rgb"] = rgb
        self.data["depth"] = depth 
        self.initialize_marker()
        self.marker_id = 0


        # send image to dnn client
        rospy.loginfo_once("Sending rgb, depth to Contact-GraspNet client")
        su.sendall_pickle(self.grasp_sock, self.data)
        pred_grasps_cam, scores, contact_pts, segm_result = su.recvall_pickle(self.grasp_sock)
        segmap, amodal_vis, visible_vis, occlusions = \
            self.bridge.cv2_to_imgmsg(segm_result["segm"]), segm_result["amodal_vis"], segm_result["visible_vis"], segm_result["occlusions"].tolist()
        self.uoais_vm_pub.publish(self.bridge.cv2_to_imgmsg(visible_vis))
        self.uoais_am_pub.publish(self.bridge.cv2_to_imgmsg(amodal_vis))


        if pred_grasps_cam is None:
            rospy.logwarn_once("No grasps are detected")
            return None

        all_pts = []
        grasps = []
        for k in pred_grasps_cam.keys():
            if len(scores[k]) == 0:
                # rospy.logwarn_once("No grasps are detected")
                continue
            for i, pred_grasp_cam in enumerate(pred_grasps_cam[k]):
                H_cam_to_target = pred_grasp_cam
                
                ## visualize gripper point
                # calculate p_cam_to_gripper
                p_hand_to_gripper = self.grasp_line_plot
                p_hand_to_gripper[2:,0] = np.sign(p_hand_to_gripper[2:,0]) * 0.08/2
                p_gripper_to_cam = np.matmul(p_hand_to_gripper, H_cam_to_target[:3, :3].T)
                p_gripper_to_cam += np.expand_dims(H_cam_to_target[:3, 3], 0) 
                H_gripper_to_cam = np.concatenate((p_gripper_to_cam, np.ones([7, 1])), axis=1)
                # p_base_to_gripper
                p_base_to_gripper = np.matmul(H_base_to_cam, H_gripper_to_cam.T)[:3, :]
                all_pts.append(p_base_to_gripper) # [3, 15]
                self.publish_marker(p_base_to_gripper, "panda_link0", scores[k][i])

                H_base_to_target = np.matmul(H_base_to_cam, H_cam_to_target)
                H_base_to_target[:3, :3] = np.matmul(H_base_to_target[:3, :3], H_cam_to_hand[:3, :3])
                t_target_grasp = orh.se3_to_transform_stamped(H_base_to_target, "panda_link0", "target_grasp_pose")
                
                grasp = Grasp()
                grasp.id = "obj_{}_grasp_{}".format(int(k), i)
                grasp.score = scores[k][i]
                grasp.transform = t_target_grasp
                grasps.append(grasp)

        return GetTargetContactGraspSegmResponse(grasps, segmap, occlusions)

    def initialize_marker(self):
        # Delete all existing markers
        marker_array = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    def publish_marker(self, pts, frame_id, score):


        markers = []
        for i in range(pts.shape[1]):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time()
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.ns = str(self.marker_id)
            marker.id = self.marker_id 
            self.marker_id += 1
            marker.pose.position.x = pts[0][i]
            marker.pose.position.y = pts[1][i]
            marker.pose.position.z = pts[2][i]
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.color.r = self.cmap(score)[0]
            marker.color.g = self.cmap(score)[1]
            marker.color.b = self.cmap(score)[2]
            marker.color.a = self.cmap(score)[3]

            marker.type = Marker.SPHERE
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            markers.append(marker)

        point_pairs = [[0, 1], [1, 2], [2, 4],  [2, 5], [3, 4], [5, 6]]
        for p1, p2 in point_pairs:
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time()
            marker.type = marker.LINE_STRIP
            marker.action = marker.ADD
            marker.ns = str(self.marker_id)
            marker.id = self.marker_id 
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            self.marker_id += 1
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.color.r = self.cmap(score)[0]
            marker.color.g = self.cmap(score)[1]
            marker.color.b = self.cmap(score)[2]
            marker.color.a = self.cmap(score)[3]
            marker.points = []
            line_point_1 = Point()
            line_point_1.x = pts[0][p1]
            line_point_1.y = pts[1][p1]
            line_point_1.z = pts[2][p1]
            marker.points.append(line_point_1)
            line_point_2 = Point()
            line_point_2.x = pts[0][p2]
            line_point_2.y = pts[1][p2]
            line_point_2.z = pts[2][p2]
            marker.points.append(line_point_2)
            markers.append(marker)

        marker_array = MarkerArray()
        marker_array.markers = markers
        self.marker_pub.publish(marker_array)


if __name__ == '__main__':

    grasp_net = ContactGraspNet()
    rospy.spin()