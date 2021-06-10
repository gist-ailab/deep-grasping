#!/usr/bin/env python

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
from deep_grasping_ros.srv import GetTarget6dofGrasp

panda_gripper_coords = {
    "left_center_indent": [0.041489, 0, 0.1034],
    "left_center_flat": [0.0399, 0, 0.1034],
    "right_center_indent": [-0.041489, 0, 0.1034],
    "right_center_flat": [-0.0399, 0, 0.1034],
    "left_tip_indent": [0.041489, 0, 0.112204],
    "left_tip_flat": [0.0399, 0, 0.112204],
    "right_tip_indent": [-0.041489, 0, 0.112204],
    "right_tip_flat": [-0.0399, 0, 0.112204],
}


class ContactGraspNet():

    def __init__(self):
        
        rospy.init_node("contact_graspnet")
        rospy.loginfo("Starting contact_graspnet node")

        # initialize dnn server 
        self.sock, add = su.initialize_server('localhost', 7777)
        
        self.camera_info = rospy.wait_for_message("/azure1/rgb/camera_info", CameraInfo)
        self.data = {}
        self.data["intrinsics_matrix"] = np.array(self.camera_info.K).reshape(3, 3)
        
        self.bridge = cv_bridge.CvBridge()
        # rgb_sub = message_filters.Subscriber("/azure1/rgb/image_raw", Image, buff_size=2048*1536*3)
        # depth_sub = message_filters.Subscriber("/azure1/depth_to_rgb/image_raw", Image, buff_size=2048*1536*3)
        # self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=1, slop=1)
        # self.ts.registerCallback(self.inference)

        self.grasp_srv = rospy.Service('/get_target_grasp_pose', GetTarget6dofGrasp, self.get_target_grasp_pose)


        # tf publisher
        self.br = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1.0))
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.marker_pub = rospy.Publisher("target_grasp", MarkerArray, queue_size = 1)

    def inference(self, rgb, depth):
        
        # convert ros imgmsg to opencv img
        # rgb = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, rgb.width, -1)
        # depth = np.frombuffer(depth.data, dtype=np.float32).reshape(depth.height, depth.width)
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth, desired_encoding='32FC1')


        self.data["rgb"] = rgb
        self.data["depth"] = depth 

        # send image to dnn client
        rospy.loginfo_once("Sending rgb, depth to Contact-GraspNet client")
        su.sendall_pickle(self.sock, self.data)
        pred_grasps_cam, scores, contact_pts = su.recvall_pickle(self.sock)
        if pred_grasps_cam is None:
            return
        for k in pred_grasps_cam.keys():
            if len(scores[k]) == 0:
                return
            best_grasp = pred_grasps_cam[k][np.argmax(scores[k])]
            if k == 1:
                print(k, pred_grasps_cam[k], scores[k], contact_pts[k])
                exit()
        
        # publish as tf
        while True:
            try:
                T_base_to_cam = self.tf_buffer.lookup_transform("world", "azure1_rgb_camera_link", rospy.Time(), rospy.Duration(5.0))
                T_base_to_cam = orh.msg_to_se3(T_base_to_cam)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                print(e)
                rospy.sleep(0.5)
            else:
                break
        T_cam_to_grasp = best_grasp
        T_base_to_grasp = np.matmul(T_base_to_cam, T_cam_to_grasp)
        t_target_grasp = orh.se3_to_transform_stamped(T_base_to_grasp, "panda_link0", "grasp_pose")
        self.br.sendTransform(t_target_grasp)
        
        self.publish_marker()

    def get_target_grasp_pose(self, msg):
        
        rgb = rospy.wait_for_message("/azure1/rgb/image_raw", Image)
        depth = rospy.wait_for_message("/azure1/depth_to_rgb/image_raw", Image)
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth, desired_encoding='32FC1')
        self.data["rgb"] = rgb
        self.data["depth"] = depth 

        # send image to dnn client
        rospy.loginfo_once("Sending rgb, depth to Contact-GraspNet client")
        su.sendall_pickle(self.sock, self.data)
        pred_grasps_cam, scores, contact_pts = su.recvall_pickle(self.sock)
        if pred_grasps_cam is None:
            return
        for k in pred_grasps_cam.keys():
            if len(scores[k]) == 0:
                return
            best_grasp = pred_grasps_cam[k][np.argmax(scores[k])]
            if k == 1:
                print(k, pred_grasps_cam[k], scores[k], contact_pts[k])
                exit()
        
        # publish as tf
        while True:
            try:
                T_base_to_cam = self.tf_buffer.lookup_transform("world", "azure1_rgb_camera_link", rospy.Time(), rospy.Duration(5.0))
                T_base_to_cam = orh.msg_to_se3(T_base_to_cam)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                print(e)
                rospy.sleep(0.5)
            else:
                break
        T_cam_to_grasp = best_grasp
        T_base_to_grasp = np.matmul(T_base_to_cam, T_cam_to_grasp)
        t_target_grasp = orh.se3_to_transform_stamped(T_base_to_grasp, "panda_link0", "target_grasp_pose")
        self.br.sendTransform(t_target_grasp)
        self.publish_marker("target_grasp_pose")
        return t_target_grasp

    def publish_marker(self, frame_id):

        # Delete all existing markers
        marker_array = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

        markers = []
        for k in panda_gripper_coords.keys():
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time()
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.ns = k
            marker.pose.position.x = panda_gripper_coords[k][0]
            marker.pose.position.y = panda_gripper_coords[k][1]
            marker.pose.position.z = panda_gripper_coords[k][2]
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.color.r = 1
            marker.color.g = 0
            marker.color.b = 0
            marker.color.a = 1.0
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            markers.append(marker)
        marker_array = MarkerArray()
        marker_array.markers = markers
        self.marker_pub.publish(marker_array)

if __name__ == '__main__':

    grasp_net = ContactGraspNet()
    rospy.spin()