#!/usr/bin/env python

import rospy
import ros_numpy
import cv2, cv_bridge
import numpy as np
import message_filters

from sensor_msgs.msg import Image, CameraInfo, PointCloud2

from easy_tcp_python2_3 import socket_utils as su
import PIL.Image  # @UnresolvedImport
import numpy


class ContactGraspNet():

    def __init__(self):
        
        rospy.init_node("contact_graspnet")
        rospy.loginfo("Starting contact_graspnet node")

        # initialize dnn server 
        self.sock, add = su.initialize_server('localhost', 7777)
        
        self.camera_info = rospy.wait_for_message("/azure1/rgb/camera_info", CameraInfo)
        self.data = {}
        self.data["intrinsics_matrix"] = np.array(self.camera_info.K).reshape(3, 3)
        
        rgb_sub = message_filters.Subscriber("/azure1/rgb/image_raw", Image, buff_size=2048*1536*3)
        depth_sub = message_filters.Subscriber("/azure1/depth_to_rgb/image_raw", Image, buff_size=2048*1536*3)
        point_sub = message_filters.Subscriber("/azure1/points2", PointCloud2, buff_size=2048*1536*3)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, point_sub], queue_size=1, slop=1)
        self.ts.registerCallback(self.inference)


    def inference(self, rgb, depth, pcl_msg):
        
        # convert ros imgmsg to opencv img
        rgb = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, rgb.width, -1)

        depth = np.frombuffer(depth.data, dtype=np.float32).reshape(depth.height, depth.width)
        # depth_img_raw = cv2.imdecode(np.fromstring(depth.data, np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
        # depth = self.cv_bridge.imgmsg_to_cv2(depth, desired_encoding='32FC1')

        self.data["rgb"] = rgb
        self.data["depth"] = depth 

        # send image to dnn client
        rospy.loginfo_once("Sending rgb, depth, pcl to 6 dof graspnet client")
        su.sendall_pickle(self.sock, self.data)
        pred_grasps_cam = su.recvall_pickle(self.sock)


    
if __name__ == '__main__':

    grasp_net = ContactGraspNet()
    rospy.spin()