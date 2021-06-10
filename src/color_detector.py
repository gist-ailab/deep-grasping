#!/usr/bin/env python

## sudo apt install ros-melodic-ros-numpy
## pip install open3d==0.9.0, pyrsistent=0.16
## pip install open3d-ros-helper


import rospy
import cv2, cv_bridge
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import Point
from open3d_ros_helper import open3d_ros_helper as orh
import copy
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from panda_control.srv import get_target_pos


class ColorDetector():

    def __init__(self):
        
        rospy.init_node("color_detector")
        rospy.loginfo("Starting color_detector node")
        
        
        self.cv_bridge = cv_bridge.CvBridge()
        self.result_pub = rospy.Publisher("/color_detect", Image, queue_size=1)
        # RED
        self.lower_bound = np.array([155, 25, 0]) # hsv
        self.upper_bound = np.array([179, 255, 255]) # hsv
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1.0))
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        # get K and H from camera to robot base
        camera_info = rospy.wait_for_message("/azure1/rgb/camera_info", CameraInfo)
        self.K = np.array(camera_info.K).reshape(3, 3)
        get_XYZ_to_pick_srv = rospy.Service('/get_XYZ_to_pick', get_target_pos, self.get_XYZ_to_pick)
        self.XYZ_to_pick_markers_pub = rospy.Publisher('/XYZ_to_Pick', MarkerArray, queue_size=1)


    def get_transform_cam_to_world(self):
        while True:
            try:
                transform_cam_to_world = self.tf_buffer.lookup_transform("world", "azure1_rgb_camera_link", rospy.Time(), rospy.Duration(5.0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                print(e)
                rospy.sleep(0.5)
            else:
                self.H_cam_to_world = orh.msg_to_se3(transform_cam_to_world)
                print("Got transform from camera to the robot base")
                break

    def get_XYZ_to_pick(self, msg):
        
        rospy.loginfo("Wating for /azure1/rgb/image_raw")
        # get rgb
        rgb_msg = rospy.wait_for_message("/azure1/rgb/image_raw", Image)
        rgb = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')        
        # get point cloud
        pc_msg = rospy.wait_for_message("/azure1/points2", PointCloud2)
        cloud_cam = orh.rospc_to_o3dpc(pc_msg, remove_nans=True) 

        self.get_transform_cam_to_world()


        # red region detection in 2D
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(rgb, self.lower_bound, self.upper_bound)
        ys, xs = np.where(mask)
        xy_to_pick = (int(np.median(xs)), int(np.median(ys)))
        rospy.loginfo("xy_to_pick: {}".format(xy_to_pick))
        
        # get corresponding 3D pointcloud and get the median position
        cloud_center = orh.crop_with_2dmask(copy.deepcopy(cloud_cam), np.array(mask, dtype=bool), self.K)
        cloud_center = cloud_center.transform(self.H_cam_to_world)
        cloud_center_npy = np.asarray(cloud_center.points)
        valid_mask = (np.nan_to_num(cloud_center_npy) != 0).any(axis=1)
        cloud_center_npy = cloud_center_npy[valid_mask]
        XYZ_to_pick = np.median(cloud_center_npy, axis=0)

        rospy.loginfo("XYZ_to_pick: {}".format(XYZ_to_pick))
        self.publish_markers(XYZ_to_pick)

        result = cv2.bitwise_and(rgb, rgb, mask=mask)
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
        result = cv2.circle(result, xy_to_pick, 50, (0, 255, 0), 10)

        # convert opencv img to ros imgmsg
        img_msg = self.cv_bridge.cv2_to_imgmsg(result, encoding='bgr8')
        
        # publish it as topic
        self.result_pub.publish(img_msg)
        rospy.loginfo_once("Published the result as topic. check /color_detect")
        
        pos = Point()
        pos.x = XYZ_to_pick[0] 
        pos.y = XYZ_to_pick[1] 
        pos.z = XYZ_to_pick[2] 

        return pos

    def publish_markers(self, XYZ):

        # Delete all existing markers
        markers = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        markers.markers.append(marker)
        self.XYZ_to_pick_markers_pub.publish(markers)

        # Object markers
        markers = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time()
        marker.action = Marker.ADD
        marker.pose.position.x = XYZ[0]
        marker.pose.position.y = XYZ[1]
        marker.pose.position.z = XYZ[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.color.r = 1
        marker.color.g = 0
        marker.color.b = 0
        marker.color.a = 0.5
        marker.ns = "XYZ_to_pick"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        markers.markers.append(marker)
        self.XYZ_to_pick_markers_pub.publish(markers)


if __name__ == '__main__':

    color_detector = ColorDetector()
    rospy.spin()