#!/usr/bin/env python3

import rospy
import numpy as np
import math
from math import *
from geometry_msgs.msg import Twist, Point, Pose
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from src.respawnGoal import Respawn
import cv2
class Env():
    def __init__(self, action_size):
        self.init_x = -2.3 #m
        self.init_y = -1.0
        self.goal_x = -1.0   #| random goal
        self.goal_y = 0.0    #|
        # self.goal_x = 1.65  #| fixed goal
        # self.goal_y = 2.0   #|
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.prev_distance = 0
        # self.k_r = 2 
        # self.k_alpha = 15 
        # self.k_beta = -3
        self.const_vel = 0.3    #0.25
        # self.goal_dist_thres = 0.2  #0.55
        # self.goal_angle_thres = 15 #degrees
        self.current_theta = 0
        self.goal_counters = 0
        self.enable_feedback_control = False
        self.safe_dist = 1.0
        self.lidar = []
        self.position = Pose()
        self.self_rotation_z_speed=0
        self.linearx = 0
        self.lineary = 0
        
        
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        
    '''def OdometryCallBack(self, odometry):
        self.self_linear_x_speed = odometry.twist.twist.linear.x
        self.self_linear_y_speed = odometry.twist.twist.linear.y
        self.self_rotation_z_speed = odometry.twist.twist.angular.z'''
    '''def getGoalDistance(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance'''

    def getOdometry(self, odom):
        self.linearx = odom.twist.twist.linear.x
        self.lineary = odom.twist.twist.linear.y
        self.self_rotation_z = odom.twist.twist.angular.z
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, cur_theta = euler_from_quaternion(orientation_list)

        self.current_theta = cur_theta #radian
        
        return self.position.x, self.position.y, self.current_theta

    def getState(self, scan,image):
        done = False
        min_range = 0.3
        scan_range = []
        bridge = CvBridge()
        image=bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        image=np.nan_to_num(image, nan=0.0)
        
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            done = True

        image =image.astype(np.float32)
        
        
        return image , done

    def setReward(self, done, action):
        
        v = np.sqrt(self.linearx**2 + self.lineary**2)
        theta=self.self_rotation_z
        reward = v * np.cos(theta) * 0.2 - 0.01
        
        if done:
            rospy.loginfo("*****************")
            rospy.loginfo("* COLLISION !!! *")
            rospy.loginfo("*****************")
            reward = -1.
            self.pub_cmd_vel.publish(Twist())
        
        
        return reward, self.goal_counters
    

    def step(self, action):
        max_angular_vel = 0.75  #1.5 0.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = self.const_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        odom = None
        data1= None
        while data is None and data1 is None:
            try:
                data1 = rospy.wait_for_message('scan', LaserScan)
                data = rospy.wait_for_message('/kinect/depth/image_raw', Image)
                odom = rospy.wait_for_message('/odom', Odometry)
            except:
                pass
            
        state, done = self.getState(data1,data)
        
        # # Switching Algorithms:
        # if min(state[:20]) >= self.safe_dist:
        #     status = self.FeedBackControl(odom)
        #     self.enable_feedback_control = True
        # else:
        #     self.enable_feedback_control = False       
            
        reward, counters = self.setReward( done,action)

        return np.asarray(state), reward, done, counters

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        odom = None
        data1= None
        while data is None and data1 is None:
            try:
                data1 = rospy.wait_for_message('scan', LaserScan)
                data = rospy.wait_for_message('/kinect/depth/image_raw', Image)
                odom = rospy.wait_for_message('/odom', Odometry)
            except:
                pass

        if self.initGoal:
            
            self.initGoal = False

        self.init_x, self.init_y, self.current_theta = self.getOdometry(odom)
        
        
        state, done = self.getState(data1,data)
 

        return np.asarray(state)