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


class Env():
    def __init__(self, action_size):
        self.init_x = 0.0 #m
        self.init_y = 0.0
        self.goal_x = 0.0  #| fixed goal
        self.goal_y = 0.0   #|
        self.theta_goal = 0.0 
        self.heading = 0
        self.action_size = action_size
        self.num_scan_ranges = 20
        self.initGoal = True
        self.get_goal = False
        self.prev_distance = 0
        self.const_vel = 0.225    #0.25
        self.k_r = 2 
        self.k_alpha = 15 
        self.k_beta = -3
        self.goal_dist_thres = 0.2  #0.55
        self.goal_angle_thres = 15 #degrees
        self.current_theta = 0
        self.goal_counters = 0
        self.nearby_distance = 1
        self.safe_dist = 0.8
        self.lidar = []
        self.self_rotation_z_speed=0
        self.linearx = 0
        self.lineary = 0
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.current_distance =0


    def getGoalDistance(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        return goal_distance


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
    def get_pose_tung(self,X_GOAL,Y_GOAL,THETA_GOAL):
        return X_GOAL,Y_GOAL,THETA_GOAL
    

    def getState(self, scan,image,odom):
        scan_range = scan
        min_range = 0.35
        done = False
        toa_do=[]
        object_nearby = False
        near_goal = False
        scan_range=[]
        x,y,theta=self.getOdometry(odom)
        toa_do=[x,y,theta]
        bridge = CvBridge()
        image=bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        image=np.nan_to_num(image, nan=0.0)
        image = np.array(image, dtype=np.float32)
        image*=5
        
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        # Check object_nearbyness
        if min(scan_range) < self.nearby_distance:
            object_nearby = True
        
        # Check object collision
        if min_range > min(scan_range) > 0:
            
            done = True
        
        # Check goal is near | reached
      
        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        self.current_distance = current_distance
 
        if current_distance <= 0.5:
            near_goal = True
        return image, object_nearby, done, near_goal,toa_do


    def setReward(self, done, _):
        v = np.sqrt(self.linearx**2 + self.lineary**2)
        theta=self.self_rotation_z
        reward = v * np.cos(theta) * 0.2 - 0.01
        
        if done:
            rospy.loginfo("*****************")
            rospy.loginfo("* COLLISION !!! *")
            rospy.loginfo("*****************")
            reward = -1.
            self.pub_cmd_vel.publish(Twist())
        return reward


    def FeedBackControl(self, odom):
        x, y, theta = self.getOdometry(odom)
        
        if self.theta_goal >= pi:
            theta_goal_norm = self.theta_goal - 2 * pi
        else:
            theta_goal_norm = self.theta_goal
        
        ro = sqrt( pow( ( self.goal_x - x ) , 2 ) + pow( ( self.goal_y - y ) , 2) )
        lamda = atan2( self.goal_y - y , self.goal_x - x )
        
        alpha = (lamda -  theta + pi) % (2 * pi) - pi
        beta = (self.theta_goal - lamda + pi) % (2 * pi) - pi

        if ro < self.goal_dist_thres and degrees(abs(theta - theta_goal_norm)) < self.goal_angle_thres:
            rospy.loginfo("********************")
            rospy.loginfo("* GOAL REACHED !!! *")
            rospy.loginfo("********************")
            v = 0
            w = 0
            self.get_goal=True
            v_scal = 0
            w_scal = 0
            vel_cmd = Twist()
            vel_cmd.linear.x = v_scal
            vel_cmd.angular.z = w_scal
            self.pub_cmd_vel.publish(vel_cmd)
            self.goal_counters += 1
            if self.goal_counters > 5:
                self.goal_counters = 0
                
        else:
            v = self.k_r * ro
            w = self.k_alpha * alpha + self.k_beta * beta
            v_scal = v / abs(v) * self.const_vel
            w_scal = w / abs(v) * self.const_vel
            vel_cmd = Twist()
            vel_cmd.linear.x = v_scal
            vel_cmd.angular.z = w_scal
            self.pub_cmd_vel.publish(vel_cmd)
        

    def step(self, action, goal_x, goal_y, theta_goal):
        scan_data = None
        odom_data = None
        depth_image_data=None
        while scan_data is None:
            try:
                data_data = rospy.wait_for_message('scan', LaserScan)
                depth_image_data = rospy.wait_for_message('/kinect/depth/image_raw', Image)
                odom = rospy.wait_for_message('/odom', Odometry)
            except:
                pass
            
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.theta_goal = theta_goal
        state, object_nearby, done, near_goal,toa_do = self.getState(scan_data,depth_image_data,odom_data)
        reward=0
        if (not object_nearby) or near_goal:
            self.FeedBackControl(odom)
        
        else:
            max_angular_vel = 0.75  
            ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5
            vel_cmd = Twist()
            vel_cmd.linear.x = self.const_vel
            vel_cmd.angular.z = ang_vel
            self.pub_cmd_vel.publish(vel_cmd)   
            reward = self.setReward( done, action)
        return np.asarray(state), reward, done, self.goal_counters,toa_do


    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")
        data = None
        odom = None
        depth_image_data=None
        while data is None:
            try:
                
                data = rospy.wait_for_message('scan', LaserScan)
                depth_image_data = rospy.wait_for_message('/kinect/depth/image_raw', Image)
                odom = rospy.wait_for_message('/odom', Odometry)
            except:
                pass
        if self.initGoal:
            self.initGoal = False
        self.init_x, self.init_y, self.current_theta = self.getOdometry(odom)
        state, _, _, _,__ = self.getState(data,depth_image_data,odom)
        self.goal_counters = 0
        self.lidar = state
        return np.asarray(state)