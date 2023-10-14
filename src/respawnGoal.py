#!/usr/bin/env python3

import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

class Respawn():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('/home/tung/Downloads/dueling_dqn_gazebo/src',
                                                '/home/tung/Differential_Robot_Gazebo_Simulation/src/robot_simulations/robot_gazebo/models/turtlebot3_square/goal_box/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.goal_position = Pose()
        self.init_goal_x = -1.0  # random goal position
        self.init_goal_y = 0.0
        # self.init_goal_x = 1.65  # fixed goal position
        # self.init_goal_y = 2.0
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        self.obstacle_1 = 0.633707, 1.26704
        self.obstacle_2 = 0.938707, 0.967037
        self.obstacle_3 = 2.61794, -0.80051
        self.obstacle_4 = -0.291293, -1.00051
        self.obstacle_5 = -2.223, -2.49812
        self.obstacle_6 = 1.0, -2.51219
        self.obstacle_7 = 2.53, 2.53
        self.obstacle_8 = -1.42, 0.95
        self.obstacle_9 = 0.008707, 2.92449
        self.obstacle_10 = 2.93371, -0.000509
        self.obstacle_11 = -2.91629, -0.000509
        self.obstacle_12 = 0.008707, -2.92551
        self.random_goal = True
        # self.random_goal = False  # fixed goal
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False
        self.index = 0

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
                rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,
                              self.goal_position.position.y)
                break
            else:
                pass

    def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass

    def getPosition(self, position_check=False, delete=False):
        if delete:
            self.deleteModel()

        if self.random_goal:
            while position_check:
                goal_x = random.randrange(-24, 24) / 10.0
                goal_y = random.randrange(-24, 24) / 10.0
                if abs(goal_x - self.obstacle_1[0]) <= 0.85 and abs(goal_y - self.obstacle_1[1]) <= 0.85:
                    position_check = True
                elif abs(goal_x - self.obstacle_2[0]) <= 0.85 and abs(goal_y - self.obstacle_2[1]) <= 0.85:
                    position_check = True
                elif abs(goal_x - self.obstacle_3[0]) <= 0.85 and abs(goal_y - self.obstacle_3[1]) <= 0.85:
                    position_check = True
                elif abs(goal_x - self.obstacle_4[0]) <= 0.85 and abs(goal_y - self.obstacle_4[1]) <= 0.85:
                    position_check = True
                elif abs(goal_x - self.obstacle_5[0]) <= 0.85 and abs(goal_y - self.obstacle_5[1]) <= 0.85:
                    position_check = True
                elif abs(goal_x - self.obstacle_6[0]) <= 0.85 and abs(goal_y - self.obstacle_6[1]) <= 0.85:
                    position_check = True
                elif abs(goal_x - self.obstacle_7[0]) <= 0.85 and abs(goal_y - self.obstacle_4[1]) <= 0.85:
                    position_check = True
                elif abs(goal_x - self.obstacle_8[0]) <= 0.85 and abs(goal_y - self.obstacle_8[1]) <= 0.85:
                    position_check = True
                elif abs(goal_x - self.obstacle_9[0]) <= 0.85 and abs(goal_y - self.obstacle_9[1]) <= 0.85:
                    position_check = True
                elif abs(goal_x - self.obstacle_10[0]) <= 0.85 and abs(goal_y - self.obstacle_10[1]) <= 0.85:
                    position_check = True
                elif abs(goal_x - self.obstacle_11[0]) <= 0.85 and abs(goal_y - self.obstacle_11[1]) <= 0.85:
                    position_check = True
                elif abs(goal_x - self.obstacle_12[0]) <= 0.85 and abs(goal_y - self.obstacle_12[1]) <= 0.85:
                    position_check = True
                elif abs(goal_x + 2.3) <= 0.85 and abs(goal_y + 1.0 ) <= 0.85:
                    position_check = True
                else:
                    position_check = False

                if abs(goal_x - self.last_goal_x) < 1 and abs(goal_y - self.last_goal_y) < 1:
                    position_check = True

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y

        time.sleep(0.5)
        self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y