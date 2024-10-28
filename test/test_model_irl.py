#!usr/bin/env python3
import csv
import rospy
import numpy as np
import math
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, Point, Pose
from testing_env import Env
from ..tool.agent import *


def TestModel():
    mode = "test"
    load_episodes = 840
    rospy.init_node('dqn_test_node_irl')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    NUMBER_TRIALS = 500
    TRIAL_LENGTH = 5
    state_size = 22
    action_size = 5
    
    # define the environment
    env = Env(action_size)

    # define the agent
    agent = DuelingQAgent(state_size, action_size, mode, load_episodes)
    rewards_per_trial, episodes, reward_list,list_toa_do = [], [], [],[]
    global_steps = 0
    total_dist=0
    for e in range(1, NUMBER_TRIALS):
        print("start")
        text = '\r\n' + '_____ TRIAL: ' + str(e) + ' _____' + '\r\n'
        text = text + '-----------------------------------------------------------\n'
        print(text)
        done = False
        state = env.reset()
        score = 0
        x_re=-2.3
        y_re=-1
        while not done:
            state = np.float32(state)
            
            # get action
            action = agent.getAction(state)
            X_GOAL, Y_GOAL, THETA_GOAL=0,0,0
                
            # take action and return next_state, reward and other status
            next_state, reward, done, counters,toa_do = env.step(action, X_GOAL, Y_GOAL, THETA_GOAL)
            next_state = np.float32(next_state)
            
            score += reward
            reward_list.append(reward)
            list_toa_do.append(toa_do)

            x_pre=toa_do[0]
            y_pre=toa_do[1]
            dist=math.sqrt((x_pre-x_re)**2+(y_pre-y_re)**2)
            total_dist=dist+total_dist
            
            x_re=x_pre
            y_re=y_pre

            # update state, publish actions
            state = next_state
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)
                
            # store states into log data
            # logger.store_data()
            
            # check if goal is reached
            if env.get_goal:
                env.pub_cmd_vel.publish(Twist())
                env.get_goal = False
                env.get_pose_tung(X_GOAL,Y_GOAL,THETA_GOAL) 
                if counters >= TRIAL_LENGTH:
                    print("Deployment Terminated Successfully!!!")
            if done:
                list_toa_do = []
                env.pub_cmd_vel.publish(Twist())
                result.data = [score, action]
                pub_result.publish(result)
                agent.updateTargetModel()
                rewards_per_trial.append(score)
                episodes.append(e)
                break    
                
            global_steps += 1

            

if __name__ == '__main__':
    try:
        TestModel()
    
    except rospy.ROSInterruptException:
        print("<--------- Test mode completed --------->")
        print('Deployment Break!')
        pass