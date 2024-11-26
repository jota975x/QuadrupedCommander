import pybullet as p
import numpy as np
from scripts.lidar import lidar_sim
import random
import math

# HYPERPARAMETERS
STATE_DIM = 386
NUM_JOINTS = 12
DIST_FACTOR = 0.01
GOAL_REACHED_THRESHOLD = -0.2
COLLISION_FACTOR = -100
COLLISION_DETECT = 0.2

FALL_FACTOR = -100
HEIGH_FACTOR = -0.01
FALL_DETECT = 0.2

STABILITY_FACTOR_RP = 0.01
STABILITY_FACTOR_Y = 0.001



def get_state(robot, goal, state_dim: int = STATE_DIM, num_joints: int = NUM_JOINTS):
    
    state = np.zeros(shape=(state_dim))
    
    # goal
    state[0:2] = goal
    
    # base link (position + orientation + velocities)
    position, orientation = p.getBasePositionAndOrientation(robot)
    orientation = p.getEulerFromQuaternion(orientation)
    linear_velocity, angular_velocity = p.getBaseVelocity(robot)
    state[2:5] = position
    state[5:8] = orientation
    state[8:11] = linear_velocity
    state[11:14] = angular_velocity
    
    # get joint states
    joint_states = p.getJointStates(robot, range(num_joints))
    for i, joint_state in enumerate(joint_states):
        state[14+i] = joint_state[0]
        
    # lidar information
    lidar_data = lidar_sim(robot, num_joints)
    state[26:] = lidar_data
    
    return state


def get_spawn(range_radius=5.0):
    range = (-range_radius, range_radius)
    
    x = random.uniform(*range)
    y = random.uniform(*range)
    z_orientation = random.uniform(0, 2 * math.pi)
    
    return x, y, z_orientation

def get_goal(spawn_x, spawn_y, radius=10.0):
    
    angle = random.uniform(0, 2 * math.pi)
    distance = random.uniform(0.5, radius)
    
    goal_x = spawn_x + distance * math.cos(angle)
    goal_y = spawn_y + distance * math.sin(angle)
    
    return (goal_x, goal_y)

def get_reward_done(robot_id, goal, num_joints: int = NUM_JOINTS):
    reward = -5.0
    done = False
    
    # get robot position and orientation (RPY)
    position, orientation = p.getBasePositionAndOrientation(robot_id)
    orientation = p.getEulerFromQuaternion(orientation)
    
    # distance to goal
    dist_to_goal = np.sqrt((position[0] - goal[0])**2 + (position[1] - goal[1])**2) * DIST_FACTOR
    # check if goal is reached
    if dist_to_goal < GOAL_REACHED_THRESHOLD:
        print("a")
        dist_to_goal = 100
        done = True                
    reward += dist_to_goal
    
    # check for collision (based on lidar sensor data)
    lidar_data = lidar_sim(robot_id, num_joints)
    lidar_min = np.min(lidar_data)
    if lidar_min < COLLISION_DETECT:
        print(lidar_min)
        print("b")
        reward += COLLISION_FACTOR
        done = True
        
    # check for fall (based on base height)
    reward += position[2] * HEIGH_FACTOR
    if position[2] < FALL_DETECT:
        print("c")
        reward += FALL_FACTOR
        done = True
    
    # stability factor
    stability = np.sqrt((orientation[0]*STABILITY_FACTOR_RP)**2 + (orientation[1]*STABILITY_FACTOR_RP)**2 + (orientation[2]*STABILITY_FACTOR_Y)**2)
    reward -= stability
    
    return reward, done
    
    