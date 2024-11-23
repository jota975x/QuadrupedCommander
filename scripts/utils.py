import pybullet as p
import numpy as np
from scripts.lidar import lidar_sim
import random
import math

# HYPERPARAMETERS
STATE_DIM = 386
NUM_JOINTS = 12

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


def get_spawn(range_radius=50.0):
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