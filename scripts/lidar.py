import pybullet as p
import math
import numpy as np

def lidar_sim(robot_id, lidar_idx, num_beams=360, range=5.0):
    # Get LiDAR position and orientation (quaternion) - relative to world
    lidar_pos, lidar_orientation = p.getLinkState(robot_id, lidar_idx)[:2]
    r_matrix = p.getMatrixFromQuaternion(lidar_orientation)
    
    lidar_data = []
    angle_range = np.linspace(0, 2*np.pi, num_beams)

    for angle in angle_range:
        ray_offset = np.array([math.cos(angle), math.sin(angle), 0])
        ray_direction = np.dot(np.array(r_matrix).reshape(3, 3), ray_offset)
        
        # compute ray ends and check for hits 
        ray_end = lidar_pos + range * ray_direction
        hit_result = p.rayTest(lidar_pos, ray_end)
        
        # extract information from result
        hit_distance = range    # default to max range if no hit
        if hit_result[0][0] != -1:
            hit_distance = hit_result[0][3][0]
            
        lidar_data.append(hit_distance)
        
    return np.array(lidar_data)
        