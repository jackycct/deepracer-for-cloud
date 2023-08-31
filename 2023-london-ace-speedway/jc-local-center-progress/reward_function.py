from numpy import array
import math

def reward_function(params):
    # Read input parameters
    distance_from_center = params['distance_from_center']
    progress = params['progress']
    steps = params['steps']
    is_offtrack = params['is_offtrack']

    if is_offtrack:
        reward = 1e-3
    else:
        raceline_reward = 1e-3
        if distance_from_center >= 0.0 and distance_from_center <= 0.02:
            raceline_reward = 1.0
        elif distance_from_center >= 0.02 and distance_from_center <= 0.03:
            raceline_reward = 0.3
        elif distance_from_center >= 0.03 and distance_from_center <= 0.05:
            raceline_reward = 0.1
        
        progress_reward = progress / steps

        print("raceline_reward = " + format(raceline_reward, ".3f"))
        print("progress_reward = " + format(progress_reward, ".3f"))

        reward = progress_reward + raceline_reward ** 2

    return reward