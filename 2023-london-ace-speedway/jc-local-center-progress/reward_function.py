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
        elif distance_from_center >= 0.02 and distance_from_center <= 0.5:
            raceline_reward = 0.5 - distance_from_center
        
        progress_reward = progress / steps

        if progress == 100:
            final_reward = 100

        final_reward = 100 if progress == 100.0 else 0

        print("raceline_reward = " + format(raceline_reward, ".3f"))
        print("progress_reward = " + format(progress_reward, ".3f"))
        print("final reward = " + format(final_reward, ".3f"))

        reward = progress_reward + raceline_reward / 2 + final_reward

    return reward