import numpy as np
from numpy import array
from numpy.linalg import norm
import math

def distance_to_tangent_of_line(point, line_start_point, line_end_point):
    """Calculates the distance of point to the tangent of line_start_point and line_end_point."""
    x, y = point
    x1, y1 = line_start_point
    x2, y2 = line_end_point
    
    # Calculate the direction vector of the line
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the length of the direction vector
    length = math.sqrt(dx**2 + dy**2)

    # Calculate the normalized direction vector
    if length == 0:
        raise ValueError("The line is just a point")
    else:
        dx /= length
        dy /= length

    # Calculate the vector from (x1, y1) to (x, y)
    vx = x - x1
    vy = y - y1

    # Calculate the dot product of the two vectors
    dot_product = vx * dx + vy * dy

    # Calculate the projection of (x, y) onto the line
    projection_x = x1 + dot_product * dx
    projection_y = y1 + dot_product * dy

    # Calculate the distance from (x, y) to the projection point
    distance = math.sqrt((x - projection_x)**2 + (y - projection_y)**2)

    return distance

def find_most_far_away_visible_waypoint(current_x_y, closest_waypoint, waypoints, width=.5, debug=False):
    """Find the most far away visible waypoint without blocked by the track
    """
    results = closest_waypoint

    end_of_waypoints = round(len(waypoints) * 1.5)

    for pos in range(closest_waypoint, end_of_waypoints):
        end_waypoint = waypoints[pos % len(waypoints)]
        is_all_within_width = True
        for i in range((closest_waypoint + 1) % len(waypoints), pos):
            wp_in_middle = waypoints[i % len(waypoints)]
            is_within_width = (distance_to_tangent_of_line(wp_in_middle, np.asarray(current_x_y), end_waypoint) < width)
            is_all_within_width = is_all_within_width and is_within_width
        if is_all_within_width:
            results = pos % len(waypoints)
        else:
            break

    if debug:
        print("find_most_far_way_visible_waypoint for x,y", current_x_y, "closest_waypoint", closest_waypoint, "width", width, "is", results)

    return results

def get_degree_between_points(point1, point2, debug=False):
    """
    Get the degrees between two given points
    return 
        
    Args:
    point 1 - array of x, y
    point 2 - array of x, y

    Returns:
    180 (left) to -179 (right)
    """
    x1, y1 = point1
    x2, y2 = point2
    direction = math.atan2(y2 - y1, x2 - x1)
    
    # Convert to degree
    direction_degree = math.degrees(direction)
    if debug:
        print("point 1 " + format(x1, ".1f") + "," + format(y1, ".1f") 
              + " point 2 " + format(x2, ".1f") + "," + format(y2, ".1f") 
              + " degree " + format(direction_degree, ".3f"))
    return direction_degree

def distance_to_raceline(current_point, prev_point, next_point):
    """Calculates the distance of (x, y) to the tangent of (x1, y1) and (x2, y2)."""
    x, y = current_point
    x1, y1 = prev_point
    x2, y2 = next_point

    # Calculate the distance from (x, y) to the next point
    distance1 = math.sqrt((x - x1)**2 + (y - y1)**2)
    distance2 = math.sqrt((x - x2)**2 + (y - y2)**2)

    return min(distance1, distance2)

def calculate_reward_using_signmoid(x, power=10):
    """Returns the sigmoid value of x."""
    return ((1 - 1 / (1 + math.exp(-x))) * 2) ** power

class Reward:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.past_params = []
        self.one_step_count = 1

    def reward_function(self, params):
        # Max speed. Although the max speed is lower in corner, it will just give more bias to the racing line and progress
        MAX_SPEED = 3.5
        MAX_STEERING = 22

        # Read input parameters
        speed = params['speed']
        waypoints = params['waypoints']
        closest_waypoints = params['closest_waypoints']
        heading = params['heading']
        progress = params['progress']
        steps = params['steps']
        x = params['x']
        y = params['y']
        steering_angle = params['steering_angle']
        is_offtrack = params['is_offtrack']
        all_wheels_on_track = params['all_wheels_on_track']

        if is_offtrack:
            return 0.0001
        else:
            most_distance_wp = find_most_far_away_visible_waypoint([x, y], closest_waypoints[1], waypoints, .5)
            
            most_far_away_wp_degree = get_degree_between_points([x, y], waypoints[most_distance_wp])

            direction_diff = heading - most_far_away_wp_degree
            if direction_diff > 180:
                direction_diff = 360 - direction_diff
            elif heading < -180:
                direction_diff = 360 + direction_diff
            steering_diff = abs(direction_diff + steering_angle)
            print("most_far_away_wp_degree " + format(most_far_away_wp_degree, ".1f") + " heading "+ format(heading, ".1f"))
            print("steering_angle " + format(steering_angle, ".1f") + " direction_diff " + format(direction_diff, ".1f") + " steering_diff "+ format(steering_diff, ".1f"))

            STEERING_THRESHOLD = 10
            FULL_SPEED_WAYPOINT_THRESHOLD = 15
            if steering_diff < STEERING_THRESHOLD:
                heading_reward = 1
            else:
                heading_reward = (1 - abs(steering_diff - STEERING_THRESHOLD) / (180 - STEERING_THRESHOLD)) ** 5
            
            speed_reward = 1e-3
            if heading_reward > 0.8:
                if most_distance_wp - closest_waypoints[1] > FULL_SPEED_WAYPOINT_THRESHOLD: 
                    speed_reward = calculate_reward_using_signmoid(abs(speed - 4.0), 5)
                else:
                    speed_reward = (speed - 2.4) / (4.0 - 2.4)

            total_reward = (speed_reward + heading_reward) ** 2 + speed_reward * heading_reward 

            print("speed rewards = " + format(speed_reward, ".3f"))
            print("heading rewards = " + format(heading_reward, ".3f"))

            return total_reward
    
reward_object = Reward()  # add parameter verbose=True to get noisy output for testing

def reward_function(params):
    return reward_object.reward_function(params)
