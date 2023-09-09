from numpy import array
import math

def distance_to_tangent(points):
    """Calculates the distance of (x, y) to the tangent of (x1, y1) and (x2, y2)."""
    x, y = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]
    
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

        # Read input parameters
        speed = params['speed']
        is_offtrack = params['is_offtrack']
        if is_offtrack:
            return 0.0001
        else:
            return speed
    
reward_object = Reward()  # add parameter verbose=True to get noisy output for testing

def reward_function(params):
    return reward_object.reward_function(params)
