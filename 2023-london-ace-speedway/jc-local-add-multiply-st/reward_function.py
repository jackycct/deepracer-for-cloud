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

def calculate_reward_using_signmoid(x):
    """Returns the sigmoid value of x."""
    return ((1 - 1 / (1 + math.exp(-x))) * 2) ** 10

class Reward:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.past_params = []
        self.one_step_count = 1

    def reward_function(self, params):
        track_line = array([[-3.67571035e-01, -3.75680483e+00],
       [-2.53615499e-01, -3.81936789e+00],
       [-1.39659930e-01, -3.88193088e+00],
       [ 6.00407779e-02, -3.99131716e+00],
       [ 2.95187422e-01, -4.11981796e+00],
       [ 5.46101083e-01, -4.25655733e+00],
       [ 8.06283965e-01, -4.39751359e+00],
       [ 1.07023642e+00, -4.53919536e+00],
       [ 1.33604223e+00, -4.68004187e+00],
       [ 1.60316353e+00, -4.81921829e+00],
       [ 1.87160740e+00, -4.95615042e+00],
       [ 2.14155151e+00, -5.09032611e+00],
       [ 2.41320933e+00, -5.22122724e+00],
       [ 2.68677779e+00, -5.34830314e+00],
       [ 2.96240517e+00, -5.47094970e+00],
       [ 3.24016124e+00, -5.58848364e+00],
       [ 3.52000732e+00, -5.70011178e+00],
       [ 3.80182057e+00, -5.80464348e+00],
       [ 4.08519230e+00, -5.90115757e+00],
       [ 4.36962587e+00, -5.98837154e+00],
       [ 4.65448370e+00, -6.06484782e+00],
       [ 4.93901515e+00, -6.12903290e+00],
       [ 5.22239473e+00, -6.17933343e+00],
       [ 5.50378104e+00, -6.21430209e+00],
       [ 5.78243672e+00, -6.23309558e+00],
       [ 6.05788045e+00, -6.23587388e+00],
       [ 6.32934463e+00, -6.22128354e+00],
       [ 6.59554496e+00, -6.18647994e+00],
       [ 6.85474813e+00, -6.12828012e+00],
       [ 7.10379493e+00, -6.04173525e+00],
       [ 7.34257485e+00, -5.93149844e+00],
       [ 7.57043750e+00, -5.79984117e+00],
       [ 7.78583860e+00, -5.64741089e+00],
       [ 7.98769144e+00, -5.47590632e+00],
       [ 8.17430284e+00, -5.28637278e+00],
       [ 8.34412901e+00, -5.08037186e+00],
       [ 8.49510081e+00, -4.85936004e+00],
       [ 8.62566628e+00, -4.62556954e+00],
       [ 8.73451459e+00, -4.38149153e+00],
       [ 8.82063068e+00, -4.12975412e+00],
       [ 8.88340912e+00, -3.87302487e+00],
       [ 8.92279577e+00, -3.61388527e+00],
       [ 8.93950095e+00, -3.35466687e+00],
       [ 8.93556494e+00, -3.09720743e+00],
       [ 8.91135074e+00, -2.84319490e+00],
       [ 8.86716138e+00, -2.59432294e+00],
       [ 8.80259991e+00, -2.35253735e+00],
       [ 8.71664807e+00, -2.12027623e+00],
       [ 8.60709959e+00, -1.90114783e+00],
       [ 8.47251212e+00, -1.69909694e+00],
       [ 8.30893209e+00, -1.52114730e+00],
       [ 8.11070033e+00, -1.37893352e+00],
       [ 7.88624854e+00, -1.26894979e+00],
       [ 7.64095923e+00, -1.18871375e+00],
       [ 7.38323254e+00, -1.12953535e+00],
       [ 7.11746788e+00, -1.08695996e+00],
       [ 6.84194613e+00, -1.05690503e+00],
       [ 6.56183816e+00, -1.03701962e+00],
       [ 6.27225101e+00, -1.00631984e+00],
       [ 5.98151712e+00, -9.66221187e-01],
       [ 5.69080726e+00, -9.17389229e-01],
       [ 5.40080864e+00, -8.60064252e-01],
       [ 5.11210991e+00, -7.94223767e-01],
       [ 4.82526163e+00, -7.19751559e-01],
       [ 4.54087720e+00, -6.36238408e-01],
       [ 4.25949404e+00, -5.43552729e-01],
       [ 3.98164523e+00, -4.41503466e-01],
       [ 3.70805775e+00, -3.29459164e-01],
       [ 3.43929905e+00, -2.07255926e-01],
       [ 3.17579210e+00, -7.50278337e-02],
       [ 2.91779925e+00,  6.68332183e-02],
       [ 2.66541838e+00,  2.17720315e-01],
       [ 2.41857998e+00,  3.76845457e-01],
       [ 2.17702108e+00,  5.43243160e-01],
       [ 1.94015340e+00,  7.15655233e-01],
       [ 1.70742744e+00,  8.93018807e-01],
       [ 1.47838896e+00,  1.07447771e+00],
       [ 1.25262963e+00,  1.25930439e+00],
       [ 1.03014195e+00,  1.44718802e+00],
       [ 8.23800921e-01,  1.62695301e+00],
       [ 6.34600878e-01,  1.81535804e+00],
       [ 4.56182212e-01,  2.01681089e+00],
       [ 2.89994210e-01,  2.22744703e+00],
       [ 1.39871597e-01,  2.44566011e+00],
       [ 7.51650007e-03,  2.67030811e+00],
       [-1.05336301e-01,  2.90367007e+00],
       [-2.01081395e-01,  3.14755392e+00],
       [-3.05903528e-01,  3.40747657e+00],
       [-4.17680312e-01,  3.66717437e+00],
       [-5.39297081e-01,  3.92563722e+00],
       [-6.73329495e-01,  4.18089057e+00],
       [-8.20784488e-01,  4.43033099e+00],
       [-9.81433661e-01,  4.67145334e+00],
       [-1.15379575e+00,  4.90303021e+00],
       [-1.33719344e+00,  5.12393481e+00],
       [-1.53124435e+00,  5.33318248e+00],
       [-1.73573850e+00,  5.52985783e+00],
       [-1.95054689e+00,  5.71309041e+00],
       [-2.17552391e+00,  5.88208257e+00],
       [-2.41042288e+00,  6.03615384e+00],
       [-2.65484002e+00,  6.17477087e+00],
       [-2.90819030e+00,  6.29755050e+00],
       [-3.16970967e+00,  6.40423919e+00],
       [-3.43847502e+00,  6.49467968e+00],
       [-3.71343367e+00,  6.56877621e+00],
       [-3.99343599e+00,  6.62646622e+00],
       [-4.27726656e+00,  6.66770247e+00],
       [-4.56367063e+00,  6.69244569e+00],
       [-4.85137415e+00,  6.70066511e+00],
       [-5.13909637e+00,  6.69234297e+00],
       [-5.42555579e+00,  6.66747887e+00],
       [-5.70947035e+00,  6.62609141e+00],
       [-5.98955358e+00,  6.56821621e+00],
       [-6.26450816e+00,  6.49390183e+00],
       [-6.53301855e+00,  6.40320668e+00],
       [-6.79374382e+00,  6.29620075e+00],
       [-7.04531225e+00,  6.17297576e+00],
       [-7.28631926e+00,  6.03366560e+00],
       [-7.51533026e+00,  5.87847736e+00],
       [-7.73089013e+00,  5.70773042e+00],
       [-7.93154169e+00,  5.52190047e+00],
       [-8.11589628e+00,  5.32169053e+00],
       [-8.28304623e+00,  5.10830166e+00],
       [-8.43145826e+00,  4.88258873e+00],
       [-8.56027725e+00,  4.64602098e+00],
       [-8.66801011e+00,  4.39989745e+00],
       [-8.75370902e+00,  4.14598267e+00],
       [-8.81670411e+00,  3.88622805e+00],
       [-8.85665839e+00,  3.62265482e+00],
       [-8.87378887e+00,  3.35722730e+00],
       [-8.86965622e+00,  3.09165646e+00],
       [-8.84478836e+00,  2.82734414e+00],
       [-8.79890148e+00,  2.56568217e+00],
       [-8.73135464e+00,  2.30810976e+00],
       [-8.63845707e+00,  2.05701302e+00],
       [-8.51548433e+00,  1.81594078e+00],
       [-8.37107672e+00,  1.58335412e+00],
       [-8.20849509e+00,  1.35867880e+00],
       [-8.03025048e+00,  1.14140923e+00],
       [-7.83857692e+00,  9.31047234e-01],
       [-7.63556642e+00,  7.27060015e-01],
       [-7.42272466e+00,  5.29155286e-01],
       [-7.20144284e+00,  3.37043959e-01],
       [-6.97277641e+00,  1.50606163e-01],
       [-6.73808230e+00, -3.06410450e-02],
       [-6.49829975e+00, -2.07013884e-01],
       [-6.25422585e+00, -3.78850297e-01],
       [-6.00652967e+00, -5.46510171e-01],
       [-5.75577567e+00, -7.10378645e-01],
       [-5.50236975e+00, -8.70761040e-01],
       [-5.24672886e+00, -1.02807268e+00],
       [-4.98922263e+00, -1.18274413e+00],
       [-4.73017213e+00, -1.33519122e+00],
       [-4.46985852e+00, -1.48580256e+00],
       [-4.20852782e+00, -1.63492959e+00],
       [-3.94638525e+00, -1.78286841e+00],
       [-3.68362427e+00, -1.92989760e+00],
       [-3.42045482e+00, -2.07631970e+00],
       [-3.15709386e+00, -2.22245309e+00],
       [-2.89352554e+00, -2.36826686e+00],
       [-2.62982733e+00, -2.51387211e+00],
       [-2.36600094e+00, -2.65926072e+00],
       [-2.10205646e+00, -2.80443972e+00],
       [-1.83803701e+00, -2.94948411e+00],
       [-1.57396901e+00, -3.09446812e+00],
       [-1.30989802e+00, -3.23945189e+00],
       [-1.04582703e+00, -3.38443303e+00],
       [-7.81755507e-01, -3.52941203e+00],
       [-5.17685473e-01, -3.67439008e+00],
       [-3.67571035e-01, -3.75680483e+00]]) 
      
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

        if is_offtrack:
            return 0.0001
        else:
            smooth_reward = 1e-3
            print("Steps : " + format(steps, "0.3f"))
            if steps == 1:
                self.past_params = []
                self.one_step_count = 1
                prev_speed = 0
                print("smooth_reward : " + format(smooth_reward, "0.1f"))
            else:
                last_idx = len(self.past_params) - 1
                if last_idx >= 0:
                    prev_steering_angle = self.past_params[last_idx]['steering_angle']
                    prev_speed = self.past_params[last_idx]['speed']
                    if prev_steering_angle == steering_angle and prev_speed == speed:
                        smooth_reward = 1

                    print("smooth_reward {}, previous {},{}, current {},{}".format(smooth_reward, prev_steering_angle, prev_speed, steering_angle, speed))

            # adding past params for calculating rewards
            self.past_params.append(params)

            # Calculate the deviation from raceline    
            next_waypoint = waypoints[closest_waypoints[1]]
            prev_waypoint = waypoints[closest_waypoints[0]]
            next_point = track_line[closest_waypoints[1]]
            prev_point = track_line[closest_waypoints[0]]            
            distance = distance_to_raceline([x,y], prev_point, next_point)
            #distance = distance_to_tangent([[x,y], next_point, prev_point])
            print("wp " + format(prev_waypoint[0], "0.1f") + "," + format(prev_waypoint[1], "0.1f") + "  " + format(next_waypoint[0], "0.1f") + "," + format(next_waypoint[1], "0.1f"))
            print("rl " + format(prev_point[0], "0.1f") + "," + format(prev_point[1], "0.1f") + "  " + format(next_point[0], "0.1f") + "," + format(next_point[1], "0.1f") + "  " + format(x, "0.1f") + "," + format(y, "0.1f"))
            raceline_reward = calculate_reward_using_signmoid(distance)

            speed_reward = speed / MAX_SPEED

            heading_reward = 0.0001
            # Heading reward, if it is suppose to turn left but turning right, then penalty
            WP_LOOKAHEAD = 2
            lookahead = round(WP_LOOKAHEAD * (prev_speed + speed))
            next_wp = closest_waypoints[1] + lookahead % len(waypoints)
            further_point = track_line[next_wp]
            next_x, next_y = further_point
            prev_x, prev_y = prev_point
            wp_direction = math.atan2(next_y - prev_y, next_x - prev_x) 
            print("next_x " + format(next_x, ".3f") + " next y " + format(next_y, ".3f") + " trackline_direction " + format(wp_direction, ".3f"))
            # Convert to degree
            wp_direction_deg = math.degrees(wp_direction)
            # Calculate the difference between the track direction and the heading direction of the car
            direction_diff = wp_direction_deg - heading
            steering_diff = abs(steering_angle - direction_diff)
            if steering_diff < 10:
                heading_reward = 1
            else:
                heading_reward = max(1 - abs(steering_diff) / (MAX_STEERING+1), 1e-3)

            if closest_waypoints[0] > 135 or closest_waypoints[1] < 10:
                if distance < 0.2:
                    if abs(direction_diff) < 10 and abs(steering_angle) <= 11:
                        heading_reward *= 1.2
                    if abs(direction_diff) < 5 and abs(steering_angle) == 0:
                        heading_reward *= 1.2
                if speed < 3.4:
                    speed_reward *= 0.01

            if progress > self.one_step_count * 10:
                progress_reward = progress * 250 / steps
                self.one_step_count += 1
                print("progress rewards = " + format(progress_reward, ".3f"))
            else:
                progress_reward = 0

            # adjust weighting
            raceline_reward *= 2

            if raceline_reward < 1:
                heading_reward = 1e-3
            if heading_reward < 0.5:
                speed_reward = 1e-3
            if speed_reward < 0.5:
                smooth_reward = 1e-3

            total_reward = progress_reward \
                            + (raceline_reward + speed_reward + heading_reward + smooth_reward) ** 2 \
                            + raceline_reward * speed_reward * heading_reward * smooth_reward 

            if steps > 280:
                total_reward /= (steps - 280)

            print("raceline rewards = " + format(raceline_reward, ".3f") + " for distance = " + format(distance, "0.3f"))
            print("speed rewards = " + format(speed_reward, ".3f"))
            print("heading rewards = " + format(heading_reward, ".3f"))
            print("smooth rewards = " + format(smooth_reward, ".3f"))

            return total_reward
    
reward_object = Reward()  # add parameter verbose=True to get noisy output for testing

def reward_function(params):
    return reward_object.reward_function(params)
