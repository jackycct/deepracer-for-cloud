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
        track_line = array([[-1.10870266e-01, -3.28923649e+00],
       [ 3.08530033e-03, -3.35179949e+00],
       [ 1.17040878e-01, -3.41436247e+00],
       [ 2.67155035e-01, -3.49677694e+00],
       [ 5.31224236e-01, -3.64175510e+00],
       [ 7.95292050e-01, -3.78673708e+00],
       [ 1.05935791e+00, -3.93172204e+00],
       [ 1.32342249e+00, -4.07670999e+00],
       [ 1.58749253e+00, -4.22168684e+00],
       [ 1.85157150e+00, -4.36664498e+00],
       [ 2.11565953e+00, -4.51158404e+00],
       [ 2.37975252e+00, -4.65651345e+00],
       [ 2.64380300e+00, -4.80153155e+00],
       [ 2.90779948e+00, -4.94666100e+00],
       [ 3.17183395e+00, -5.09171176e+00],
       [ 3.43579563e+00, -5.23691449e+00],
       [ 3.69967445e+00, -5.38228999e+00],
       [ 3.96343526e+00, -5.52791161e+00],
       [ 4.22739272e+00, -5.67245588e+00],
       [ 4.49234180e+00, -5.81264262e+00],
       [ 4.75895548e+00, -5.94467301e+00],
       [ 5.02749127e+00, -6.06527802e+00],
       [ 5.29792136e+00, -6.17097540e+00],
       [ 5.56984651e+00, -6.25832383e+00],
       [ 5.84246897e+00, -6.32391190e+00],
       [ 6.11457879e+00, -6.36453755e+00],
       [ 6.38461859e+00, -6.37774032e+00],
       [ 6.65061874e+00, -6.36061208e+00],
       [ 6.91041521e+00, -6.31158817e+00],
       [ 7.16190640e+00, -6.23045952e+00],
       [ 7.40336134e+00, -6.11840506e+00],
       [ 7.63371915e+00, -5.97792716e+00],
       [ 7.85273454e+00, -5.81253590e+00],
       [ 8.06073257e+00, -5.62594341e+00],
       [ 8.25795798e+00, -5.42094103e+00],
       [ 8.44436155e+00, -5.19937675e+00],
       [ 8.61931513e+00, -4.96211838e+00],
       [ 8.78107932e+00, -4.70933382e+00],
       [ 8.92582601e+00, -4.44268382e+00],
       [ 9.04705597e+00, -4.16855913e+00],
       [ 9.13978354e+00, -3.89160008e+00],
       [ 9.20064134e+00, -3.61473574e+00],
       [ 9.22754383e+00, -3.34027936e+00],
       [ 9.21838069e+00, -3.07050136e+00],
       [ 9.17198186e+00, -2.80768898e+00],
       [ 9.08966521e+00, -2.55377830e+00],
       [ 8.97533196e+00, -2.31003499e+00],
       [ 8.83283641e+00, -2.07765390e+00],
       [ 8.66473025e+00, -1.85855605e+00],
       [ 8.47261841e+00, -1.65647338e+00],
       [ 8.26033878e+00, -1.47740602e+00],
       [ 8.07913017e+00, -1.35819101e+00],
       [ 7.87196922e+00, -1.26239598e+00],
       [ 7.63838816e+00, -1.18789995e+00],
       [ 7.38414001e+00, -1.12982202e+00],
       [ 7.11011748e+00, -1.05219283e+00],
       [ 6.83451060e+00, -9.65199629e-01],
       [ 6.55170114e+00, -8.72427042e-01],
       [ 6.26338478e+00, -7.80404231e-01],
       [ 5.97274285e+00, -6.89273812e-01],
       [ 5.68122116e+00, -5.98670415e-01],
       [ 5.39318851e+00, -5.09408189e-01],
       [ 5.10597011e+00, -4.20103415e-01],
       [ 4.81960973e+00, -3.30761962e-01],
       [ 4.53397931e+00, -2.41395491e-01],
       [ 4.24924767e+00, -1.52005936e-01],
       [ 3.98267460e+00, -6.78351831e-02],
       [ 3.71787596e+00,  1.27055803e-02],
       [ 3.46107411e+00,  8.48529786e-02],
       [ 3.20762706e+00,  1.72465697e-01],
       [ 2.95438099e+00,  2.79054195e-01],
       [ 2.69609189e+00,  3.98223490e-01],
       [ 2.44365001e+00,  5.23248494e-01],
       [ 2.19247890e+00,  6.62405908e-01],
       [ 1.94371402e+00,  8.02308679e-01],
       [ 1.70519304e+00,  9.53636706e-01],
       [ 1.46866202e+00,  1.10957801e+00],
       [ 1.24636400e+00,  1.27218103e+00],
       [ 1.03014195e+00,  1.44718802e+00],
       [ 8.23800921e-01,  1.62695301e+00],
       [ 6.34600878e-01,  1.81535804e+00],
       [ 4.56182212e-01,  2.01681089e+00],
       [ 2.89994210e-01,  2.22744703e+00],
       [ 1.39871597e-01,  2.44566011e+00],
       [ 7.51650007e-03,  2.67030811e+00],
       [-1.05336301e-01,  2.90367007e+00],
       [-2.01081395e-01,  3.14755392e+00],
       [-2.80062199e-01,  3.39966297e+00],
       [-3.70366805e-01,  3.66139442e+00],
       [-4.71667655e-01,  3.92062467e+00],
       [-5.87089495e-01,  4.17537595e+00],
       [-7.19318942e-01,  4.42320896e+00],
       [-8.71154297e-01,  4.66094042e+00],
       [-1.04344722e+00,  4.88580278e+00],
       [-1.23566473e+00,  5.09573389e+00],
       [-1.44608234e+00,  5.28970994e+00],
       [-1.67178942e+00,  5.46818334e+00],
       [-1.90982135e+00,  5.63225514e+00],
       [-2.15839694e+00,  5.78187857e+00],
       [-2.41562856e+00,  5.91692315e+00],
       [-2.67977309e+00,  6.03571701e+00],
       [-2.88996291e+00,  6.11553621e+00],
       [-3.09752202e+00,  6.15597105e+00],
       [-3.33751297e+00,  6.16032505e+00],
       [-3.62611890e+00,  6.14956522e+00],
       [-3.93037009e+00,  6.14272022e+00],
       [-4.23101091e+00,  6.13526201e+00],
       [-4.53225708e+00,  6.12790823e+00],
       [-4.83340120e+00,  6.12053680e+00],
       [-5.13456917e+00,  6.11316681e+00],
       [-5.43569183e+00,  6.10580683e+00],
       [-5.73706579e+00,  6.09838915e+00],
       [-6.03696823e+00,  6.09130716e+00],
       [-6.34545422e+00,  6.08230209e+00],
       [-6.61645889e+00,  6.08493519e+00],
       [-6.85472488e+00,  6.04875183e+00],
       [-7.10467911e+00,  5.99326086e+00],
       [-7.32566881e+00,  5.92254114e+00],
       [-7.48515701e+00,  5.84475279e+00],
       [-7.73304845e+00,  5.70563778e+00],
       [-7.97730612e+00,  5.54507266e+00],
       [-8.21261986e+00,  5.36523311e+00],
       [-8.43426629e+00,  5.16728229e+00],
       [-8.63743376e+00,  4.95238398e+00],
       [-8.81747200e+00,  4.72218111e+00],
       [-8.97065357e+00,  4.47903561e+00],
       [-9.09285015e+00,  4.22531622e+00],
       [-9.18025314e+00,  3.96389458e+00],
       [-9.22982472e+00,  3.69810370e+00],
       [-9.24018184e+00,  3.43156827e+00],
       [-9.21214252e+00,  3.16779559e+00],
       [-9.14931202e+00,  2.90959824e+00],
       [-9.05507713e+00,  2.65936638e+00],
       [-8.93213666e+00,  2.41960672e+00],
       [-8.78332316e+00,  2.19292955e+00],
       [-8.60931637e+00,  1.98457994e+00],
       [-8.41394234e+00,  1.80264294e+00],
       [-8.19192337e+00,  1.65139423e+00],
       [-7.96107562e+00,  1.48073870e+00],
       [-7.73096064e+00,  1.30207097e+00],
       [-7.49883894e+00,  1.11866499e+00],
       [-7.26132717e+00,  9.34881544e-01],
       [-7.02040761e+00,  7.52792123e-01],
       [-6.77683659e+00,  5.73401737e-01],
       [-6.53082089e+00,  3.97218077e-01],
       [-6.28235781e+00,  2.24635510e-01],
       [-6.03141965e+00,  5.59688429e-02],
       [-5.77804872e+00, -1.08636517e-01],
       [-5.52240073e+00, -2.69280513e-01],
       [-5.26472860e+00, -4.26290787e-01],
       [-5.00534105e+00, -5.80149110e-01],
       [-4.74456144e+00, -7.31406478e-01],
       [-4.48269483e+00, -8.80608842e-01],
       [-4.22001428e+00, -1.02825982e+00],
       [-3.95678533e+00, -1.17486003e+00],
       [-3.69322285e+00, -1.32081901e+00],
       [-3.42949808e+00, -1.46646525e+00],
       [-3.16563596e+00, -1.61184614e+00],
       [-2.90166623e+00, -1.75701819e+00],
       [-2.63762701e+00, -1.90205455e+00],
       [-2.37354350e+00, -2.04700506e+00],
       [-2.10946548e+00, -2.19196647e+00],
       [-1.84539294e+00, -2.33693850e+00],
       [-1.58132648e+00, -2.48192108e+00],
       [-1.31725949e+00, -2.62690461e+00],
       [-1.05319187e+00, -2.77188647e+00],
       [-7.89123505e-01, -2.91686606e+00],
       [-5.25053859e-01, -3.06184399e+00],
       [-2.60984318e-01, -3.20682204e+00],
       [-1.10870266e-01, -3.28923649e+00]])
      
        BENCHMARK_STEPS = [35, 63, 89, 119, 147, 175, 202, 226, 254, 283]
        #BENCHMARK_STEPS = [33, 56, 80, 105, 128, 153, 177, 202, 227, 252]

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
            smooth_reward = 1e-3
            print("Steps : " + format(steps, "0.3f"))
            prev_speed = 0
            if steps == 1:
                self.past_params = []
                self.one_step_count = 1
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
            #print("wp " + format(prev_waypoint[0], "0.1f") + "," + format(prev_waypoint[1], "0.1f") + "  " + format(next_waypoint[0], "0.1f") + "," + format(next_waypoint[1], "0.1f"))
            #print("rl " + format(prev_point[0], "0.1f") + "," + format(prev_point[1], "0.1f") + "  " + format(next_point[0], "0.1f") + "," + format(next_point[1], "0.1f") + "  " + format(x, "0.1f") + "," + format(y, "0.1f"))
            raceline_reward = calculate_reward_using_signmoid(distance)

            heading_reward = 0.0001
            # Heading reward, if it is suppose to turn left but turning right, then penalty
            WP_LOOKAHEAD = 1.5
            lookahead = round(WP_LOOKAHEAD * (prev_speed + speed) / 2)
            next_wp = (closest_waypoints[1] + lookahead) % len(waypoints)
            further_point = track_line[next_wp]
            next_x, next_y = further_point
            wp_direction = math.atan2(next_y - y, next_x - x) 
            print("next_x " + format(next_x, ".3f") + " next y " + format(next_y, ".3f") + " trackline_direction " + format(wp_direction, ".3f"))
            # Convert to degree
            wp_direction_deg = math.degrees(wp_direction)
            # Calculate the difference between the track direction and the heading direction of the car
            direction_diff = heading - wp_direction_deg
            if direction_diff > 180:
                direction_diff = 360 - direction_diff
            elif heading < -180:
                direction_diff = 360 + direction_diff
            steering_diff = abs(direction_diff + steering_angle)
            print("wp_dir_deg " + format(wp_direction_deg, ".1f") + " heading "+ format(heading, ".1f"))
            print("steering_angle " + format(steering_angle, ".1f") + " direction_diff " + format(direction_diff, ".1f") + " steering_diff "+ format(steering_diff, ".1f"))
            STEERING_THRESHOLD = 10
            if steering_diff < STEERING_THRESHOLD:
                heading_reward = 1
            else:
                heading_reward = (1 - abs(steering_diff - STEERING_THRESHOLD) / (180 - STEERING_THRESHOLD)) ** 5
            wp = closest_waypoints[0]
            # if 25 <= wp <= 28:
                # speed_reward = calculate_reward_using_signmoid(abs(speed - 2.6), 5)             
            if 42 <= wp <= 49:    
                speed_reward = calculate_reward_using_signmoid(abs(speed - 2.0), 5)
            # elif 46 <= wp <= 56:
            #     speed_reward = calculate_reward_using_signmoid(abs(speed - 2.5), 5)
            elif 85 <= wp <= 90:
                speed_reward = calculate_reward_using_signmoid(abs(speed - 3.1), 5)
            elif 98 <= wp <= 102:
                speed_reward = calculate_reward_using_signmoid(abs(speed - 2.0), 5)
            elif 112 <= wp <= 116:
                speed_reward = calculate_reward_using_signmoid(abs(speed - 3.6), 5)
            elif wp > 135 or wp < 10:
                speed_reward = calculate_reward_using_signmoid(abs(speed - 4.0), 5)
                if distance < 0.2:
                    if abs(direction_diff) < 10 and abs(steering_angle) <= 11:
                        heading_reward *= 1.2
                    if abs(direction_diff) < 5 and abs(steering_angle) == 0:
                        heading_reward *= 1.2
                        if speed >= 3.3:
                            speed_reward *= 1.2
            else:
                speed_reward = (speed - 2.4) / (4.0 - 2.4)

            if progress > self.one_step_count * 10:
                faster_steps = BENCHMARK_STEPS[self.one_step_count - 1] - steps
                progress_reward = 5 * faster_steps if faster_steps > 0 else 0
                self.one_step_count += 1
                print("progress rewards = " + format(progress_reward, ".3f"))
            else:
                progress_reward = 0

            # adjust weighting
            raceline_reward *= 2

            # if raceline_reward < 1:
            #     heading_reward *= .3
            if heading_reward < 0.6:
                speed_reward = 1e-3
            if speed_reward < 0.6:
                smooth_reward = 1e-3

            total_reward = progress_reward \
                            + (raceline_reward + speed_reward + heading_reward + smooth_reward) ** 2 \
                            + raceline_reward * speed_reward * heading_reward * smooth_reward 

            if progress == 100:
                total_reward += max(260 - steps, 0) ** 2

            print("raceline rewards = " + format(raceline_reward, ".3f") + " for distance = " + format(distance, "0.3f"))
            print("speed rewards = " + format(speed_reward, ".3f"))
            print("heading rewards = " + format(heading_reward, ".3f"))
            print("smooth rewards = " + format(smooth_reward, ".3f"))

            return total_reward
    
reward_object = Reward()  # add parameter verbose=True to get noisy output for testing

def reward_function(params):
    return reward_object.reward_function(params)
