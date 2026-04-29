from TanksBattleEnv_v3.env import *
from TanksBattleEnv_v3.models import *
import random
import numpy as np

MAX_FORWARD_SPEED = 5
MAX_BACKWARD_SPEED = 2
MAX_BODY_ANGULAR_VELOCITY = 3
MAX_TURRET_ANGULAR_VELOCITY = 1


def convert_action(action_raw, action_dim):
    action_ = [0, 0, 0, 0]
    if 0 in action_dim:

        if action_raw[0] == 0:
            action_[0] = -MAX_BACKWARD_SPEED
        elif action_raw[0] == 1:
            action_[0] = 0
        else:
            action_[0] = MAX_FORWARD_SPEED
    else:
        action_[0] = 0

    if 1 in action_dim:
        if action_raw[1] == 0:
            action_[1] = -MAX_BODY_ANGULAR_VELOCITY
        elif action_raw[1] == 1:
            action_[1] = 0
        else:
            action_[1] = MAX_BODY_ANGULAR_VELOCITY
    else:
        action_[1] = 0

    if 2 in action_dim:
        if action_raw[2] == 0:
            action_[2] = -MAX_TURRET_ANGULAR_VELOCITY
        elif action_raw[2] == 1:
            action_[2] = 0
        else:
            action_[2] = MAX_TURRET_ANGULAR_VELOCITY
    else:
        action_[2] = 0

    if 3 in action_dim:
        if action_raw[3] == 0:
            action_[3] = 0
        else:
            action_[3] = 1
    else:
        action_[3] = 0

    return action_


class env_1v1:
    def __init__(self, max_time=int(1e3), headless=True, speed_control=True, borderlines=True, textual_info=False):
        self.env = TanksBattleEnv(max_time=max_time, headless=headless, speed_control=speed_control,
                                  borderlines=borderlines)
        self.missions = [{'type': 'battle'} for _ in range(2)]
        self.camps = ['red', 'blue']
        self.positions = [[50, 50], [950, 950]]
        self.angles = [0, 180]
        self.tank_sizes = [[8, 8] for _ in range(2)]

        self.obstacle_coords = [[500, 500]]
        self.nan_action = [0, 0, 0, 0]
        self.mission_check = [True, False]

        self.textual_info = textual_info

    def init(self, mission_type='battle'):
        if mission_type == 'battle':
            self.missions[0] = {'type': 'battle'}
        elif mission_type == 'aim':
            self.missions[0] = {'type': 'aim'}
        elif mission_type == 'scout':
            self.missions[0] = {'type': 'scout', 'target': np.random.uniform(0, 1000, 2)}
        elif mission_type == 'hide':
            self.missions[0] = {'type': 'hide'}
        elif mission_type == 'attack':
            self.missions[0] = {'type': 'attack'}
        else:
            self.missions[0] = None
        obss, trainable_dims = self.env.init(self.missions, self.camps, self.positions, self.angles, self.tank_sizes)
        return obss[0], trainable_dims[0]

    def assign_mission(self, mission):
        obs = self.env.assign_mission([mission, None])
        return obs[0]

    def reset(self, random=True, num_obstacle=3, mission_type='battle'):

        if random:
            # 创建障碍物
            obstacle_coords = np.random.uniform(200, 800, (num_obstacle, 2))
            # self.positions[0] = [random.]
        else:
            # 创建障碍物
            obstacle_coords = [[400, 400]]

        if mission_type == 'battle':
            self.missions[0] = {'type': 'battle'}
        elif mission_type == 'scout':
            self.missions[0] = {'type': 'scout', 'target': np.random.uniform(0, 1000, 2)}
        elif mission_type == 'aim':
            self.missions[0] = {'type': 'aim'}
        elif mission_type == 'hide':
            self.missions[0] = {'type': 'hide'}
        elif mission_type == 'attack':
            self.missions[0] = {'type': 'attack'}
        else:
            self.missions[0] = None

        if random and mission_type == 'scout':
            objects = [virtual_Obstacle(obstacle_coords[i][0], obstacle_coords[i][1], 50, 50)
                       for i in range(len(obstacle_coords))]
            self.positions = []
            self.angles = []
            for i in range(2):
                while True:
                    x = np.random.uniform(50, 950)
                    y = np.random.uniform(50, 950)
                    angle = np.random.uniform(-180, 180)
                    tank = virtual_Tank(x, y, angle, self.tank_sizes[i][0], self.tank_sizes[i][0])
                    if not any(rectangles_overlap(tank.get_rotated_points(), obj.get_rotated_points())
                               for obj in objects):
                        self.positions.append([x, y])
                        self.angles.append(angle)
                        objects.append(tank)
                        break

        elif random and mission_type == 'hide':
            objects = [virtual_Obstacle(obstacle_coords[i][0], obstacle_coords[i][1], 50, 50)
                       for i in range(len(obstacle_coords))]
            self.positions = [[0, 0], [0, 0]]
            self.angles = [0, 0]
            while True:
                x = np.random.uniform(50, 950)
                y = np.random.uniform(50, 950)
                angle = np.random.uniform(-180, 180)
                tank2 = virtual_Tank(x, y, angle, self.tank_sizes[1][0], self.tank_sizes[1][0])
                if not any(rectangles_overlap(tank2.get_rotated_points(), obj.get_rotated_points())
                           for obj in objects):
                    self.positions[1] = [x, y]
                    self.angles[1] = angle
                    objects.append(tank2)
                    break

        elif random and mission_type == 'aim':
            objects = [virtual_Obstacle(obstacle_coords[i][0], obstacle_coords[i][1], 50, 50)
                       for i in range(len(obstacle_coords))]
            self.positions = [[0, 0], [0, 0]]
            self.angles = [0, 0]
            while True:
                x = np.random.uniform(50, 950)
                y = np.random.uniform(50, 950)
                angle = np.random.uniform(-180, 180)
                tank2 = virtual_Tank(x, y, angle, self.tank_sizes[1][0], self.tank_sizes[1][1])
                if not any(rectangles_overlap(tank2.get_rotated_points(), obj.get_rotated_points())
                           for obj in objects):
                    self.positions[1] = [x, y]
                    self.angles[1] = angle
                    objects.append(tank2)
                    break

        elif random and mission_type == 'attack' or self.textual_info:
            objects = [virtual_Obstacle(obstacle_coords[i][0], obstacle_coords[i][1], 50, 50)
                       for i in range(len(obstacle_coords))]
            self.positions = []
            self.angles = []
            for i in range(2):
                while True:
                    x = np.random.uniform(50, 950)
                    y = np.random.uniform(50, 950)
                    angle = np.random.uniform(-180, 180)
                    tank = virtual_Tank(x, y, angle, self.tank_sizes[i][0], self.tank_sizes[i][0])
                    if not any(rectangles_overlap(tank.get_rotated_points(), obj.get_rotated_points())
                               for obj in objects):
                        self.positions.append([x, y])
                        self.angles.append(angle)
                        objects.append(tank)
                        break
        if self.textual_info:
            obss, info = self.env.reset(self.missions, self.camps, self.positions, self.angles, self.tank_sizes,
                                        obstacle_coords, self.textual_info)
            return obss[0], info[0]

        else:
            obss = self.env.reset(self.missions, self.camps, self.positions, self.angles, self.tank_sizes,
                                  obstacle_coords)
            return obss[0]

    def aim_fn(self, tank, target):
        action_ = [0, 0, 0, 0]
        if target is None:
            return action_

        angle_to_enemy = math.atan2(target[1] - tank.y, target[0] - tank.x)
        angle_to_enemy = math.degrees(angle_to_enemy) % 360
        angle_diff = angle_difference(tank.aim_angle, angle_to_enemy)

        if - MAX_TURRET_ANGULAR_VELOCITY < angle_diff < 0:
            action_[2] = angle_diff
        elif angle_diff < 0:
            action_[2] = - MAX_TURRET_ANGULAR_VELOCITY
        elif 0 < angle_diff < MAX_TURRET_ANGULAR_VELOCITY:
            action_[2] = angle_diff
        elif angle_diff > 0:
            action_[2] = MAX_TURRET_ANGULAR_VELOCITY
        else:
            action_[2] = 0
        return action_

    def auto_pilot_fn(self, tank, target):
        action_ = [0, 0, 0, 0]
        x = tank.x
        y = tank.y
        distance = math.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2)
        f_target = np.array([target[0] - x, target[1] - y]) / distance

        obstacles = self.env.obstacles
        f_obst = np.array([0., 0.])
        weight_sum = 0
        obs_num = len(obstacles) - 4
        for i in range(4, len(obstacles)):
            obst = obstacles[i]
            center = obst.center
            distance = math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            force_unit = np.array([x - center[0], y - center[1]]) / distance
            weight_sum += 1 / distance
            if distance < 50:
                f_obst += 1.2 * force_unit
            elif distance < 60:
                f_obst += 0.8 * force_unit
            elif distance < 70:
                f_obst += 0.6 * force_unit
            elif distance < 80:
                f_obst += 0.4 * force_unit
            # elif distance < 90:
            #     f_obst += 0.2 * force_unit
            # elif distance < 100:
            #     f_obst += 0.1 * force_unit
            # elif distance < 150:
            #     f_obst += 0.05 * force_unit
            else:
                obs_num -= 1
                weight_sum -= 1 / distance

        tanks = self.env.tanks
        tank_num = len(tanks) - 1
        f_tank = np.array([0., 0.])
        for i, ta in enumerate(tanks):
            if ta == tank:
                continue
            center = [ta.x, ta.y]
            distance = math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            force_unit = np.array([x - center[0], y - center[1]]) / distance
            if distance < 10:
                f_tank += 1.1 * force_unit
            elif distance < 18:
                f_tank += 0.8 * force_unit
            elif distance < 32:
                f_tank += 0.6 * force_unit
            else:
                tank_num -= 1

        f = f_target + f_obst / max(weight_sum, 1) + f_tank / max(tank_num, 1)
        angle_to_go = math.atan2(f[1], f[0])

        angle_to_go = math.degrees(angle_to_go) % 360
        angle_diff = angle_difference(tank.body_angle, angle_to_go)

        if - MAX_TURRET_ANGULAR_VELOCITY < angle_diff < 0:
            action_[1] = angle_diff
            action_[0] = MAX_FORWARD_SPEED
        elif angle_diff < 0:
            action_[1] = - MAX_BODY_ANGULAR_VELOCITY
        elif 0 < angle_diff < MAX_BODY_ANGULAR_VELOCITY:
            action_[1] = angle_diff
            action_[0] = MAX_FORWARD_SPEED
        elif angle_diff > 0:
            action_[1] = MAX_BODY_ANGULAR_VELOCITY
        elif abs(angle_diff) < 1e-3:
            action_[0] = MAX_FORWARD_SPEED

        return action_

    def step_from_keyboard(self, action):
        return self.env.step([action, nan_action], self.missions, self.mission_check)

    def step_from_keyboard_v2(self, action, auto_fire=False):
        blue_target = None
        blue_enemy = self.env.tanks[1].enemy_last_seen[0]
        if blue_enemy['position'][0] >= 0:
            blue_target = [blue_enemy['position'][0], blue_enemy['position'][1]]
        if self.env.tanks[1].aim_target_type == 4 and auto_fire:
            blue_action = [0, 0, 0, 1]
        else:
            blue_action = self.aim_fn(self.env.tanks[1], blue_target)

        return self.env.step([action, blue_action], self.missions, self.mission_check)

    def step(self, action, auto_aim=True, auto_fire=False, auto_pilot=False):
        if auto_pilot and self.missions[0]['type'] == 'scout':
            action = self.auto_pilot_fn(self.env.tanks[0], self.missions[0]['target'])
        else:
            action = convert_action(action, self.env.tanks[0].get_trainable_dim(self.missions[0]['type']))

        if self.env.tanks[0].aim_target_type == 4 and auto_fire:
            action = [0, 0, 0, 1]
        elif auto_aim:
            red_target = None
            red_enemy = self.env.tanks[0].enemy_last_seen[0]
            if red_enemy['position'][0] >= 0:
                red_target = [red_enemy['position'][0], red_enemy['position'][1]]
            red_action = self.aim_fn(self.env.tanks[0], red_target)
            action[2] = red_action[2]

        blue_target = None
        blue_enemy = self.env.tanks[1].enemy_last_seen[0]
        if blue_enemy['position'][0] >= 0:
            blue_target = [blue_enemy['position'][0], blue_enemy['position'][1]]
        if self.env.tanks[1].aim_target_type == 4 and auto_fire:
            blue_action = [0, 0, 0, 1]
        else:
            blue_action = self.aim_fn(self.env.tanks[1], blue_target)
        # obss, rewards, terminal, wins = self.env.step([action, self.nan_action], self.missions, self.mission_check)
        if self.textual_info:
            obss, mission_terminals, terminal, tanks_info = self.env.step([action, blue_action], self.missions,
                                                                          self.mission_check, self.textual_info, auto_fire)
            if self.env.tanks[1].is_distroyed or self.env.tanks[0].is_distroyed:
                terminal = True
            return obss[0], mission_terminals[0], terminal, tanks_info[0]

        else:
            obss, rewards, terminal, wins = self.env.step([action, blue_action], self.missions, self.mission_check, auto_fire)
            return obss[0], rewards[0], terminal, wins[0]


if __name__ == '__main__':
    env = env_1v1(max_time=int(100000), headless=False)
    mission_type = 'attack'
    obs = env.reset(random=True, mission_type=mission_type)
    nan_action = [0, 0, 0, 0]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 用户控制坦克1
        action_red = [0, 0, 0, 0]  # [加速度, 车体转向, 炮塔转向, 开火]
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action_red[0] = 1
        if keys[pygame.K_s]:
            action_red[0] = -1
        if keys[pygame.K_a]:
            action_red[1] = -1
        if keys[pygame.K_d]:
            action_red[1] = 1
        if keys[pygame.K_q]:  # Q键用于控制坦克1的炮塔向左转
            action_red[2] = -1
        if keys[pygame.K_e]:  # E键用于控制坦克1的炮塔向右转
            action_red[2] = 1
        if keys[pygame.K_SPACE]:  # 空格键用于坦克1开火
            action_red[3] = 1

        s, r, t, w = env.step_from_keyboard_v2(action_red)
        print(s[0])
        # if t:
        #     env.reset(random=True, mission_type=mission_type)
        # print(t)
