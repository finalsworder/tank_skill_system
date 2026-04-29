from TanksBattleEnv_v3.env import *


class env_3v3:

    def __init__(self):
        self.env = TanksBattleEnv(max_time=int(1e3), headless=False, speed_control=True)
        self.missions = [{'type': 'battle'} for _ in range(6)]
        self.camps = ['red', 'red', 'red', 'blue', 'blue', 'blue']
        self.positions = [[50, 50], [100, 50], [150, 50], [950, 950], [900, 950], [850, 950]]
        self.tank_sizes = [[8, 8] for _ in range(6)]
        self.angles = [0, 0, 0, 180, 180, 180]
        self.obstacle_coords = [[500, 500]]
        self.nan_action = [1, 1, 1, 0]

    def reset(self):
        return self.env.reset(self.missions, self.camps, self.positions, self.angles,
                              self.tank_sizes, self.obstacle_coords)

    def step(self, actions):
        return self.env.step(actions, self.missions)


if __name__ == '__main__':
    env = env_3v3()
    obs = env.reset()
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
            action_red[0] = 5
        if keys[pygame.K_s]:
            action_red[0] = -5
        if keys[pygame.K_a]:
            action_red[1] = -5
        if keys[pygame.K_d]:
            action_red[1] = 5
        if keys[pygame.K_q]:  # Q键用于控制坦克1的炮塔向左转
            action_red[2] = -5
        if keys[pygame.K_e]:  # E键用于控制坦克1的炮塔向右转
            action_red[2] = 5
        if keys[pygame.K_SPACE]:  # 空格键用于坦克1开火
            action_red[3] = 1

        action_blue = [0, 0, 0, 0]  # [加速度, 车体转向, 炮塔转向, 开火]
        if keys[pygame.K_i]:
            action_blue[0] = 1
        if keys[pygame.K_k]:
            action_blue[0] = -1
        if keys[pygame.K_j]:
            action_blue[1] = -1
        if keys[pygame.K_l]:
            action_blue[1] = 1
        if keys[pygame.K_u]:  # Q键用于控制坦克1的炮塔向左转
            action_blue[2] = -0.1
        if keys[pygame.K_o]:  # E键用于控制坦克1的炮塔向右转
            action_blue[2] = 0.1
        if keys[pygame.K_m]:  # 空格键用于坦克1开火
            action_blue[3] = 1

        actions = [action_red, nan_action, nan_action, action_blue, nan_action, nan_action]

        s, r, t, w = env.step(actions)
        if t:
            print(t)
