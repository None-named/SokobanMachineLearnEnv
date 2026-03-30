import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SokobanEnv(gym.Env):
    def __init__(self, custom_map):
        super(SokobanEnv, self).__init__()

        # 1. 定义映射表
        '''
        0: 空地 ( )
        1: 墙壁 (#)
        2: 箱子 ($)
        3: 目标点 (.)
        4: 玩家 (@)
        5: 箱子在目标点上 (*)
        6: 玩家在目标点上 (+)
        '''
        self.char_to_int = {
            ' ': 0, '#': 1, '$': 2, '.': 3, '@': 4, '*': 5, '+': 6
        }

        # 2. 初始化地图
        self.initial_map = np.array([[self.char_to_int[c] for c in row] for row in custom_map])
        self.height, self.width = self.initial_map.shape

        # 3. 定义动作空间: 0:上, 1:下, 2:左, 3:右
        self.action_space = spaces.Discrete(4)

        # 4. 定义观测空间 (地图矩阵)
        self.observation_space = spaces.Box(low=0, high=6, shape=(self.height, self.width), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initial_map.copy()
        self.player_pos = self._find_player()
        return self.state, {}

    def _find_player(self):
        # 寻找玩家 (@ 或 +) 的位置
        pos = np.where((self.state == 4) | (self.state == 6))
        return pos[0][0], pos[1][0]

    def step(self, action):
        # 动作增量映射
        moves = {
            0: (-1, 0),  # 上
            1: (1, 0),  # 下
            2: (0, -1),  # 左
            3: (0, 1)  # 右
        }

        dr, dc = moves[action]
        r, c = self.player_pos
        nr, nc = r + dr, c + dc  # 目标位置
        nnr, nnc = nr + dr, nc + dc  # 目标位置再往前一格（推箱子用）

        reward = -0.1  # 每走一步的惩罚
        terminated = False

        # 边界检查
        if not (0 <= nr < self.height and 0 <= nc < self.width):
            return self.state, reward, False, False, {}

        target_cell = self.state[nr, nc]

        # 情况1：目标是空地或目标点 (直接移动)
        if target_cell in [0, 3]:
            self._move_player(r, c, nr, nc)

        # 情况2：目标是箱子 (尝试推)
        elif target_cell in [2, 5]:
            if 0 <= nnr < self.height and 0 <= nnc < self.width:
                beyond_cell = self.state[nnr, nnc]
                if beyond_cell in [0, 3]:  # 箱子后面是空地或目标
                    # 移动箱子
                    self.state[nnr, nnc] = 5 if beyond_cell == 3 else 2
                    # 移动玩家
                    self._move_player(r, c, nr, nc)
                    if beyond_cell == 3: reward += 1.0  # 推入目标奖励

        # 更新玩家坐标
        self.player_pos = self._find_player()

        # 检查是否胜利 (地图上不再有单独的箱子 $ 或 目标点 .)
        if not np.any(self.state == 2) and not np.any(self.state == 3):
            reward += 10.0
            terminated = True

        return self.state, reward, terminated, False, {}

    def _move_player(self, r, c, nr, nc):
        # 处理旧位置
        self.state[r, c] = 3 if self.state[r, c] == 6 else 0
        # 处理新位置
        self.state[nr, nc] = 6 if self.state[nr, nc] in [3, 5] else 4

    def render(self):
        inv_map = {v: k for k, v in self.char_to_int.items()}
        for row in self.state:
            print("".join([inv_map[cell] for cell in row]))
        print("-" * self.width)