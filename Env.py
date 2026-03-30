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
        self.char_to_int = {' ': 0, '#': 1, '$': 2, '.': 3, '@': 4, '*': 5, '+': 6}
        self.initial_map = np.array([[self.char_to_int[c] for c in row] for row in custom_map])
        self.height, self.width = self.initial_map.shape

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=6, shape=(self.height, self.width), dtype=np.int32)

        # 用于存储历史状态的哈希值
        self.history = set()
        self.reset()

    def _get_hash(self, state):
        """将当前的地图状态转换为唯一的哈希值"""
        return hash(state.tobytes())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initial_map.copy()
        self.player_pos = self._find_player()

        # 重置历史记录并存入初始状态
        self.history = set()
        self.history.add(self._get_hash(self.state))

        return self.state, {}

    def _find_player(self):
        pos = np.where((self.state == 4) | (self.state == 6))
        return pos[0][0], pos[1][0]

    def step(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves[action]
        r, c = self.player_pos
        nr, nc = r + dr, c + dc
        nnr, nnc = nr + dr, nc + dc

        # 先复制一份当前状态进行“虚拟移动”
        next_state = self.state.copy()
        reward = -0.1
        terminated = False

        # 1. 基础逻辑：撞墙判断
        if not (0 <= nr < self.height and 0 <= nc < self.width) or next_state[nr, nc] == 1:
            return self.state, -0.5, False, False, {}  # 撞墙惩罚更高

        target_cell = next_state[nr, nc]
        moved = False

        # 2. 移动逻辑
        if target_cell in [0, 3]:  # 移动到空地或目标点
            self._apply_move(next_state, r, c, nr, nc)
            moved = True
        elif target_cell in [2, 5]:  # 推箱子
            if 0 <= nnr < self.height and 0 <= nnc < self.width:
                beyond_cell = next_state[nnr, nnc]
                if beyond_cell in [0, 3]:
                    next_state[nnr, nnc] = 5 if beyond_cell == 3 else 2
                    self._apply_move(next_state, r, c, nr, nc)
                    moved = True
                    if beyond_cell == 3: reward += 1.0

        # 3. 关键改动：重复状态检查 (Hash Check)
        if moved:
            new_hash = self._get_hash(next_state)
            if new_hash in self.history:
                # 如果这个状态以前出现过，视为非法移动，退回原状态
                return self.state, -0.2, False, False, {"reason": "repeated_state"}
            else:
                # 合法的新状态，更新地图和历史记录
                self.state = next_state
                self.history.add(new_hash)
                self.player_pos = (nr, nc)

        # 4. 胜利检查
        if not np.any(self.state == 2) and not np.any(self.state == 3):
            reward += 10.0
            terminated = True

        return self.state, reward, terminated, False, {}

    def _apply_move(self, state_map, r, c, nr, nc):
        """在给定的地图矩阵上应用移动"""
        state_map[r, c] = 3 if state_map[r, c] == 6 else 0
        state_map[nr, nc] = 6 if state_map[nr, nc] in [3, 5] else 4

    def render(self):
        inv_map = {v: k for k, v in self.char_to_int.items()}
        for row in self.state:
            print("".join([inv_map[cell] for cell in row]))
        print(f"History Size: {len(self.history)}")
        print("-" * self.width)