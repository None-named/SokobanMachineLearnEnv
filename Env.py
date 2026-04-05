import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SokobanEnv(gym.Env):
    # 地图元素常量定义
    EMPTY = 0
    WALL = 1
    BOX = 2
    TARGET = 3
    PLAYER = 4
    BOX_ON_TARGET = 5  # 虽然逻辑上箱子会消失，但保留定义以防初始化需要
    PLAYER_ON_TARGET = 6

    def __init__(self, custom_map, render=None, seed=None):
        super(SokobanEnv, self).__init__()
        self.render_mode = render
        self.seed = seed

        # 建立字符映射
        self.char_to_int = {' ': 0, '#': 1, '$': 2, '.': 3, '@': 4, '*': 5, '+': 6}
        self.initial_map = np.array([[self.char_to_int[c] for c in row] for row in custom_map], dtype=np.int32)
        self.height, self.width = self.initial_map.shape

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=6, shape=(self.height, self.width), dtype=np.int32)

        self.history = set()
        self.is_win_init = False

        # 颜色映射表
        self.colors = {
            self.EMPTY: (255, 255, 255),  # 白色
            self.WALL: (0, 0, 0),  # 黑色
            self.BOX: (255, 255, 0),  # 黄色
            self.TARGET: (0, 255, 0),  # 绿色
            self.PLAYER: (0, 0, 255),  # 蓝色
            self.BOX_ON_TARGET: (128, 128, 0),
            self.PLAYER_ON_TARGET: (0, 0, 128)
        }
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initial_map.copy()
        self.player_pos = self._find_player()
        self.history = {self._get_hash(self.state)}
        return self.state, {}

    def _find_player(self):
        pos = np.where((self.state == self.PLAYER) | (self.state == self.PLAYER_ON_TARGET))
        return pos[0][0], pos[1][0]

    def _get_hash(self, state):
        return hash(state.tobytes())

    def _is_valid_pos(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width

    def step(self, action):
        """核心步骤控制"""
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # 上, 下, 左, 右
        dr, dc = moves[action]
        r, c = self.player_pos
        nr, nc = r + dr, c + dc  # 目标格 (Next)
        nnr, nnc = nr + dr, nc + dc  # 目标格的下一格 (Next-Next)

        # 碰撞检查：是否撞墙或越界
        if not self._is_valid_pos(nr, nc) or self.state[nr, nc] == self.WALL:
            return self.state, -5, False, True, {"reason": "collision"}

        # 核心逻辑分支
        moved = False
        reward = 0
        temp_state = self.state.copy()
        target_cell = temp_state[nr, nc]

        if target_cell in [self.EMPTY, self.TARGET]:
            # 情况 A: 普通移动
            self._apply_player_move(temp_state, r, c, nr, nc)
            moved = True
            reward = 1

        elif target_cell in [self.BOX, self.BOX_ON_TARGET]:
            # 情况 B: 推箱子逻辑
            if self._is_valid_pos(nnr, nnc):
                beyond_cell = temp_state[nnr, nnc]

                if beyond_cell == self.TARGET:
                    # 特殊规则：箱子进入目标点，两者同时消失
                    self._handle_disappear(temp_state, r, c, nr, nc, nnr, nnc)
                    moved = True
                    reward = 100
                elif beyond_cell == self.EMPTY:
                    # 普通推箱子：推入空地
                    self._handle_normal_push(temp_state, r, c, nr, nc, nnr, nnc)
                    moved = True
                    reward = 2

        # 状态一致性检查（重复路径/无效移动）
        if moved:
            new_hash = self._get_hash(temp_state)
            if new_hash in self.history:
                return self.state, -5, False, True, {"reason": "loop_detected"}

            self.state = temp_state
            self.history.add(new_hash)
            self.player_pos = (nr, nc)
        else:
            return self.state, -5, False, True, {"reason": "blocked"}

        # 4. 胜利条件判断
        terminated = self._check_win()
        if terminated:
            reward += 1000

        return self.state, reward, terminated, False, {}

    def _apply_player_move(self, state, r, c, nr, nc):
        """处理玩家从 (r,c) 移动到 (nr,nc) 的地标变换"""
        # 离开原位置
        state[r, c] = self.TARGET if state[r, c] == self.PLAYER_ON_TARGET else self.EMPTY
        # 进入新位置
        state[nr, nc] = self.PLAYER_ON_TARGET if state[nr, nc] == self.TARGET else self.PLAYER

    def _handle_disappear(self, state, r, c, nr, nc, nnr, nnc):
        """处理箱子推入目标点消失的特殊逻辑"""
        # 1. 玩家离开原位
        state[r, c] = self.TARGET if state[r, c] == self.PLAYER_ON_TARGET else self.EMPTY
        # 2. 玩家占据箱子原来的位置 (由于原位置不是TARGET，所以直接设为PLAYER)
        state[nr, nc] = self.PLAYER
        # 3. 箱子和目标点消失 -> 目标位置变为空地
        state[nnr, nnc] = self.EMPTY

    def _handle_normal_push(self, state, r, c, nr, nc, nnr, nnc):
        """处理普通推箱子移动"""
        # 1. 箱子移动到下一格
        state[nnr, nnc] = self.BOX
        # 2. 玩家移动到箱子原位
        self._apply_player_move(state, r, c, nr, nc)

    def _check_win(self):
        """如果没有箱子且没有目标点，则获胜"""
        has_box = np.any(self.state == self.BOX)
        has_target = np.any(self.state == self.TARGET)
        return not (has_box or has_target)

    def render(self):
        if not self.is_win_init:
            pygame.init()
            self.cell_size = 50
            self.screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
            pygame.display.set_caption("Sokoban - Disappearing Mode")
            self.is_win_init = True

        self.screen.fill((200, 200, 200))
        for r in range(self.height):
            for c in range(self.width):
                val = self.state[r, c]
                # 画背景方块
                pygame.draw.rect(self.screen, self.colors[val],
                                 (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit()

    def close(self):
        pygame.quit()
