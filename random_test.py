from stable_baselines3 import PPO
import Env
import read_map
import time
import matplotlib.pyplot as plt
import numpy as np

# 定义地图（符号表示）
test_map_data = read_map.parse_map_file("map.txt")

# 创建环境
env = Env.SokobanEnv(test_map_data)

# 运行示例
obs, _ = env.reset()
for _ in range(5):
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated:
        obs, _ = env.reset()
env.close()
