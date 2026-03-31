from stable_baselines3 import PPO
import Env
import read_map
import time
import matplotlib.pyplot as plt
import numpy as np

# 定义地图（符号表示）
test_map_data = read_map.parse_map_file("maps/explore/map.txt")

# 创建环境
env = Env.SokobanEnv(test_map_data)

# 运行示例
obs, _ = env.reset()
actions_name = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
for _ in range(50):
    action = env.action_space.sample()  # 随机动作
    print(f"Action: {actions_name[action]}")
    obs, reward, terminated, truncated, info = env.step(action)
    print("After")
    if terminated:
        obs, _ = env.reset()
        print("rest!")
    else:
        env.render()
        time.sleep(0.25)
    print(" ")
