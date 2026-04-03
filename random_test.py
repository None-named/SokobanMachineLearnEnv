from stable_baselines3 import PPO
import Env
import utils
import time
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_checker import check_env

# 定义地图（符号表示）
test_map_data = utils.parse_map_file("maps/explore/map1.txt")

# 创建环境
env = Env.SokobanEnv(test_map_data)
# 检查环境
check_env(env)
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
