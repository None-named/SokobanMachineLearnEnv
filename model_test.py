from stable_baselines3 import PPO
import Env
import read_map
import time
import matplotlib.pyplot as plt

# 定义地图（符号表示）
test_map_data = read_map.parse_map_file("maps/explore/map.txt")

# 创建环境
env = Env.SokobanEnv(test_map_data)

loaded_model = PPO.load("Sokoban_Refined", env=env)

# 运行示例
obs, _ = env.reset()
while True:
    action, _ = loaded_model.predict(obs, deterministic=True)
    action = int(action)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        obs, _ = env.reset()
        break
    env.render()
env.close()
