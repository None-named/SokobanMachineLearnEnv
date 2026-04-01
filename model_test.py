from stable_baselines3 import PPO
import Env
import utils
import time
import matplotlib.pyplot as plt

category = utils.read_json_variable("states.json", "category")
map_index = utils.read_json_variable("states.json", "map_index")

map_path = "maps/" + category + "/map" + str(map_index) + ".txt"
out_path = "out/" + category

# 定义地图（符号表示）
test_map_data = utils.parse_map_file(map_path)
print("地图:", map_path)

# 创建环境
env = Env.SokobanEnv(test_map_data)

model = utils.get_latest_file_concise(out_path)
print("模型:", model)
loaded_model = PPO.load(model, env=env)

# 运行示例
obs, _ = env.reset()

steps = 0  # 记录步数
start_time = time.time()  # 记录开始时间

while True:
    action, _ = loaded_model.predict(obs, deterministic=True)
    action = int(action)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        obs, _ = env.reset()
        break
    env.render()
    time.sleep(0.25)
    steps += 1
env.close()

end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time - steps * 0.25  # 计算耗时
print("步数:", steps, "推测用时:", elapsed_time, "s")
