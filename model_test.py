from stable_baselines3 import PPO
import Env
import utils
import time

# category = utils.read_json_variable("states.json", "category")
category = "explore"
# map_index = utils.read_json_variable("states.json", "map_index")
map_index = 1

map_path = "maps/" + category + "/map" + str(map_index) + ".txt"
out_path = "out/" + category
actions_name = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

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

steps = 1  # 记录步数
start_time = time.time()  # 记录开始时间
total_reward = 0

while True:
    action, _ = loaded_model.predict(obs, deterministic=True)
    action = int(action)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print("Step Info:", "step:", steps, "action:", actions_name[action], "reward:", reward)
    if terminated or truncated:
        obs, _ = env.reset()
        break
    env.render()
    time.sleep(0.25)
    steps += 1
env.close()

if reward > 0:
    print("Successful")
else:
    print("Fail")

end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time - steps * 0.25  # 计算耗时
print("步数:", steps, "推测用时:", elapsed_time, "s")
print("累计奖励:", total_reward)

