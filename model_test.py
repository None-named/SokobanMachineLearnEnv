from stable_baselines3 import PPO
import Env
import utils
import time


def test_ppo_model(model: PPO, env: Env.SokobanEnv, render: bool) -> bool:
    obs, _ = env.reset()
    actions_name = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    # 统计信息
    total_reward = 0
    total_steps = 0
    start_time = time.time()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1
        if render:
            env.render()
            time.sleep(0.25)
            print("Step Info:", "step:", total_steps, "action:", actions_name[action], "reward:", reward)
        if terminated or truncated:
            obs, _ = env.reset()
            break
    end_time = time.time()
    env.close()
    if render:
        elapsed_time = end_time - start_time - total_steps * 0.25
        print("Total Reward:", total_reward)
        print("Total Time:", elapsed_time)
        print("Total Steps:", total_steps)
    else:
        elapsed_time = end_time - total_steps
    return reward > 0


if "__main__" == __name__:
    # category = utils.read_json_variable("states.json", "category")
    category = "finish_goal"
    # map_index = utils.read_json_variable("states.json", "map_index")
    map_index = 2

    map_path = "maps/" + category + "/map" + str(map_index) + ".txt"
    out_path = "out/" + category
    actions_name = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    # 定义地图（符号表示）
    test_map_data = utils.parse_map_file(map_path)
    print("地图:", map_path)

    test_ppo_model(PPO.load(utils.get_latest_file_concise(out_path)), Env.SokobanEnv(test_map_data), render=True)
