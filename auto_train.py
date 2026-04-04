import Env
import utils
from pathlib import Path
import maps.map_manager as map_manager
import train
import time
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv

envs: list[DummyVecEnv] = []
for map in map_manager.get_maps(Path("maps")):
    for category, path in map.items():
        if category == "finish_goal":
            envs.append(DummyVecEnv([lambda: Env.SokobanEnv(utils.parse_map_file(path))]))

model = train.init_ppo_model()
map_index = 1
project_time = time.strftime("%Y_%m_%d-%H_%M_%S")
model.tensorboard_log = "sokoban_log/" + project_time

for env in envs:
    current_time = time.strftime("%Y%m%d-%H%M%S")
    print(current_time, " : ", ("map" + str(map_index)))
    model.env = env
    model.learn(
        total_timesteps=1500_0000,  # 推箱子较难，建议增加步数
        callback=[train.get_eval_callback(env, "out/" + project_time + "/" + current_time),
                  train.get_checkpoint_callback("out/" + project_time + "/" + current_time)],
    )
    map_index += 1

model.save("out/Sokoban-" + time.strftime("%Y%m%d-%H%M%S"))
print("训练完成并保存")
model.env.close()
