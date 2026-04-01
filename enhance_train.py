import Env
import utils
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
import time

category = utils.read_json_variable("states.json", "category")
map_index = 3

map_path = "maps/" + category +"/map" + str(map_index) + ".txt"
out_path = "out/"+ category
save_model_path = "models/" + category + "/"

def make_env():
    return Env.SokobanEnv(utils.parse_map_file("maps/" + category +"/map" + str(map_index) + ".txt"))  # 训练时关闭渲染以提高速度

env = DummyVecEnv([make_env])
buffer_model = PPO.load(utils.get_latest_file_concise("out/"+utils.read_json_variable("states.json","category")))

# 3. 添加定时保存的回调函数
checkpoint_callback = CheckpointCallback(
    save_freq=1_0000,
    save_path='./models/'+category+'/',
    name_prefix='sokoban_checkpoint'+time.strftime("%Y%m%d-%H%M%S"),
)

eval_callback = EvalCallback(
    env,
    best_model_save_path="./"+category+"/best_model",
    log_path="./"+category+"/eval_logs",
    eval_freq=5_0000,          # 每5万步评估一次
    deterministic=True,
    render=False,
)

buffer_model.env = env

buffer_model.learn(
    total_timesteps=5_0000,
    callback=[checkpoint_callback, eval_callback],
)
buffer_model.save(out_path+"/Sokoban-"+time.strftime("%Y%m%d-%H%M%S"))

buffer_model.env.close()