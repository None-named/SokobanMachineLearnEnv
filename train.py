import Env
import utils
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import time

category = "finish_goal"
map_index = 1

map_path = "maps/" + category +"/map" + str(map_index) + ".txt"
out_path = "out/"+ category
save_model_path = "models/" + category + "/"


# 1. 环境包装 (Vectorization & Normalization)# 使用 DummyVecEnv 包装环境，并进行奖励归一化，这对 PPO 的稳定性至关重要
def make_env():
    return Env.SokobanEnv(utils.parse_map_file("maps/" + category +"/map" + str(map_index) + ".txt"))  # 训练时关闭渲染以提高速度

env = DummyVecEnv([make_env])
# 归一化奖励和观测值，有助于算法更快收敛
# env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# 2. 精细化 PPO 参数配置
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,  # 学习率：控制模型权重更新的步长
    n_steps=2048,  # 采样步数：每次更新前收集多少步数据
    batch_size=64,  # 批次大小：每次梯度下降使用的样本数
    n_epochs=10,  # 更新频率：每批数据重复优化的次数
    gamma=0.99,  # 折扣因子：对未来奖励的重视程度
    gae_lambda=0.95,  # GAE 参数：权衡方差与偏差
    clip_range=0.2,  # PPO 剪切范围：防止策略更新过大
    ent_coef=0.5,  # 熵系数：鼓励探索，防止过早陷入局部最优
    vf_coef=0.5,  # 价值函数系数：平衡策略损失和价值损失
    max_grad_norm=0.5,  # 梯度裁剪：防止梯度爆炸
    tensorboard_log="./sokoban_log/"+category,  # TensorBoard 日志，用于观察训练曲线
    verbose=1
)

# 3. 添加定时保存的回调函数
checkpoint_callback = CheckpointCallback(
    save_freq=1_0000,
    save_path='./models/'+category+'/',
    name_prefix='sokoban_ppo'+time.strftime("%Y%m%d-%H%M%S"),
)

# 4. 开始训练
print("开始精细化训练...")
model.learn(
    total_timesteps=50_0000,  # 推箱子较难，建议增加步数
)

# 5. 保存最终模型与统计信息
model.save(out_path+"/Sokoban-"+time.strftime("%Y%m%d-%H%M%S"))
# env.save("vec_normalize.pkl") # 别忘了保存归一化参数，否则推理时效果很差
print("训练完成并保存")