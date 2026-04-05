import Env
import model_test
import utils
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
import time

default_env = Env.SokobanEnv(utils.parse_map_file("maps/finish_goal/map1.txt"))


def make_env() -> Env.SokobanEnv:
    return Env.SokobanEnv(
        utils.parse_map_file("maps/" + category + "/map" + str(map_index) + ".txt")
    )  # 训练时关闭渲染以提高速度


# 2. 精细化 PPO 参数配置
def init_ppo_model() -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=default_env,
        learning_rate=3e-4,  # 学习率：控制模型权重更新的步长
        n_steps=4096,  # 采样步数：每次更新前收集多少步数据
        batch_size=512,  # 批次大小：每次梯度下降使用的样本数
        n_epochs=20,  # 更新频率：每批数据重复优化的次数
        gamma=0.99,  # 折扣因子：对未来奖励的重视程度
        gae_lambda=0.95,  # GAE 参数：权衡方差与偏差
        clip_range=0.6,  # PPO 剪切范围：防止策略更新过大
        ent_coef=0,  # 熵系数：鼓励探索，防止过早陷入局部最优
        vf_coef=0.5,  # 价值函数系数：平衡策略损失和价值损失
        max_grad_norm=0.5,  # 梯度裁剪：防止梯度爆炸
        tensorboard_log="./sokoban_log/",  # TensorBoard 日志，用于观察训练曲线
        verbose=1,
    )


def get_eval_callback(__env, log_path: str) -> EvalCallback:
    return EvalCallback(
        eval_env=__env,
        best_model_save_path=log_path + "/best_model",
        log_path=log_path + "/eval_logs",
        eval_freq=50_0000,  # 每5万步评估一次
        deterministic=True,
        render=False,
    )


def get_checkpoint_callback(log_path: str) -> CheckpointCallback:
    return CheckpointCallback(
        save_freq=50_0000,
        save_path=log_path + "/checkpoint",
        name_prefix="sokoban_checkpoint",
    )


if __name__ == "__main__":
    category = "finish_goal"
    map_index = 2

    map_path = "maps/" + category + "/map" + str(map_index) + ".txt"
    out_path = "out/" + category
    save_model_path = "models/" + category + "/"

    env = DummyVecEnv([make_env])

    # 3. 添加定时保存的回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=1_0000,
        save_path="./models/" + category + "/",
        name_prefix="sokoban_checkpoint" + time.strftime("%Y%m%d-%H%M%S"),
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path="./out/" + category + "/best_model",
        log_path="./" + category + "/eval_logs",
        eval_freq=1_0000,  # 每5万步评估一次
        deterministic=True,
        render=False,
    )

    model = init_ppo_model()

    # 4. 开始训练
    print("开始精细化训练...")
    model.learn(
        total_timesteps=10_0000,  # 推箱子较难，建议增加步数
        callback=[checkpoint_callback, eval_callback],
    )

    # 5. 保存最终模型与统计信息
    model.save(out_path + "/Sokoban-" + time.strftime("%Y%m%d-%H%M%S"))
    print("训练完成并保存")
