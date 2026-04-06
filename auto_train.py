from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

import Env
import utils
from pathlib import Path
import maps.map_manager as map_manager
import train
import time
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO

time_format = "%Y_%m_%d-%H_%M_%S"

if __name__ == "__main__":
    project_time = time.strftime(time_format)
    envs: list[Env.SokobanEnv] = []
    subproc_envs_funcs = []
    evl_envs_funcs = []
    maps_path = []
    for index, map_path in enumerate(map_manager.get_maps(Path("maps"))):
        for category, path in map_path.items():
            if category == "finish_goal":
                maps_path.append(path)

    for index, path in enumerate(maps_path):
        subproc_envs_funcs.append(lambda: Env.SokobanEnv(utils.parse_map_file(path), seed=index))
        # evl_envs_funcs.append(lambda: Monitor(Env.SokobanEnv(utils.parse_map_file(path))))

    chunk_size = 3
    sub_subproc_env_func_lists = utils.split_list(subproc_envs_funcs, 3)

    for func_list in sub_subproc_env_func_lists:
        if len(func_list) != 3:
            sub_subproc_env_func_lists.remove(func_list)

    model = PPO(
        policy="MlpPolicy",
        env=SubprocVecEnv(sub_subproc_env_func_lists[0]),
        learning_rate=3e-4,  # 学习率：控制模型权重更新的步长
        n_steps=4096,  # 采样步数：每次更新前收集多少步数据
        batch_size=512,  # 批次大小：每次梯度下降使用的样本数
        n_epochs=20,  # 更新频率：每批数据重复优化的次数
        gamma=0.99,  # 折扣因子：对未来奖励的重视程度
        gae_lambda=0.95,  # GAE 参数：权衡方差与偏差
        clip_range=0.6,  # PPO 剪切范围：防止策略更新过大
        ent_coef=0.2,  # 熵系数：鼓励探索，防止过早陷入局部最优
        vf_coef=0.5,  # 价值函数系数：平衡策略损失和价值损失
        max_grad_norm=0.5,  # 梯度裁剪：防止梯度爆炸
        tensorboard_log="./sokoban_log/" + project_time,  # TensorBoard 日志，用于观察训练曲线
        verbose=1,
    )

    for sub_subproc_env_func_list in sub_subproc_env_func_lists:
        model.env = SubprocVecEnv(sub_subproc_env_func_list)
        model.learn(
            total_timesteps=500_0000,  # 推箱子较难，建议增加步数
            callback=[
                CheckpointCallback(
                    save_freq=10_0000,
                    save_path="./out/" + project_time + "/checkpoint/",
                    name_prefix="sokoban_checkpoint",
                ),
                EvalCallback(
                    eval_env=SubprocVecEnv([lambda: Env.SokobanEnv(utils.parse_map_file("maps/finish_goal/map4.txt"))]),
                    eval_freq=10_0000,
                    best_model_save_path="./out/" + project_time + "/best_model/",
                )],
        )

    # best_model = str(utils.get_latest_item("out/" + project_time)) + "/best_model/best_model.zip"
    # result = model_test.test_ppo_model(PPO.load(best_model), envs[index])

    model.save("out/latest/Sokoban-" + time.strftime(time_format))
    print("训练完成并保存")
    model.env.close()
