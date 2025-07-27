import numpy as np
import os
import sys

# add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from Environment.dfjspt_env import FjspMaEnv


def est_eet_rule(env, verbose=False):
    """
    基础规则：每次选择最早可开始作业（EST），分配最早可用机器（EET）。
    """
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        # 作业选择阶段
        job_mask = obs['agent0']['action_mask']
        job_features = obs['agent0']['observation']
        available_jobs = np.where(job_mask == 1)[0]
        if len(available_jobs) == 0:
            job_action = 0  # fallback
        else:
            arrival_times = job_features[available_jobs, 2]
            job_action = available_jobs[np.argmin(arrival_times)]
        action = {'agent0': job_action}
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward['agent0']
        if terminated['__all__']:
            break
        # 机器选择阶段
        machine_mask = obs['agent1']['action_mask']
        machine_features = obs['agent1']['observation']
        available_machines = np.where(machine_mask == 1)[0]
        if len(available_machines) == 0:
            machine_action = 0  # fallback
        else:
            idle_times = machine_features[available_machines, 3]
            machine_action = available_machines[np.argmin(idle_times)]
        action = {'agent1': machine_action}
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward['agent1']
        done = terminated['__all__']
    return env.final_makespan, total_reward

def est_eet_rule_weighted(env, alpha=0.7, verbose=False):
    """
    优先级加权规则：作业选择时，优先级高的订单更易被选中（带权重），机器分配仍为EET。
    alpha: 优先级权重，0~1，越大越偏向优先级，越小越偏向到达时间。
    """
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        # 作业选择阶段
        job_mask = obs['agent0']['action_mask']
        job_features = obs['agent0']['observation']
        available_jobs = np.where(job_mask == 1)[0]
        if len(available_jobs) == 0:
            job_action = 0  # fallback
        else:
            priorities = np.array([env.job_priority[job_id] for job_id in available_jobs])
            arrival_times = job_features[available_jobs, 2]
            # 归一化
            norm_priority = priorities / (priorities.max() if priorities.max() > 0 else 1)
            norm_arrival = (arrival_times - arrival_times.min()) / (np.ptp(arrival_times) + 1e-6)
            score = alpha * norm_priority - (1 - alpha) * norm_arrival
            job_action = available_jobs[np.argmax(score)]
        action = {'agent0': job_action}
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward['agent0']
        if terminated['__all__']:
            break
        # 机器选择阶段
        machine_mask = obs['agent1']['action_mask']
        machine_features = obs['agent1']['observation']
        available_machines = np.where(machine_mask == 1)[0]
        if len(available_machines) == 0:
            machine_action = 0  # fallback
        else:
            idle_times = machine_features[available_machines, 3]
            machine_action = available_machines[np.argmin(idle_times)]
        action = {'agent1': machine_action}
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward['agent1']
        done = terminated['__all__']
    return env.final_makespan, total_reward

if __name__ == '__main__':
    input_case_name = 'input_data_example_W3_O3_P10.json'
    input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'Data', 'InputData', input_case_name)
    # 基础调度
    env1 = FjspMaEnv({'train_or_eval_or_test': 'test', 'inputdata_json': input_path})
    env1.build_and_save_output_json = lambda output_path=None: FjspMaEnv.build_and_save_output_json(env1, output_path or 'Data/OutputData/output_EST_EET_' + input_case_name)
    makespan1, total_reward1 = est_eet_rule(env1, verbose=True)
    print(f"[基础规则] Makespan: {makespan1}, Total Reward: {total_reward1}")
    print("基础调度输出: Data/OutputData/output_EST_EET_" + input_case_name)

    # 优先级加权调度
    env2 = FjspMaEnv({'train_or_eval_or_test': 'test', 'inputdata_json': input_path})
    env2.build_and_save_output_json = lambda output_path=None: FjspMaEnv.build_and_save_output_json(env2, output_path or 'Data/OutputData/output_EST_EET_weighted_' + input_case_name)
    makespan2, total_reward2 = est_eet_rule_weighted(env2, alpha=0.7, verbose=True)
    print(f"[优先级加权规则] Makespan: {makespan2}, Total Reward: {total_reward2}")
    print("优先级加权调度输出: Data/OutputData/output_EST_EET_weighted_" + input_case_name)




