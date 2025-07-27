import numpy as np
import os
import sys

# add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from Environment.dfjspt_env import FjspMaEnv


def est_spt_rule(env, verbose=False):
    """
    启发式规则：每次选择最早可开始作业（EST），分配最短加工时间机器（SPT）。
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
            # 选择加工时间最短的机器（SPT）
            processing_times = machine_features[available_machines, 5]
            machine_action = available_machines[np.argmin(processing_times)]
        action = {'agent1': machine_action}
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward['agent1']
        done = terminated['__all__']
        # if verbose:
        #     print(f"Job: {job_action}, Machine: {machine_action}, Reward: {reward}, Done: {done}")
    return env.final_makespan, total_reward

if __name__ == '__main__':
    # Use the generated example input file
    input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'Data', 'InputData', 'input_data_example_W3_O3_P10.json')
    env = FjspMaEnv({'train_or_eval_or_test': 'test', 'inputdata_json': input_path})
    # Patch output path for this rule
    env.build_and_save_output_json = lambda output_path=None: FjspMaEnv.build_and_save_output_json(env, output_path or 'Data/OutputData/output_data_EST_SPT_example.json')
    makespan, total_reward = est_spt_rule(env, verbose=True)
    print(f"Test finished. Makespan: {makespan}, Total Reward: {total_reward}")
    print("Output JSON should be in Data/OutputData/output_data_EST_SPT_example.json or similar.")

