import numpy as np
import os
import sys
import random

# add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from Environment.dfjspt_env import FjspMaEnv


def est_spt_rule(env, verbose=False):
    """
    Basic rule: Select the earliest start time job (EST) and assign to the shortest processing time machine (SPT).
    """
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Job selection phase
        job_mask = obs['agent0']['action_mask']
        job_features = obs['agent0']['observation']
        available_jobs = np.where(job_mask == 1)[0]
        if len(available_jobs) == 0:
            job_action = 0  # fallback
        else:
            arrival_times = job_features[available_jobs, 2]
            min_arrival_time = np.min(arrival_times)
            # Find all jobs with the minimum arrival time
            min_time_jobs = available_jobs[arrival_times == min_arrival_time]
            # Randomly select from jobs with minimum arrival time
            job_action = random.choice(min_time_jobs)
        action = {'agent0': job_action}
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward['agent0']
        if terminated['__all__']:
            break
        # Machine selection phase
        machine_mask = obs['agent1']['action_mask']
        machine_features = obs['agent1']['observation']
        available_machines = np.where(machine_mask == 1)[0]
        if len(available_machines) == 0:
            machine_action = 0  # fallback
        else:
            # Select machine with shortest processing time (SPT)
            processing_times = machine_features[available_machines, 5]
            min_processing_time = np.min(processing_times)
            # Find all machines with the minimum processing time
            min_time_machines = available_machines[processing_times == min_processing_time]
            # Randomly select from machines with minimum processing time
            machine_action = random.choice(min_time_machines)
        action = {'agent1': machine_action}
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward['agent1']
        done = terminated['__all__']
    return env.final_makespan, total_reward

def est_spt_rule_weighted(env, alpha=0.7, verbose=False):
    """
    Priority-weighted rule: Job selection considers order priority with weight, machine assignment still uses SPT.
    alpha: Priority weight, 0~1, larger values favor priority, smaller values favor arrival time.
    """
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Job selection phase
        job_mask = obs['agent0']['action_mask']
        job_features = obs['agent0']['observation']
        available_jobs = np.where(job_mask == 1)[0]
        if len(available_jobs) == 0:
            job_action = 0  # fallback
        else:
            priorities = np.array([env.job_priority[job_id] for job_id in available_jobs])
            arrival_times = job_features[available_jobs, 2]
            # Normalize
            norm_priority = priorities / (priorities.max() if priorities.max() > 0 else 1)
            norm_arrival = (arrival_times - arrival_times.min()) / (np.ptp(arrival_times) + 1e-6)
            score = alpha * norm_priority - (1 - alpha) * norm_arrival
            # min_arrival_time = np.min(arrival_times)
            min_arrival_time = np.max(score)
            # Find all jobs with the minimum arrival time
            min_time_jobs = available_jobs[score == min_arrival_time]
            # Randomly select from jobs with minimum arrival time
            job_action = random.choice(min_time_jobs)
        action = {'agent0': job_action}
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward['agent0']
        if terminated['__all__']:
            break
        # Machine selection phase
        machine_mask = obs['agent1']['action_mask']
        machine_features = obs['agent1']['observation']
        available_machines = np.where(machine_mask == 1)[0]
        if len(available_machines) == 0:
            machine_action = 0  # fallback
        else:
            # Select machine with shortest processing time (SPT)
            processing_times = machine_features[available_machines, 5]
            min_processing_time = np.min(processing_times)
            # Find all machines with the minimum processing time
            min_time_machines = available_machines[processing_times == min_processing_time]
            # Randomly select from machines with minimum processing time
            machine_action = random.choice(min_time_machines)
        action = {'agent1': machine_action}
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward['agent1']
        done = terminated['__all__']
    return env.final_makespan, total_reward

if __name__ == '__main__':
    import json
    # input_case_name = 'input_test_1.json'
    input_case_name = 'input_test_generated.json'
    input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'Data', 'InputData', input_case_name)
    with open(input_path, 'r') as f:
        input_data_json = json.load(f)
    
    # Basic scheduling
    env1 = FjspMaEnv({'inputdata_json': input_data_json})
    env1.build_and_save_output_json = lambda output_path=None: FjspMaEnv.build_and_save_output_json(env1, output_path or 'Data/OutputData/output_EST_SPT_' + input_case_name)
    makespan1, total_reward1 = est_spt_rule(env1, verbose=True)
    print(f"[Basic Rule] Makespan: {makespan1}, Total Reward: {total_reward1}")
    print("Basic scheduling output: Data/OutputData/output_EST_SPT_" + input_case_name)

    # Priority-weighted scheduling
    env2 = FjspMaEnv({'inputdata_json': input_data_json})
    env2.build_and_save_output_json = lambda output_path=None: FjspMaEnv.build_and_save_output_json(env2, output_path or 'Data/OutputData/output_EST_SPT_weighted_' + input_case_name)
    makespan2, total_reward2 = est_spt_rule_weighted(env2, alpha=0.7, verbose=True)
    print(f"[Priority-weighted Rule] Makespan: {makespan2}, Total Reward: {total_reward2}")
    print("Priority-weighted scheduling output: Data/OutputData/output_EST_SPT_weighted_" + input_case_name)

