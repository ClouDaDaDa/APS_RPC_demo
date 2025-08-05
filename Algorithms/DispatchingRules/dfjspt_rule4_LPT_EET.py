import numpy as np
import os
import sys
import random

# add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from Environment.dfjspt_env import FjspMaEnv


def lpt_eet_rule(env, verbose=False):
    """
    LPT-EET rule: Select the longest processing time job (LPT) and assign to the earliest end time machine (EET).
    Job selection: Choose job with longest current operation processing time
    Machine selection: Choose machine with earliest available time (earliest end time)
    """
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Job selection phase - LPT (Longest Processing Time)
        job_mask = obs['agent0']['action_mask']
        job_features = obs['agent0']['observation']
        available_jobs = np.where(job_mask == 1)[0]
        if len(available_jobs) == 0:
            job_action = 0  # fallback
        else:
            # Use current operation processing time (column 6)
            processing_times = job_features[available_jobs, 6]
            # Filter out invalid processing times (-1)
            valid_jobs = available_jobs[processing_times > 0]
            if len(valid_jobs) == 0:
                job_action = random.choice(available_jobs)
            else:
                valid_processing_times = job_features[valid_jobs, 6]
                max_processing_time = np.max(valid_processing_times)
                # Find all jobs with the maximum processing time
                max_time_jobs = valid_jobs[valid_processing_times == max_processing_time]
                # Randomly select from jobs with maximum processing time
                job_action = random.choice(max_time_jobs)
        
        action = {'agent0': job_action}
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward['agent0']
        if terminated['__all__']:
            break
            
        # Machine selection phase - EET (Earliest End Time)
        machine_mask = obs['agent1']['action_mask']
        machine_features = obs['agent1']['observation']
        available_machines = np.where(machine_mask == 1)[0]
        if len(available_machines) == 0:
            machine_action = 0  # fallback
        else:
            # Use machine available time (column 3)
            idle_times = machine_features[available_machines, 3]
            machine_action = available_machines[np.argmin(idle_times)]
            
        action = {'agent1': machine_action}
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward['agent1']
        done = terminated['__all__']
    return env.final_makespan, total_reward


def lpt_eet_rule_weighted(env, alpha=0.7, verbose=False):
    """
    Priority-weighted LPT-EET rule: Job selection considers priority and processing time, machine assignment uses EET.
    alpha: Priority weight, 0~1, larger values favor priority, smaller values favor processing time.
    """
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Job selection phase - Weighted LPT with priority
        job_mask = obs['agent0']['action_mask']
        job_features = obs['agent0']['observation']
        available_jobs = np.where(job_mask == 1)[0]
        if len(available_jobs) == 0:
            job_action = 0  # fallback
        else:
            priorities = np.array([env.job_priority[job_id] for job_id in available_jobs])
            processing_times = job_features[available_jobs, 6]
            
            # Filter out invalid processing times
            valid_mask = processing_times > 0
            if np.any(valid_mask):
                valid_jobs = available_jobs[valid_mask]
                valid_priorities = priorities[valid_mask]
                valid_processing_times = processing_times[valid_mask]
                
                # Normalize (higher priority and longer processing time are better for LPT)
                norm_priority = valid_priorities / (valid_priorities.max() if valid_priorities.max() > 0 else 1)
                norm_processing = (valid_processing_times - valid_processing_times.min()) / (np.ptp(valid_processing_times) + 1e-6)
                
                score = alpha * norm_priority + (1 - alpha) * norm_processing
                job_action = valid_jobs[np.argmax(score)]
            else:
                job_action = random.choice(available_jobs)
        
        action = {'agent0': job_action}
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward['agent0']
        if terminated['__all__']:
            break
            
        # Machine selection phase - EET (Earliest End Time)
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
    import json
    input_case_name = 'input_test_1.json'
    input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'Data', 'InputData', input_case_name)
    with open(input_path, 'r') as f:
        input_data_json = json.load(f)
        
    # Basic scheduling
    env1 = FjspMaEnv({'inputdata_json': input_data_json})
    env1.build_and_save_output_json = lambda output_path=None: FjspMaEnv.build_and_save_output_json(env1, output_path or 'Data/OutputData/output_LPT_EET_' + input_case_name)
    makespan1, total_reward1 = lpt_eet_rule(env1, verbose=True)
    print(f"[LPT-EET Rule] Makespan: {makespan1}, Total Reward: {total_reward1}")
    print("LPT-EET scheduling output: Data/OutputData/output_LPT_EET_" + input_case_name)

    # Priority-weighted scheduling
    env2 = FjspMaEnv({'inputdata_json': input_data_json})
    env2.build_and_save_output_json = lambda output_path=None: FjspMaEnv.build_and_save_output_json(env2, output_path or 'Data/OutputData/output_LPT_EET_weighted_' + input_case_name)
    makespan2, total_reward2 = lpt_eet_rule_weighted(env2, alpha=0.7, verbose=True)
    print(f"[LPT-EET Weighted Rule] Makespan: {makespan2}, Total Reward: {total_reward2}")
    print("LPT-EET weighted scheduling output: Data/OutputData/output_LPT_EET_weighted_" + input_case_name)