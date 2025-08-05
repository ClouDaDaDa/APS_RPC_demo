import numpy as np
import os
import sys
import random

# add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from Environment.dfjspt_env import FjspMaEnv


def cr_eet_rule(env, verbose=False):
    """
    CR-EET rule: Select the job with smallest critical ratio (CR) and assign to the earliest end time machine (EET).
    Job selection: Choose job with smallest critical ratio (due_date - current_time) / remaining_processing_time
    Machine selection: Choose machine with earliest available time (earliest end time)
    """
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Job selection phase - CR (Critical Ratio)
        job_mask = obs['agent0']['action_mask']
        job_features = obs['agent0']['observation']
        available_jobs = np.where(job_mask == 1)[0]
        if len(available_jobs) == 0:
            job_action = 0  # fallback
        else:
            # Calculate critical ratio for each available job
            critical_ratios = []
            current_time = env.env_current_time
            
            for job_id in available_jobs:
                # Get job due date (if available, otherwise use a large value)
                if hasattr(env, 'job_due_date') and len(env.job_due_date) > job_id:
                    due_date = env.job_due_date[job_id]
                    if isinstance(due_date, str):
                        try:
                            import datetime
                            due_date = datetime.datetime.fromisoformat(due_date).timestamp()
                        except:
                            due_date = current_time + 1000  # fallback to large value
                else:
                    due_date = current_time + 1000  # fallback to large value
                
                # Get remaining processing time (column 7)
                remaining_time = job_features[job_id, 7]
                if remaining_time <= 0:
                    remaining_time = 1  # avoid division by zero
                
                # Calculate critical ratio
                time_to_due = due_date - current_time
                cr = time_to_due / remaining_time
                critical_ratios.append(cr)
            
            critical_ratios = np.array(critical_ratios)
            # Select job with minimum critical ratio (most critical)
            min_cr = np.min(critical_ratios)
            min_cr_jobs = available_jobs[critical_ratios == min_cr]
            job_action = random.choice(min_cr_jobs)
        
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


def cr_eet_rule_weighted(env, alpha=0.7, verbose=False):
    """
    Priority-weighted CR-EET rule: Job selection considers priority and critical ratio, machine assignment uses EET.
    alpha: Priority weight, 0~1, larger values favor priority, smaller values favor critical ratio.
    """
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Job selection phase - Weighted CR with priority
        job_mask = obs['agent0']['action_mask']
        job_features = obs['agent0']['observation']
        available_jobs = np.where(job_mask == 1)[0]
        if len(available_jobs) == 0:
            job_action = 0  # fallback
        else:
            priorities = np.array([env.job_priority[job_id] for job_id in available_jobs])
            
            # Calculate critical ratios
            critical_ratios = []
            current_time = env.env_current_time
            
            for job_id in available_jobs:
                # Get job due date
                if hasattr(env, 'job_due_date') and len(env.job_due_date) > job_id:
                    due_date = env.job_due_date[job_id]
                    if isinstance(due_date, str):
                        try:
                            import datetime
                            due_date = datetime.datetime.fromisoformat(due_date).timestamp()
                        except:
                            due_date = current_time + 1000
                else:
                    due_date = current_time + 1000
                
                remaining_time = job_features[job_id, 7]
                if remaining_time <= 0:
                    remaining_time = 1
                
                time_to_due = due_date - current_time
                cr = time_to_due / remaining_time
                critical_ratios.append(cr)
            
            critical_ratios = np.array(critical_ratios)
            
            # Normalize (higher priority and lower critical ratio are better)
            norm_priority = priorities / (priorities.max() if priorities.max() > 0 else 1)
            norm_cr = 1 - (critical_ratios - critical_ratios.min()) / (np.ptp(critical_ratios) + 1e-6)
            
            score = alpha * norm_priority + (1 - alpha) * norm_cr
            job_action = available_jobs[np.argmax(score)]
        
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
    env1.build_and_save_output_json = lambda output_path=None: FjspMaEnv.build_and_save_output_json(env1, output_path or 'Data/OutputData/output_CR_EET_' + input_case_name)
    makespan1, total_reward1 = cr_eet_rule(env1, verbose=True)
    print(f"[CR-EET Rule] Makespan: {makespan1}, Total Reward: {total_reward1}")
    print("CR-EET scheduling output: Data/OutputData/output_CR_EET_" + input_case_name)

    # Priority-weighted scheduling
    env2 = FjspMaEnv({'inputdata_json': input_data_json})
    env2.build_and_save_output_json = lambda output_path=None: FjspMaEnv.build_and_save_output_json(env2, output_path or 'Data/OutputData/output_CR_EET_weighted_' + input_case_name)
    makespan2, total_reward2 = cr_eet_rule_weighted(env2, alpha=0.7, verbose=True)
    print(f"[CR-EET Weighted Rule] Makespan: {makespan2}, Total Reward: {total_reward2}")
    print("CR-EET weighted scheduling output: Data/OutputData/output_CR_EET_weighted_" + input_case_name)