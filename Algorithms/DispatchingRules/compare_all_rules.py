import numpy as np
import os
import sys
import json
import time

# add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from Environment.dfjspt_env import FjspMaEnv
from dfjspt_rule1_EST_EET import est_eet_rule, est_eet_rule_weighted
from dfjspt_rule2_EST_SPT import est_spt_rule, est_spt_rule_weighted
from dfjspt_rule3_SPT_EET import spt_eet_rule, spt_eet_rule_weighted
from dfjspt_rule4_LPT_EET import lpt_eet_rule, lpt_eet_rule_weighted
from dfjspt_rule5_FIFO_LUM import fifo_lum_rule, fifo_lum_rule_weighted
from dfjspt_rule6_CR_EET import cr_eet_rule, cr_eet_rule_weighted
from dfjspt_rule7_MWKR_EET import mwkr_eet_rule, mwkr_eet_rule_weighted
from dfjspt_rule8_LWKR_SPT import lwkr_spt_rule, lwkr_spt_rule_weighted


def compare_dispatching_rules(input_case_name='input_test_1.json', verbose=False):
    """
    Compare all implemented dispatching rules on a given input case.
    """
    input_path = os.path.join(project_root, 'Data', 'InputData', input_case_name)
    
    try:
        with open(input_path, 'r') as f:
            input_data_json = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {input_path} not found.")
        return
    
    # Define all rules to test
    rules = [
        ("EST-EET", est_eet_rule),
        ("EST-EET (Weighted)", est_eet_rule_weighted),
        ("EST-SPT", est_spt_rule),
        ("EST-SPT (Weighted)", est_spt_rule_weighted),
        ("SPT-EET", spt_eet_rule),
        ("SPT-EET (Weighted)", spt_eet_rule_weighted),
        ("LPT-EET", lpt_eet_rule),
        ("LPT-EET (Weighted)", lpt_eet_rule_weighted),
        ("FIFO-LUM", fifo_lum_rule),
        ("FIFO-LUM (Weighted)", fifo_lum_rule_weighted),
        ("CR-EET", cr_eet_rule),
        ("CR-EET (Weighted)", cr_eet_rule_weighted),
        ("MWKR-EET", mwkr_eet_rule),
        ("MWKR-EET (Weighted)", mwkr_eet_rule_weighted),
        ("LWKR-SPT", lwkr_spt_rule),
        ("LWKR-SPT (Weighted)", lwkr_spt_rule_weighted),
    ]
    
    results = []
    
    print(f"\n{'='*80}")
    print(f"DISPATCHING RULES COMPARISON - {input_case_name}")
    print(f"{'='*80}")
    print(f"{'Rule Name':<25} {'Makespan':<12} {'Reward':<12} {'Time (s)':<10}")
    print(f"{'-'*80}")
    
    for rule_name, rule_func in rules:
        try:
            # Create fresh environment for each rule
            env = FjspMaEnv({'inputdata_json': input_data_json})
            
            # Measure execution time
            start_time = time.time()
            
            # Run the rule
            if "Weighted" in rule_name:
                makespan, total_reward = rule_func(env, alpha=0.7, verbose=verbose)
            else:
                makespan, total_reward = rule_func(env, verbose=verbose)
            
            execution_time = time.time() - start_time
            
            results.append({
                'rule': rule_name,
                'makespan': makespan,
                'reward': total_reward,
                'time': execution_time
            })
            
            print(f"{rule_name:<25} {makespan:<12.2f} {total_reward:<12.2f} {execution_time:<10.3f}")
            
        except Exception as e:
            print(f"{rule_name:<25} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10}")
            if verbose:
                print(f"  Error details: {str(e)}")
    
    print(f"{'-'*80}")
    
    # Find best results
    if results:
        best_makespan = min(results, key=lambda x: x['makespan'])
        best_reward = max(results, key=lambda x: x['reward'])
        fastest = min(results, key=lambda x: x['time'])
        
        print(f"\nBEST RESULTS:")
        print(f"Best Makespan: {best_makespan['rule']} ({best_makespan['makespan']:.2f})")
        print(f"Best Reward:   {best_reward['rule']} ({best_reward['reward']:.2f})")
        print(f"Fastest:       {fastest['rule']} ({fastest['time']:.3f}s)")
    
    return results


def run_multiple_cases():
    """
    Run comparison on multiple input cases if available.
    """
    input_dir = os.path.join(project_root, 'Data', 'InputData')
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    print(f"Found {len(input_files)} input files: {input_files}")
    
    all_results = {}
    
    for input_file in input_files:
        print(f"\n\nProcessing {input_file}...")
        try:
            results = compare_dispatching_rules(input_file, verbose=False)
            all_results[input_file] = results
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
    
    # Summary across all cases
    if all_results:
        print(f"\n{'='*80}")
        print("SUMMARY ACROSS ALL CASES")
        print(f"{'='*80}")
        
        # Calculate average performance for each rule
        rule_stats = {}
        for case_name, results in all_results.items():
            for result in results:
                rule_name = result['rule']
                if rule_name not in rule_stats:
                    rule_stats[rule_name] = {'makespans': [], 'rewards': [], 'times': []}
                rule_stats[rule_name]['makespans'].append(result['makespan'])
                rule_stats[rule_name]['rewards'].append(result['reward'])
                rule_stats[rule_name]['times'].append(result['time'])
        
        print(f"{'Rule Name':<25} {'Avg Makespan':<15} {'Avg Reward':<12} {'Avg Time':<10}")
        print(f"{'-'*80}")
        
        for rule_name, stats in rule_stats.items():
            avg_makespan = np.mean(stats['makespans'])
            avg_reward = np.mean(stats['rewards'])
            avg_time = np.mean(stats['times'])
            print(f"{rule_name:<25} {avg_makespan:<15.2f} {avg_reward:<12.2f} {avg_time:<10.3f}")


if __name__ == '__main__':
    # Test on single case
    compare_dispatching_rules('input_test_generated.json', verbose=True)
    
    # Test on all available cases
    # run_multiple_cases()