import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
# import random
import numpy as np

# Color palette for jobs/products (default)
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
]

def get_color_list(n, base_palette=COLORS, cmap_name='tab20'):
    if n <= len(base_palette):
        return base_palette[:n]
    # Use matplotlib colormap for more colors
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / n) for i in range(n)]

def plot_gantt_from_output(json_path, save_path=None, color_by='job'):
    with open(json_path, 'r') as f:
        data = json.load(f)
    result = data['scheduling_result']
    gantt = result['gantt_chart_data']['machine_timelines']
    makespan = result['schedule_summary']['total_makespan']
    schedule_start_time = float(result['schedule_summary']['schedule_start_time'])
    schedule_end_time = float(result['schedule_summary']['schedule_end_time'])

    # Collect all unique color keys
    color_keys = set()
    for machine in gantt:
        for op in machine['timeline_bars']:
            color_val = op[color_by + '_id'] if color_by+'_id' in op else op['job_id']
            color_keys.add(color_val)
    color_keys = sorted(list(color_keys))
    color_list = get_color_list(len(color_keys), cmap_name='tab20' if len(color_keys)<=20 else 'nipy_spectral')
    color_dict = {k: color_list[i] for i, k in enumerate(color_keys)}

    # Prepare plot
    fig, ax = plt.subplots(figsize=(16, max(6, len(gantt)*0.7)))
    yticks = []
    yticklabels = []
    legend_handles = {}
    for i, machine in enumerate(gantt):
        machine_id = machine['machine_id']
        yticks.append(i)
        yticklabels.append(str(machine_id))
        for op in machine['timeline_bars']:
            # Convert to relative time (hours since schedule_start_time)
            bar_start = op['start_time'] - schedule_start_time
            bar_end = op['end_time'] - schedule_start_time
            bar_len = bar_end - bar_start
            # label = f"{op['operation_id']}\nJob:{op['job_id']}\nProd:{op['product_id']}\nOrder:{op['order_id']}\n[{bar_start:.1f}-{bar_end:.1f}]h"
            label = " "
            color_val = op[color_by + '_id'] if color_by+'_id' in op else op['job_id']
            color = color_dict[color_val]
            rect = ax.barh(i, bar_len, left=bar_start, height=0.6, color=color, edgecolor='black', alpha=0.85)
            # Annotate
            ax.text(bar_start + bar_len/2, i, label, ha='center', va='center', fontsize=8, color='white', weight='bold')
            if color_val not in legend_handles:
                legend_handles[color_val] = mpatches.Patch(color=color, label=f"{color_by.capitalize()} {color_val}")

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=12)
    ax.set_xlabel('Time (hours since schedule start)', fontsize=14)
    ax.set_ylabel('Machine', fontsize=14)
    ax.set_title(f"Scheduling Gantt Chart\nMakespan: {makespan}h  Start: 0.0  End: {makespan:.1f}", fontsize=16, weight='bold')
    ax.set_xlim(0, makespan)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.legend(handles=list(legend_handles.values()), bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10, title=color_by.capitalize())
    plt.tight_layout()
    if save_path is None:
        save_path = os.path.splitext(json_path)[0] + '_gantt.png'
    plt.savefig(save_path, dpi=200)
    print(f"Gantt chart saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    json_path = os.path.join(
        os.path.dirname(__file__), 
        'output_data_GA_example.json'
    )
    plot_gantt_from_output(
        json_path=json_path,
        # color_by: ['job', 'order', 'product']
        color_by='order',
    )
    