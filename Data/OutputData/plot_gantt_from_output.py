import json
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.patches as mpatches

def plot_order_gantt(workTasks, save_path=None):
    # 绘制订单整体Gantt
    fig, ax = plt.subplots(figsize=(12, max(4, len(workTasks)*0.7)))
    yticks = []
    yticklabels = []
    for i, order in enumerate(sorted(workTasks, key=lambda x: str(x['id']))):
        yticks.append(i)
        yticklabels.append(str(order['id']))
        start = order['plannedStart']
        end = order['plannedEnd']
        ax.barh(i, end-start, left=start, height=0.6, color='#1f77b4', edgecolor='black', alpha=0.85)
        ax.text((start+end)/2, i, f"{start:.1f}-{end:.1f}", ha='center', va='center', color='white', fontsize=10)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=12)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Order', fontsize=14)
    ax.set_title('Order-level Gantt Chart', fontsize=16, weight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path is None:
        save_path = "order_gantt.png"
    plt.savefig(save_path, dpi=200)
    print(f"Order Gantt chart saved to {save_path}")
    plt.show()

def plot_machine_gantt(machines, save_path=None):
    # Collect all unique order ids
    order_ids = set()
    for machine in machines:
        for task in machine['workTasks']:
            order_ids.add(str(task['id']))
    order_ids = sorted(list(order_ids))
    # Assign a color to each order
    color_list = plt.get_cmap('tab20').colors if len(order_ids) <= 20 else plt.get_cmap('nipy_spectral')(np.linspace(0, 1, len(order_ids)))
    color_dict = {oid: color_list[i % len(color_list)] for i, oid in enumerate(order_ids)}

    fig, ax = plt.subplots(figsize=(14, max(4, len(machines)*0.7)))
    yticks = []
    yticklabels = []
    legend_handles = {}
    for i, machine in enumerate(sorted(machines, key=lambda x: str(x['id']))):
        yticks.append(i)
        yticklabels.append(str(machine['id']))
        for task in machine['workTasks']:
            start = task['plannedStart']
            end = task['plannedEnd']
            order_id = str(task['id'])
            color = color_dict[order_id]
            ax.barh(i, end-start, left=start, height=0.6, color=color, edgecolor='black', alpha=0.85)
            ax.text((start+end)/2, i, f"Order {order_id}", ha='center', va='center', color='white', fontsize=9)
            if order_id not in legend_handles:
                legend_handles[order_id] = mpatches.Patch(color=color, label=f"Order {order_id}")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=12)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Machine', fontsize=14)
    ax.set_title('Machine-level Gantt Chart', fontsize=16, weight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.legend(handles=list(legend_handles.values()), bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10, title='Order')
    plt.tight_layout()
    if save_path is None:
        save_path = "machine_gantt.png"
    plt.savefig(save_path, dpi=200)
    print(f"Machine Gantt chart saved to {save_path}")
    plt.show()

def plot_gantt_from_output(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    workTasks = data['workTasks']
    machines = data['machines']
    # 绘制订单Gantt
    plot_order_gantt(workTasks, save_path=os.path.splitext(json_path)[0] + '_order_gantt.png')
    # 绘制机器Gantt
    plot_machine_gantt(machines, save_path=os.path.splitext(json_path)[0] + '_machine_gantt.png')

if __name__ == '__main__':
    json_path = os.path.join(
        os.path.dirname(__file__),
        'output_EST_EET_weighted_input_test_1.json'
        # 'output_EST_EET_weighted_input_test_generated.json'
        # 'output_EST_SPT_weighted_input_test_1.json'
    )
    plot_gantt_from_output(json_path)