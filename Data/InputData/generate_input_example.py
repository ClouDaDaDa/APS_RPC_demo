import json
import random
import datetime
import os

def generate_input_example(n_workstations=3, n_orders=3, min_products=2, max_products=4, output_dir=None):
    # Helper to generate timestamps
    def random_timestamp(start_days=0, end_days=10):
        base = datetime.datetime.now() + datetime.timedelta(days=start_days)
        delta = datetime.timedelta(days=random.randint(0, end_days))
        return (base + delta).isoformat()

    # Generate machines
    def gen_machine(idx):
        return {
            "machine_id": f"M{idx}",
            "capacity": {"max_parallel_jobs": 1},
            "availability": {
                "status": "available",
                "unavailable_periods": []
            }
        }

    # Generate workstations
    def gen_workstation(idx, machine_start_idx):
        n_machines = random.randint(2, 4)
        machines = [gen_machine(machine_start_idx + i) for i in range(n_machines)]
        return {
            "workstation_id": f"WS{idx}",
            "capacity": {"buffer_capacity": random.choice([500, 1000, 2000])},
            "machines": machines
        }, machine_start_idx + n_machines, machines

    # --- Main structure ---
    workstations = []
    machine_ids = []
    ws_ids = []
    ws_to_machine_ids = {}
    machine_idx = 1
    all_ws_machines = []
    for ws_idx in range(n_workstations):
        ws, machine_idx, machines = gen_workstation(ws_idx+1, machine_idx)
        workstations.append(ws)
        ws_ids.append(ws["workstation_id"])
        ids = [m["machine_id"] for m in machines]
        machine_ids.extend(ids)
        ws_to_machine_ids[ws["workstation_id"]] = ids
        all_ws_machines.extend(machines)

    # Generate eligible machines for an operation (from a specific workstation)
    def gen_eligible_machines(process_ws_id):
        ids = ws_to_machine_ids[process_ws_id]
        k = random.randint(1, min(2, len(ids)))
        eligible = random.sample(ids, k=k)
        return [{
            "machine_id": m_id,
            "standard_duration": round(random.uniform(1.0, 8.0), 1)
        } for m_id in eligible]

    # Generate operations for a product
    def gen_operations(ws_ids):
        n_ops = random.randint(2, 4)
        ops = []
        for i in range(n_ops):
            process_ws = random.choice(ws_ids)
            ops.append({
                "operation_id": f"OP{i+1}",
                "operation_sequence": i+1,
                "process_workstation": process_ws,
                "eligible_machines": gen_eligible_machines(process_ws)
            })
        return ops

    # Generate products
    def gen_product(idx, ws_ids):
        return {
            "product_id": f"P{idx}",
            "quantity": random.randint(2, 6),
            "operations": gen_operations(ws_ids)
        }

    # Generate orders
    def gen_order(idx, ws_ids):
        n_products = random.randint(min_products, max_products)
        products = [gen_product(i, ws_ids) for i in range(n_products)]
        return {
            "order_id": f"O{idx}",
            "order_priority": random.randint(1, 5),
            "release_time": random_timestamp(-2, 2),
            "due_date": random_timestamp(5, 15),
            "products": products
        }

    orders = [gen_order(i+1, ws_ids) for i in range(n_orders)]
    # total_products = sum(len(order["products"]) for order in orders)
    total_products = 0
    for order in orders:
        total_products += sum(product["quantity"] for product in order["products"])

    data = {
        "scheduling_instance": {
            "instance_id": f"EXAMPLE_W{n_workstations}_O{n_orders}_P{total_products}",
            "workshop": {
                "workshop_id": "WSHOP1",
                "workstations": workstations
            },
            "work_order": {
                "work_order_id": "WO1",
                "orders": orders
            }
        }
    }

    # Determine output path
    if output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(project_root, "Data", "InputData")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"input_data_example_W{n_workstations}_O{n_orders}_P{total_products}.json"
    )

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Example input data saved to {output_path}")

def generate_input_example_new_format(n_workstations=2, n_orders=2, min_products=1, max_products=2, output_dir=None):
    import uuid
    def random_timestamp(start_days=0, end_days=10):
        base = datetime.datetime.now() + datetime.timedelta(days=start_days)
        delta = datetime.timedelta(days=random.randint(0, end_days))
        return (base + delta).strftime("%Y-%m-%d %H:%M:%S")

    # Generate workstations and machines
    machine_id_counter = 1
    workstations = []
    all_machine_ids = []
    for ws_idx in range(n_workstations):
        n_machines = random.randint(2, 3)
        machine_ids = list(range(machine_id_counter, machine_id_counter + n_machines))
        workstations.append({
            "id": ws_idx + 1,
            "machineIds": machine_ids
        })
        all_machine_ids.extend(machine_ids)
        machine_id_counter += n_machines

    # Generate workTasks
    work_tasks = []
    for order_idx in range(n_orders):
        n_products = random.randint(min_products, max_products)
        products = []
        for prod_idx in range(n_products):
            n_ops = random.randint(2, 3)
            operations = []
            for op_idx in range(n_ops):
                ws = random.choice(workstations)
                eligible_machine_ids = ws["machineIds"]
                n_eligible = random.randint(1, len(eligible_machine_ids))
                machines = []
                for m_id in random.sample(eligible_machine_ids, n_eligible):
                    machines.append({
                        "id": random.randint(100, 999),
                        "machineId": str(m_id),
                        "operationId": str(uuid.uuid4()),
                        "operationSequence": op_idx + 1,
                        "standardDuration": round(random.uniform(1.0, 8.0), 2),
                        "createTime": random_timestamp(-2, 2)
                    })
                operations.append({
                    "id": random.randint(50, 99),
                    "operationId": str(uuid.uuid4()),
                    "operationSequence": op_idx + 1,
                    "machines": machines,
                    "workstationIdList": [ws["id"]],
                    "productIds": [prod_idx + 1],
                    "createTime": random_timestamp(-2, 2)
                })
            products.append({
                "productId": prod_idx + 1,
                "productName": f"产品{prod_idx + 1}",
                "quantity": random.randint(1, 5),
                "operations": operations
            })
        work_tasks.append({
            "id": order_idx + 1,
            "orderPriority": random.randint(1, 5),
            "plannedStart": random_timestamp(-5, 0),
            "plannedEnd": random_timestamp(1, 10),
            "actualStart": random_timestamp(-5, 0),
            "actualEnd": random_timestamp(1, 10),
            "priority": random.randint(1, 5),
            "products": products,
            "status": "0",
            "taskId": str(random.randint(10, 99)),
            "title": f"普通工单{order_idx + 1}",
            "assigneeId": random.randint(100, 999),
            "description": "普通工单",
            "createTime": random_timestamp(-10, 0)
        })

    data = {
        "schedulingInstance": {
            "workStation": workstations,
            "workTasks": work_tasks
        }
    }

    if output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(project_root, "Data", "InputData")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "input_test_generated.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"New format input data saved to {output_path}")

# if __name__ == "__main__":
#     generate_input_example(
#         n_workstations=3, 
#         n_orders=4, 
#         min_products=2, 
#         max_products=5
#     )

if __name__ == "__main__":
    generate_input_example_new_format(
        n_workstations=2,
        n_orders=2,
        min_products=1,
        max_products=2
    ) 