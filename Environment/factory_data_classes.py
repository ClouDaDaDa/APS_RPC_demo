from typing import List, Dict, Optional


class OperationData:
    def __init__(self, operation_id, operation_sequence: int, process_workstation, standard_duration: float):
        self.operation_id = operation_id
        self.operation_sequence = operation_sequence
        self.process_workstation = process_workstation
        self.standard_duration = standard_duration

    @staticmethod
    def from_dict(data: dict):
        return OperationData(
            # operation_id=data["id"],
            operation_id=int(data["operationSequence"] - 1),
            operation_sequence=data["operationSequence"],
            process_workstation=data["workstationIdList"][0],
            standard_duration=data["standardTime"],
        )

class JobData:
    def __init__(self, job_id, product_id, order_id, operations: List[OperationData]):
        self.job_id = job_id
        self.product_id = product_id
        self.order_id = order_id
        self.operations = operations

    def __repr__(self):
        return f"JobData(job_id={self.job_id}, product_id={self.product_id}, order_id={self.order_id}, operations={self.operations})"


class ProductTypeData:
    def __init__(self, product_id: str, operations: List[OperationData]):
        self.product_id = product_id
        self.operations = operations
    
    @staticmethod
    def from_dict(product_id: str, operations: List[dict]):
        operations = [OperationData.from_dict(op) for op in operations]
        sorted_ops = sorted(operations, key=lambda op: op.operation_sequence)
        new_ops = []
        for idx, op in enumerate(sorted_ops, 1):
            new_op = OperationData(
                operation_id=op.operation_id,
                operation_sequence=idx,
                process_workstation=op.process_workstation,
                standard_duration=op.standard_duration,
            )
            new_ops.append(new_op)
        return ProductTypeData(
            product_id=product_id,
            operations=new_ops
        )
    
    def to_jobs(self, order_id: str, job_id_start: int, quantity: int) -> List['JobData']:
        jobs = []
        for i in range(quantity):
            sorted_ops = sorted(self.operations, key=lambda op: op.operation_sequence)
            new_ops = []
            for idx, op in enumerate(sorted_ops, 1):
                new_op = OperationData(
                    operation_id=op.operation_id,
                    operation_sequence=idx,
                    process_workstation=op.process_workstation,
                    standard_duration=op.standard_duration,
                )
                new_ops.append(new_op)
            job = JobData(
                job_id=job_id_start + i,
                product_id=self.product_id,
                order_id=order_id,
                operations=new_ops
            )
            jobs.append(job)
        return jobs


class ProductData:
    def __init__(self, product_id: str, quantity: int):
        self.product_id = product_id
        self.quantity = quantity


# Add method to OrderData to convert all products to jobs
class OrderData:
    def __init__(self, order_id: str, order_priority: int, products: List[ProductData], release_time=0, due_date=1e8):
        self.order_id = order_id
        self.order_priority = order_priority
        self.release_time = release_time
        self.due_date = due_date
        self.products = products

    @staticmethod
    def from_dict(data: dict):
        # products = []
        # for prod in data["products"]:
        #     # product_type = next((pt for pt in product_types if pt.product_id == prod["productId"]), None)
        #     # if product_type is None:
        #     #     raise ValueError(f"Product type {prod['productId']} not found")
        #     products.append(ProductData(prod["productId"], prod["quantity"]))
        return OrderData(
            order_id=data["id"],
            order_priority=int(data["orderPriority"]),
            products=[ProductData(prod["productId"], prod["quantity"]) for prod in data["products"]] 
        )

    def to_jobs(self, job_id_start: int, product_types: List[ProductTypeData]) -> List['JobData']:
        jobs = []
        job_id_counter = job_id_start
        for product in self.products:
            product_type = next((pt for pt in product_types if pt.product_id == str(product.product_id)), None)
            if product_type is None:
                raise ValueError(f"Product type {product.product_id} not found")
            product_jobs = product_type.to_jobs(self.order_id, job_id_start=job_id_counter, quantity=product.quantity)
            jobs.extend(product_jobs)
            job_id_counter += len(product_jobs)
        return jobs

# Add method to WorkOrderData to convert all orders to jobs
class WorkTaskData:
    def __init__(self, orders: List[OrderData]):
        self.orders = orders

    @staticmethod
    def from_dict(data: dict):
        return WorkTaskData(
            orders=[OrderData.from_dict(order) for order in data]
        )

    def to_jobs(self, product_types: List[ProductTypeData]) -> List['JobData']:
        jobs = []
        job_id_counter = 0
        for order in self.orders:
            order_jobs = order.to_jobs(job_id_start=job_id_counter, product_types=product_types)
            jobs.extend(order_jobs)
            job_id_counter += len(order_jobs)
        return jobs


class MachineData:
    def __init__(self, machine_id: str):
        self.machine_id = machine_id


class WorkstationData:
    def __init__(self, workstation_id: str, machines: List[MachineData]):
        self.workstation_id = workstation_id
        self.machines = machines

    @staticmethod
    def from_dict(data: dict):
        return WorkstationData(
            workstation_id=data["id"],
            machines=[MachineData(machine_id=m) for m in data["machineIds"]]
        )

class WorkshopData:
    def __init__(self, workstations: List[WorkstationData]):
        self.workstations = workstations

    @staticmethod
    def from_dict(data: dict):
        return WorkshopData(
            workstations=[WorkstationData.from_dict(ws) for ws in data]
        )

class SchedulingInstanceData:
    def __init__(self, workshop: WorkshopData, work_tasks: WorkTaskData, product_types: List[ProductTypeData]):
        self.workshop = workshop
        self.work_tasks = work_tasks
        self.product_types = product_types

    @staticmethod
    def from_dict(data: dict):
        return SchedulingInstanceData(
            workshop=WorkshopData.from_dict(data["workStation"]),
            work_tasks=WorkTaskData.from_dict(data["workTasks"]),
            product_types=[ProductTypeData.from_dict(prod_id, ops) for prod_id, ops in data["operation"].items()]
        ) 
    


