from typing import List, Dict, Optional

class EligibleMachineData:
    def __init__(self, machine_id: str, standard_duration: float):
        self.machine_id = machine_id
        self.standard_duration = standard_duration

    @staticmethod
    def from_dict(data: dict):
        return EligibleMachineData(
            machine_id=data["machine_id"],
            standard_duration=float(data["standard_duration"])
        )

class OperationData:
    def __init__(self, operation_id: str, operation_sequence: int, process_workstation: str, eligible_machines: List[EligibleMachineData]):
        self.operation_id = operation_id
        self.operation_sequence = operation_sequence
        self.process_workstation = process_workstation
        self.eligible_machines = eligible_machines

    @staticmethod
    def from_dict(data: dict):
        return OperationData(
            operation_id=data["operation_id"],
            operation_sequence=int(data["operation_sequence"]),
            process_workstation=data["process_workstation"],
            eligible_machines=[EligibleMachineData.from_dict(em) for em in data["eligible_machines"]]
        )

class JobData:
    def __init__(self, job_id: str, product_id: str, order_id: str, operations: List[OperationData]):
        self.job_id = job_id
        self.product_id = product_id
        self.order_id = order_id
        self.operations = operations

    def __repr__(self):
        return f"JobData(job_id={self.job_id}, product_id={self.product_id}, order_id={self.order_id}, operations={self.operations})"

# Add method to ProductData to convert to jobs
class ProductData:
    def __init__(self, product_id: str, quantity: int, operations: List[OperationData]):
        self.product_id = product_id
        self.quantity = quantity
        self.operations = operations

    @staticmethod
    def from_dict(data: dict):
        return ProductData(
            product_id=data["product_id"],
            quantity=int(data["quantity"]),
            operations=[OperationData.from_dict(op) for op in data["operations"]]
        )

    def to_jobs(self, order_id: str, job_id_start: int = 0) -> List['JobData']:
        jobs = []
        for i in range(self.quantity):
            # Deep copy and renumber operation_sequence
            sorted_ops = sorted(self.operations, key=lambda op: op.operation_sequence)
            new_ops = []
            for idx, op in enumerate(sorted_ops, 1):
                # Create a new OperationData with updated operation_sequence
                new_op = OperationData(
                    operation_id=op.operation_id,
                    operation_sequence=idx,
                    process_workstation=op.process_workstation,
                    eligible_machines=op.eligible_machines
                )
                new_ops.append(new_op)
            job = JobData(
                job_id=f"{order_id}_{self.product_id}_{job_id_start + i + 1}",
                product_id=self.product_id,
                order_id=order_id,
                operations=new_ops
            )
            jobs.append(job)
        return jobs

# Add method to OrderData to convert all products to jobs
class OrderData:
    def __init__(self, order_id: str, order_priority: int, release_time: str, due_date: str, products: List[ProductData]):
        self.order_id = order_id
        self.order_priority = order_priority
        self.release_time = release_time
        self.due_date = due_date
        self.products = products

    @staticmethod
    def from_dict(data: dict):
        return OrderData(
            order_id=data["order_id"],
            order_priority=int(data["order_priority"]),
            release_time=data["release_time"],
            due_date=data["due_date"],
            products=[ProductData.from_dict(prod) for prod in data["products"]]
        )

    def to_jobs(self, job_id_start: int = 0) -> List['JobData']:
        jobs = []
        job_id_counter = job_id_start
        for product in self.products:
            product_jobs = product.to_jobs(self.order_id, job_id_start=job_id_counter)
            jobs.extend(product_jobs)
            job_id_counter += len(product_jobs)
        return jobs

# Add method to WorkOrderData to convert all orders to jobs
class WorkOrderData:
    def __init__(self, work_order_id: str, orders: List[OrderData]):
        self.work_order_id = work_order_id
        self.orders = orders

    @staticmethod
    def from_dict(data: dict):
        return WorkOrderData(
            work_order_id=data["work_order_id"],
            orders=[OrderData.from_dict(order) for order in data["orders"]]
        )

    def to_jobs(self) -> List['JobData']:
        jobs = []
        job_id_counter = 0
        for order in self.orders:
            order_jobs = order.to_jobs(job_id_start=job_id_counter)
            jobs.extend(order_jobs)
            job_id_counter += len(order_jobs)
        return jobs

class UnavailablePeriodData:
    def __init__(self, start_time: str, end_time: str, shutdown_type: str):
        self.start_time = start_time
        self.end_time = end_time
        self.shutdown_type = shutdown_type

    @staticmethod
    def from_dict(data: dict):
        return UnavailablePeriodData(
            start_time=data["start_time"],
            end_time=data["end_time"],
            shutdown_type=data["shutdown_type"]
        )

class MachineAvailabilityData:
    def __init__(self, status: str, unavailable_periods: List[UnavailablePeriodData]):
        self.status = status
        self.unavailable_periods = unavailable_periods

    @staticmethod
    def from_dict(data: dict):
        return MachineAvailabilityData(
            status=data["status"],
            unavailable_periods=[UnavailablePeriodData.from_dict(up) for up in data["unavailable_periods"]]
        )

class MachineCapacityData:
    def __init__(self, max_parallel_jobs: int):
        self.max_parallel_jobs = max_parallel_jobs

    @staticmethod
    def from_dict(data: dict):
        return MachineCapacityData(
            max_parallel_jobs=int(data["max_parallel_jobs"])
        )

class MachineData:
    def __init__(self, machine_id: str, capacity: MachineCapacityData, availability: MachineAvailabilityData):
        self.machine_id = machine_id
        self.capacity = capacity
        self.availability = availability

    @staticmethod
    def from_dict(data: dict):
        return MachineData(
            machine_id=data["machine_id"],
            capacity=MachineCapacityData.from_dict(data["capacity"]),
            availability=MachineAvailabilityData.from_dict(data["availability"])
        )

class BufferCapacityData:
    def __init__(self, buffer_capacity: int):
        self.buffer_capacity = buffer_capacity

    @staticmethod
    def from_dict(data: dict):
        return BufferCapacityData(
            buffer_capacity=int(data["buffer_capacity"])
        )

class WorkstationData:
    def __init__(self, workstation_id: str, capacity: BufferCapacityData, machines: List[MachineData]):
        self.workstation_id = workstation_id
        self.capacity = capacity
        self.machines = machines

    @staticmethod
    def from_dict(data: dict):
        return WorkstationData(
            workstation_id=data["workstation_id"],
            capacity=BufferCapacityData.from_dict(data["capacity"]),
            machines=[MachineData.from_dict(m) for m in data["machines"]]
        )

class WorkshopData:
    def __init__(self, workshop_id: str, workstations: List[WorkstationData]):
        self.workshop_id = workshop_id
        self.workstations = workstations

    @staticmethod
    def from_dict(data: dict):
        return WorkshopData(
            workshop_id=data["workshop_id"],
            workstations=[WorkstationData.from_dict(ws) for ws in data["workstations"]]
        )

class SchedulingInstanceData:
    def __init__(self, instance_id: str, workshop: WorkshopData, work_order: WorkOrderData):
        self.instance_id = instance_id
        self.workshop = workshop
        self.work_order = work_order

    @staticmethod
    def from_dict(data: dict):
        return SchedulingInstanceData(
            instance_id=data["instance_id"],
            workshop=WorkshopData.from_dict(data["workshop"]),
            work_order=WorkOrderData.from_dict(data["work_order"])
        ) 