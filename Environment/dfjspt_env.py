import os
import gymnasium as gym
from typing import List, Tuple, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from Environment import dfjspt_params
from Environment.factory_data_classes import JobData
from Environment.parse_inputdata_to_factory import parse_inputdata_to_factory
import json
import datetime

class MultiAgentEnv(gym.Env):
    """An environment that hosts multiple independent agents.

    Agents are identified by AgentIDs (string).
    """

    observation_spaces = None
    action_spaces = None

    # All agents currently active in the environment. This attribute may change during
    # the lifetime of the env or even during an individual episode.
    agents = []
    # All agents that may appear in the environment, ever.
    # This attribute should not be changed during the lifetime of this env.
    possible_agents = []

    observation_space: Optional[gym.Space] = None
    action_space: Optional[gym.Space] = None

    def __init__(self):
        super().__init__()

        # @OldAPIStack
        if not hasattr(self, "_agent_ids"):
            self._agent_ids = set()

        # If these important attributes are not set, try to infer them.
        if not self.agents:
            self.agents = list(self._agent_ids)
        if not self.possible_agents:
            self.possible_agents = self.agents.copy()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Resets the env and returns observations from ready agents.

        Args:
            seed: An optional seed to use for the new episode.

        Returns:
            New observations for each ready agent.

        .. testcode::
            :skipif: True

            from ray.rllib.env.multi_agent_env import MultiAgentEnv
            class MyMultiAgentEnv(MultiAgentEnv):
                # Define your env here.
            env = MyMultiAgentEnv()
            obs, infos = env.reset(seed=42, options={})
            print(obs)

        .. testoutput::

            {
                "car_0": [2.4, 1.6],
                "car_1": [3.4, -3.2],
                "traffic_light_1": [0, 3, 5, 1],
            }
        """
        # Call super's `reset()` method to (maybe) set the given `seed`.
        super().reset(seed=seed, options=options)

    def step(
        self, action_dict
    ):
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns:
            Tuple containing 1) new observations for
            each ready agent, 2) reward values for each ready agent. If
            the episode is just started, the value will be None.
            3) Terminated values for each ready agent. The special key
            "__all__" (required) is used to indicate env termination.
            4) Truncated values for each ready agent.
            5) Info values for each agent id (may be empty dicts).

        .. testcode::
            :skipif: True

            env = ...
            obs, rewards, terminateds, truncateds, infos = env.step(action_dict={
                "car_0": 1, "car_1": 0, "traffic_light_1": 2,
            })
            print(rewards)

            print(terminateds)

            print(infos)

        .. testoutput::

            {
                "car_0": 3,
                "car_1": -1,
                "traffic_light_1": 0,
            }
            {
                "car_0": False,    # car_0 is still running
                "car_1": True,     # car_1 is terminated
                "__all__": False,  # the env is not terminated
            }
            {
                "car_0": {},  # info for car_0
                "car_1": {},  # info for car_1
            }

        """
        raise NotImplementedError

    def render(self) -> None:
        """Tries to render the environment."""

        # By default, do nothing.
        pass

    def get_observation_space(self, agent_id) -> gym.Space:
        if self.observation_spaces is not None:
            return self.observation_spaces[agent_id]

        if (
            isinstance(self.observation_space, gym.spaces.Dict)
            and agent_id in self.observation_space.spaces
        ):
            return self.observation_space[agent_id]
        # `self.observation_space` is not a `gym.spaces.Dict` OR doesn't contain
        # `agent_id` -> The defined space is most likely meant to be the space
        # for all agents.
        else:
            return self.observation_space

    def get_action_space(self, agent_id) -> gym.Space:
        if self.action_spaces is not None:
            return self.action_spaces[agent_id]

        if (
            isinstance(self.action_space, gym.spaces.Dict)
            and agent_id in self.action_space.spaces
        ):
            return self.action_space[agent_id]
        # `self.action_space` is not a `gym.spaces.Dict` OR doesn't contain
        # `agent_id` -> The defined space is most likely meant to be the space
        # for all agents.
        else:
            return self.action_space

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        return len(self.possible_agents)


class FjspMaEnv(MultiAgentEnv):
    """
    Environment for Flexible Job-shop Scheduling Problem (FJSP/JSP)
    """

    def __init__(self, env_config):
        super().__init__()

        self.scheduling_instance = None
        self.jobs: List[JobData] = []
        self.machine_info = []
        self.machine_id_to_index = {}
        self.job_arrival_time = []
        self.job_due_date = []
        self.n_operations_for_jobs = []
        inputdata_json = env_config.get("inputdata_json", None)
        if inputdata_json is None:
            raise ValueError("FjspMaEnv requires 'inputdata_json' in env_config to load the scheduling instance input data.")
        self.inputdata_json = inputdata_json
        self.load_from_inputdata(inputdata_json)

        # ------ instance size parameters ------ #
        self.n_jobs = len(self.jobs)
        self.n_operations_for_jobs = [len(job.operations) for job in self.jobs]
        self.max_n_operations = max(self.n_operations_for_jobs) if self.n_operations_for_jobs else 0
        self.min_n_operations = min(self.n_operations_for_jobs) if self.n_operations_for_jobs else 0
        self.n_machines = len(self.machine_info)
        self.n_total_tasks = sum(self.n_operations_for_jobs) + 2
        self.n_total_nodes = self.n_total_tasks + self.n_machines
        self.n_processing_tasks = None
        self.n_job_features = 8
        self.n_machine_features = 7

        self.Graph = None
        self.source_task = -1
        self.sink_task = self.n_total_tasks - 2

        self.reward_this_step = None
        self.stage = None
        self.chosen_job = -1
        self.chosen_machine = -1
        self.operation_id = -1
        self.prcs_task_id = -1
        self.perform_left_shift_if_possible = dfjspt_params.perform_left_shift_if_possible
        self.schedule_done = False

        self.job_features = None
        self.machine_features = None
        self.job_action_mask = None
        self.machine_action_mask = None
        self.env_current_time = 0
        self.final_makespan = 0
        self.prev_cmax = None
        self.curr_cmax = None
        self.result_start_time_for_jobs = None
        self.result_finish_time_for_jobs = None
        self.current_instance_id = 0
        self.global_schedule = None

        # 'routes' of the machines. indicates in which order a machine processes tasks
        self.machine_routes = None
        self.machine_quality = np.ones((1, self.n_machines), dtype=float)

        # agent0: job selection agent
        # agent1: machine selection agent
        self.agents = self.possible_agents = ["agent0", "agent1"]
        self._agent_ids = set(self.agents)
        self.terminateds = set()
        self.truncateds = set()

        self.observation_space = gym.spaces.Dict({
            "agent0": gym.spaces.Dict({
                "action_mask": gym.spaces.Box(0, 1,
                                   shape=(self.n_jobs,),
                                   dtype=np.int64),
                "observation": gym.spaces.Box(-1, 1e8,
                                   shape=(self.n_jobs, self.n_job_features),
                                   dtype=np.float64),
            }),

            "agent1": gym.spaces.Dict({
                "action_mask": gym.spaces.Box(0, 1,
                                   shape=(self.n_machines,),
                                   dtype=np.int64),
                "observation": gym.spaces.Box(-1, 1e8,
                                   shape=(self.n_machines, self.n_machine_features),
                                   dtype=np.float64),
            })
        })

        self.action_space = gym.spaces.Dict({
            "agent0": gym.spaces.Discrete(self.n_jobs),
            "agent1": gym.spaces.Discrete(self.n_machines)
        })

        self.resetted = False

    def _get_info(self):
        if self.stage == 0:
            return {
                "agent0": {
                    "current_stage": self.stage
                },
            }
        else:
            return {
                "agent1": {
                    "current_stage": self.stage
                },
            }

    def _get_obs(self):
        if self.stage == 0:
            return {
                "agent0": {
                    "action_mask": self.job_action_mask,
                    "observation": self.job_features,
                },
            }
        else:
            return {
                "agent1": {
                    "action_mask": self.machine_action_mask,
                    "observation": self.machine_features,
                },
            }

    def load_from_inputdata(self, json_path):
        self.scheduling_instance = parse_inputdata_to_factory(json_path)
        # Extract jobs from work_order
        self.jobs = self.scheduling_instance.work_order.to_jobs()
        self.n_jobs = len(self.jobs)
        self.n_operations_for_jobs = [len(job.operations) for job in self.jobs]
        # For arrival/due times, use order-level info for all jobs in that order
        order_id_to_times = {}
        order_id_to_priority = {}
        for order in self.scheduling_instance.work_order.orders:
            order_id_to_times[order.order_id] = {
                "release_time": order.release_time,
                "due_date": order.due_date
            }
            order_id_to_priority[order.order_id] = order.order_priority
        self.job_arrival_time = [order_id_to_times[job.order_id]["release_time"] for job in self.jobs]
        self.job_due_date = [order_id_to_times[job.order_id]["due_date"] for job in self.jobs]
        # Store job priority for each job (for weighted scheduling)
        self.job_priority = [order_id_to_priority[job.order_id] for job in self.jobs]
        # Machine info: flatten all machines in all workstations
        self.machine_info = []
        for ws in self.scheduling_instance.workshop.workstations:
            for m in ws.machines:
                self.machine_info.append(m)
        self.n_machines = len(self.machine_info)
        self.machine_id_to_index = {m.machine_id: idx for idx, m in enumerate(self.machine_info)}

    def reset(self, seed=None, options=None):
        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()
        super().reset(seed=seed)
        assert self.scheduling_instance is not None, "Scheduling instance must be loaded before reset."

        # Use loaded scheduling_instance to initialize all arrays
        self.n_jobs = len(self.jobs)
        self.n_operations_for_jobs = [len(job.operations) for job in self.jobs]
        self.max_n_operations = max(self.n_operations_for_jobs) if self.n_operations_for_jobs else 0
        self.min_n_operations = min(self.n_operations_for_jobs) if self.n_operations_for_jobs else 0
        self.n_machines = len(self.machine_info)
        self.n_total_tasks = sum(self.n_operations_for_jobs) + 2
        self.n_total_nodes = self.n_total_tasks + self.n_machines

        # Convert all job_arrival_time to float timestamps for later use
        self.job_arrival_time_float = []
        for arrival in self.job_arrival_time:
            if isinstance(arrival, (int, float)):
                self.job_arrival_time_float.append(float(arrival))
            elif isinstance(arrival, str):
                try:
                    self.job_arrival_time_float.append(datetime.datetime.fromisoformat(arrival).timestamp())
                except Exception:
                    self.job_arrival_time_float.append(0.0)
            else:
                self.job_arrival_time_float.append(0.0)

        # Job and machine features arrays
        self.job_features = np.zeros((self.n_jobs, self.n_job_features), dtype=float)
        for job_id in range(self.n_jobs):
            self.job_features[job_id, 0] = job_id
        # Example: fill in arrival time, due date, etc. (customize as needed)
        for job_id in range(self.n_jobs):
            self.job_features[job_id, 2] = self.job_arrival_time_float[job_id]
            self.job_features[job_id, 3] = self.n_machines
        self.job_features[self.n_jobs:, 4] = 1
        # Optionally fill in more features as needed

        self.machine_features = np.zeros((self.n_machines, self.n_machine_features), dtype=float)
        for machine_id in range(self.n_machines):
            self.machine_features[machine_id, 0] = machine_id
        # Optionally fill in more features as needed
        self.machine_features[:self.n_machines, 5:] = -1

        # Action masks
        self.job_action_mask = np.zeros((self.n_jobs,), dtype=int)
        for job_id in range(self.n_jobs):
            if self.job_arrival_time_float[job_id] == 0 or self.job_arrival_time_float[job_id] == "timestamp":
                self.job_action_mask[job_id] = 1
        self.machine_action_mask = np.zeros((self.n_machines,), dtype=int)
        self.machine_action_mask[:self.n_machines] = 1

        # Reset other environment state as needed
        self.prev_cmax = 0
        self.curr_cmax = 0
        self.reward_this_step = 0.0
        self.schedule_done = False
        self.stage = 0
        self.machine_routes = {m_id: np.empty((0, 2), dtype=int) for m_id in range(self.n_machines)}
        self.result_start_time_for_jobs = np.zeros(shape=(self.n_jobs, self.max_n_operations, 2), dtype=float)
        self.result_finish_time_for_jobs = np.zeros(shape=(self.n_jobs, self.max_n_operations, 2), dtype=float)
        self.mean_processing_time_of_operations = np.zeros(shape=(self.n_jobs, self.max_n_operations), dtype=float)
        for job_id in range(self.n_jobs):
            for operation_id in range(self.n_operations_for_jobs[job_id]):
                # Here, you may want to use the operation's eligible_machines and durations from self.jobs
                op = self.jobs[job_id].operations[operation_id]
                durations = [em.standard_duration for em in op.eligible_machines]
                self.mean_processing_time_of_operations[job_id][operation_id] = np.mean(durations) if durations else 0.0
        self.mean_cumulative_processing_time_of_jobs = np.cumsum(self.mean_processing_time_of_operations, axis=1)
        self.makespan_baseline = 1.5 * self.n_jobs * self.n_machines * self.mean_processing_time_of_operations.max()

        # # generate colors for machines
        # c_map1 = plt.cm.get_cmap(self.c_map1)
        # arr1 = np.linspace(0, 1, self.n_machines, dtype=float)
        # self.machine_colors = {m_id: c_map1(val) for m_id, val in enumerate(arr1)}
        # # generate colors for jobs
        # c_map3 = plt.cm.get_cmap(self.c_map3)
        # arr3 = np.linspace(0, 1, self.n_jobs, dtype=float)
        # self.job_colors = {j_id: c_map3(val) for j_id, val in enumerate(arr3)}
        self.initialize_disjunctive_graph(self.n_operations_for_jobs)
        self.machine_routes = {m_id: np.empty((0, 2), dtype=int) for m_id in range(self.n_machines)}

        observations = self._get_obs()
        info = self._get_info()
        return observations, info

    def step(self, action):
        observations, reward, terminated, truncated, info = {}, {}, {}, {}, {}
        self.reward_this_step = 0.0
        # 初始化所有agent的reward/terminated/truncated为0/False
        for agent in self.agents:
            reward[agent] = 0.0
            terminated[agent] = False
            truncated[agent] = False

        if self.stage == 0:
            self.chosen_job = action["agent0"]

            # invalid job_id
            if self.chosen_job >= self.n_jobs or self.chosen_job < 0:
                self.schedule_done = min(self.job_features[:, 4]) >= 1
                if self.schedule_done:
                    self.final_makespan = self.curr_cmax
                    self.reward_this_step = self.reward_this_step + 1
                for i in range(2):
                    agent = f"agent{i}"
                    reward[agent] = self.reward_this_step
                    terminated[agent] = self.schedule_done
                    if terminated[agent]:
                        self.terminateds.add(agent)
                    truncated[agent] = False
                observations = self._get_obs()
                info = self._get_info()
                terminated["__all__"] = len(self.terminateds) == len(self.agents)
                truncated["__all__"] = False
                return observations, reward, terminated, truncated, info

            self.operation_id = int(self.job_features[self.chosen_job, 1])
            self.prcs_task_id = sum(self.n_operations_for_jobs[:self.chosen_job]) + self.operation_id

            # invalid operation_id
            if self.operation_id >= self.n_operations_for_jobs[self.chosen_job] or self.operation_id < 0:
                self.schedule_done = min(self.job_features[:, 4]) >= 1
                if self.schedule_done:
                    self.final_makespan = self.curr_cmax
                    self.reward_this_step = self.reward_this_step + 1
                for i in range(2):
                    agent = f"agent{i}"
                    reward[agent] = self.reward_this_step
                    terminated[agent] = self.schedule_done
                    if terminated[agent]:
                        self.terminateds.add(agent)
                    truncated[agent] = False
                observations = self._get_obs()
                info = self._get_info()
                terminated["__all__"] = len(self.terminateds) == len(self.agents)
                truncated["__all__"] = False
                return observations, reward, terminated, truncated, info

            # update machines' state
            for machine_id in range(self.n_machines):
                expected_duration = self.get_operation_duration(self.chosen_job, self.operation_id, machine_id)
                self.machine_features[machine_id, 5] = max(-1.0, expected_duration / self.machine_features[machine_id, 1])
                if self.machine_features[machine_id, 5] > 0:
                    self.machine_action_mask[machine_id] = 1
                else:
                    self.machine_action_mask[machine_id] = 0

            self.stage = 1
            reward["agent0"] = self.reward_this_step
            # agent1 reward remains 0

        else:
            self.chosen_machine = action["agent1"]

            # invalid machine_id
            if self.chosen_machine >= self.n_machines or self.chosen_machine < 0 or self.machine_features[self.chosen_machine, 5] <= 0:
                self.schedule_done = min(self.job_features[:, 4]) >= 1
                if self.schedule_done:
                    self.final_makespan = self.curr_cmax
                    self.reward_this_step = self.reward_this_step + 1
                for i in range(2):
                    agent = f"agent{i}"
                    reward[agent] = self.reward_this_step
                    terminated[agent] = self.schedule_done
                    if terminated[agent]:
                        self.terminateds.add(agent)
                    truncated[agent] = False
                observations = self._get_obs()
                info = self._get_info()
                terminated["__all__"] = len(self.terminateds) == len(self.agents)
                truncated["__all__"] = False
                return observations, reward, terminated, truncated, info

            prcs_result = self._schedule_prcs_task(
                job_id=self.chosen_job,
                operation_id=self.operation_id,
                task_id=self.prcs_task_id,
                machine_id=self.chosen_machine
            )

            self.env_current_time = max(self.machine_features[:self.n_machines,3])
            for job_id in range(self.n_jobs):
                if self.job_arrival_time_float[job_id] < self.env_current_time and self.job_features[job_id, 1] == 0:
                    self.job_action_mask[job_id] = 1

            self.machine_features[:self.n_machines, 5:] = -1

            self.prev_cmax = self.curr_cmax
            self.curr_cmax = self.result_finish_time_for_jobs.max()

            self.reward_this_step = self.reward_this_step + 1.0 * (self.prev_cmax - self.curr_cmax) / self.makespan_baseline

            self.schedule_done = min(self.job_features[:, 4]) == 1
            if self.schedule_done:
                # Compute makespan as (max finish time - min start time) over all scheduled operations
                start_times = self.result_start_time_for_jobs[:, :, 1].flatten()
                finish_times = self.result_finish_time_for_jobs[:, :, 1].flatten()
                # Only consider nonzero start/finish times (ignore padding zeros)
                valid_mask = (finish_times > 0)
                if np.any(valid_mask):
                    min_start = np.min(start_times[valid_mask])
                    max_finish = np.max(finish_times[valid_mask])
                    self.final_makespan = round(max_finish - min_start, 1)
                    self.schedule_start_time = min_start
                    self.schedule_end_time = max_finish
                else:
                    self.final_makespan = 0.0
                    self.schedule_start_time = 0.0
                    self.schedule_end_time = 0.0
                self.build_and_save_output_json()

                # global_schedule_file = os.path.dirname(os.path.abspath(__file__)) \
                #     + f"/global_schedules/....pkl"
                # self.global_schedule = convert_schedule_to_class(
                #         file_name=global_schedule_file,
                #         makespan=self.final_makespan,
                #         job_arrival_time=self.job_arrival_time,
                #         job_due_date=self.job_due_date,
                #         result_start_time_for_jobs=self.result_start_time_for_jobs,
                #         result_finish_time_for_jobs=self.result_finish_time_for_jobs,
                #         machine_routes=self.machine_routes,
                #         transbot_routes=None # 保持接口兼容性，传None
                # )
                self.reward_this_step = self.reward_this_step + 1

            reward["agent1"] = self.reward_this_step
            # agent0 reward remains 0
            for i in range(2):
                agent = f"agent{i}"
                terminated[agent] = self.schedule_done
                if terminated[agent]:
                    self.terminateds.add(agent)
                truncated[agent] = False
            self.stage = 0

        observations = self._get_obs()
        info = self._get_info()
        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = False

        return observations, reward, terminated, truncated, info

    # *********************************
    # Schedule An Processing Task
    # *********************************
    def _schedule_prcs_task(self, job_id: int, task_id: int, operation_id: int, machine_id: int) -> dict:
        """
        schedules a process task/node in the graph representation if the task can be scheduled.
        """
        if self.Graph.nodes[task_id]["node_type"] != "operation" or \
                job_id != self.Graph.nodes[task_id]["job_id"] or \
                self.Graph.nodes[task_id]["operation_id"] != self.job_features[job_id][1]:
            return {
                "schedule_success": False,
            }

        if machine_id < 0 or machine_id >= self.n_machines:
            return {
                "schedule_success": False,
            }

        new_task = np.array([[job_id, operation_id]], dtype=int)
        previous_operation_finish_time = float(self.job_features[job_id][2])

        # expected_duration = self.jobs[job_id].operations_matrix[self.Graph.nodes[task_id]["operation_id"], machine_id]
        expected_duration = self.get_operation_duration(job_id, operation_id, machine_id)
        expected_duration = expected_duration * dfjspt_params.prcs_time_factor
        if expected_duration <= 0:
            return {
                "schedule_success": False,
            }
        if self.machine_quality[0, machine_id] > 0:
            if random.random() <= self.machine_quality[0, machine_id]:
                actual_duration = int(expected_duration)
            else:
                actual_duration = int(random.uniform(expected_duration,
                                                 expected_duration / self.machine_quality[0, machine_id]))
        else:
            actual_duration = 999

        len_machine_routes = len(self.machine_routes[machine_id])
        if len_machine_routes:
            if self.perform_left_shift_if_possible:
                j_lower_bound_st = previous_operation_finish_time
                j_lower_bound_ft = j_lower_bound_st + actual_duration

                # check if task can be scheduled between src and first task
                machine_first_task_start_time = self.result_start_time_for_jobs[
                    self.machine_routes[machine_id][0][0],
                    self.machine_routes[machine_id][0][1], 1]

                if j_lower_bound_ft <= machine_first_task_start_time:
                    self.machine_routes[machine_id] = np.insert(self.machine_routes[machine_id], 0, new_task, axis=0)
                    start_time = previous_operation_finish_time
                    finish_time = start_time + actual_duration

                    self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = start_time
                    self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = finish_time
                    self.Graph.nodes[task_id]["m_start_time"] = start_time
                    self.Graph.nodes[task_id]["m_finish_time"] = finish_time
                    self.Graph.nodes[task_id]["machine_id"] = machine_id
                    self.machine_features[machine_id][2] += 1
                    last_task_in_route = self.machine_routes[machine_id][-1]
                    machine_last_finish_time = self.result_finish_time_for_jobs[last_task_in_route[0], last_task_in_route[1], 1]
                    self.machine_features[machine_id][3] = machine_last_finish_time
                    self.machine_features[machine_id][4] += actual_duration

                    self.job_features[job_id][1] += 1
                    self.job_features[job_id][2] = finish_time
                    self.job_features[job_id][5] += actual_duration
                    self.job_features[job_id][7] -= self.job_features[job_id, 6]
                    if self.job_features[job_id][7] < 0:
                        self.job_features[job_id][7] = 0
                    if self.job_features[job_id][1] >= self.n_operations_for_jobs[job_id]:
                        self.job_features[job_id, 4] = 1
                        self.job_features[job_id, 6] = -1
                        self.job_action_mask[job_id] = 0
                    else:
                        self.job_features[job_id, 6] = self.mean_processing_time_of_operations[job_id][
                            int(self.job_features[job_id][1])]

                    return {
                        "schedule_success": True,
                        "actual_duration": actual_duration,
                        "start_time": start_time,
                        "finish_time": finish_time,
                    }

                elif len_machine_routes == 1:
                    self.machine_routes[machine_id] = np.append(self.machine_routes[machine_id], new_task, axis=0)
                    machine_previous_task_finish_time = float(self.machine_features[machine_id, 3])
                    start_time = max(previous_operation_finish_time, machine_previous_task_finish_time)
                    finish_time = start_time + actual_duration

                    self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = start_time
                    self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = finish_time
                    self.Graph.nodes[task_id]["start_time"] = start_time
                    self.Graph.nodes[task_id]["finish_time"] = finish_time
                    self.Graph.nodes[task_id]["machine_id"] = machine_id
                    self.machine_features[machine_id][2] += 1
                    self.machine_features[machine_id][3] = finish_time
                    self.machine_features[machine_id][4] += actual_duration

                    self.job_features[job_id][1] += 1
                    self.job_features[job_id][2] = finish_time
                    self.job_features[job_id][5] += actual_duration
                    self.job_features[job_id][7] -= self.job_features[job_id, 6]
                    if self.job_features[job_id][7] < 0:
                        self.job_features[job_id][7] = 0
                    if self.job_features[job_id][1] >= self.n_operations_for_jobs[job_id]:
                        self.job_features[job_id, 4] = 1
                        self.job_features[job_id, 6] = -1
                        self.job_action_mask[job_id] = 0
                    else:
                        self.job_features[job_id, 6] = self.mean_processing_time_of_operations[job_id][
                            int(self.job_features[job_id][1])]

                    return {
                        "schedule_success": True,
                        "actual_duration": actual_duration,
                        "start_time": start_time,
                        "finish_time": finish_time,
                    }

                # check if task can be scheduled between two tasks
                for i, (m_prev, m_next) in enumerate(
                        zip(self.machine_routes[machine_id], self.machine_routes[machine_id][1:])):
                    m_temp_prev_ft = self.result_finish_time_for_jobs[m_prev[0], m_prev[1], 1]
                    m_temp_next_st = self.result_start_time_for_jobs[m_next[0], m_next[1], 1]

                    if j_lower_bound_ft > m_temp_next_st:
                        continue

                    m_gap = m_temp_next_st - m_temp_prev_ft
                    if m_gap < actual_duration:
                        continue

                    # at this point the task can fit in between two already scheduled task
                    start_time = max(j_lower_bound_st, m_temp_prev_ft)
                    finish_time = start_time + actual_duration
                    # insert task at the corresponding place in the machine routes list
                    self.machine_routes[machine_id] = np.insert(self.machine_routes[machine_id], i + 1, new_task, axis=0)

                    self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = start_time
                    self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = finish_time
                    self.Graph.nodes[task_id]["m_start_time"] = start_time
                    self.Graph.nodes[task_id]["m_finish_time"] = finish_time
                    self.Graph.nodes[task_id]["machine_id"] = machine_id
                    self.machine_features[machine_id][2] += 1
                    last_task_in_route = self.machine_routes[machine_id][-1]
                    machine_last_finish_time = self.result_finish_time_for_jobs[last_task_in_route[0], last_task_in_route[1], 1]
                    self.machine_features[machine_id][3] = machine_last_finish_time
                    self.machine_features[machine_id][4] += actual_duration

                    self.job_features[job_id][1] += 1
                    self.job_features[job_id][2] = finish_time
                    self.job_features[job_id][5] += actual_duration
                    self.job_features[job_id][7] -= self.job_features[job_id, 6]
                    if self.job_features[job_id][7] < 0:
                        self.job_features[job_id][7] = 0
                    if self.job_features[job_id][1] >= self.n_operations_for_jobs[job_id]:
                        self.job_features[job_id, 4] = 1
                        self.job_features[job_id, 6] = -1
                        self.job_action_mask[job_id] = 0
                    else:
                        self.job_features[job_id, 6] = self.mean_processing_time_of_operations[job_id][
                            int(self.job_features[job_id][1])]

                    return {
                        "schedule_success": True,
                        "actual_duration": actual_duration,
                        "start_time": start_time,
                        "finish_time": finish_time,
                    }

                self.machine_routes[machine_id] = np.append(self.machine_routes[machine_id], new_task, axis=0)
                machine_previous_task_finish_time = float(self.machine_features[machine_id, 3])
                start_time = max(previous_operation_finish_time, machine_previous_task_finish_time)
                finish_time = start_time + actual_duration

            else:
                self.machine_routes[machine_id] = np.append(self.machine_routes[machine_id], new_task, axis=0)
                machine_previous_task_finish_time = float(self.machine_features[machine_id, 3])
                start_time = max(previous_operation_finish_time, machine_previous_task_finish_time)
                finish_time = start_time + actual_duration
        else:
            self.machine_routes[machine_id] = np.insert(self.machine_routes[machine_id], 0, new_task, axis=0)
            start_time = previous_operation_finish_time
            finish_time = start_time + actual_duration

        self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = start_time
        self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = finish_time
        self.Graph.nodes[task_id]["m_start_time"] = start_time
        self.Graph.nodes[task_id]["m_finish_time"] = finish_time
        self.Graph.nodes[task_id]["machine_id"] = machine_id
        self.machine_features[machine_id][2] += 1
        self.machine_features[machine_id][3] = finish_time
        self.machine_features[machine_id][4] += actual_duration

        self.job_features[job_id][1] += 1
        self.job_features[job_id][2] = finish_time
        self.job_features[job_id][5] += actual_duration
        self.job_features[job_id][7] -= self.job_features[job_id, 6]
        if self.job_features[job_id][7] < 0:
            self.job_features[job_id][7] = 0
        if self.job_features[job_id][1] >= self.n_operations_for_jobs[job_id]:
            self.job_features[job_id, 4] = 1
            self.job_features[job_id, 6] = -1
            self.job_action_mask[job_id] = 0
        else:
            self.job_features[job_id, 6] = self.mean_processing_time_of_operations[job_id][int(self.job_features[job_id][1])]

        return {
            "schedule_success": True,
            "actual_duration": actual_duration,
            "start_time": start_time,
            "finish_time": finish_time,
        }

    def initialize_disjunctive_graph(self,
                                     n_operations_for_jobs,
                                     ) -> None:
        """
        Get a new disjunctive graph (transportation logic removed).
        """
        self.Graph = nx.DiGraph()

        # add nodes for processing machines
        for machine_id in range(self.n_machines):
            self.Graph.add_node(
                "machine" + str(machine_id),
                machine_id=machine_id,
                node_type="machine",
                shape='p',
                pos=(machine_id, -self.n_jobs - 2),
                # color=self.machine_colors[machine_id],
            )

        # add src node
        self.Graph.add_node(
            self.source_task,
            node_type="dummy",
            shape='o',
            pos=(-2, int(-self.n_jobs * 0.5) - 1),
            # color=self.dummy_task_color,
        )
        # add sink task at the end to avoid permutation in the adj matrix.
        self.Graph.add_node(
            self.sink_task,
            node_type="dummy",
            shape='o',
            pos=(2 * max(n_operations_for_jobs), int(-self.n_jobs * 0.5) - 1),
            # color=self.dummy_task_color,
        )

        # add nodes for tasks
        task_id = -1
        self.task_job_operation_dict = {}
        for job_id in range(self.n_jobs):
            for operation_id in range(n_operations_for_jobs[job_id]):
                task_id += 1  # start from operation task id 0, -1 is dummy starting task
                self.task_job_operation_dict[task_id] = [job_id, operation_id]
                # add an operation task node
                self.Graph.add_node(
                    task_id,
                    node_type="operation",
                    job_id=job_id,
                    operation_id=operation_id,
                    shape='s',
                    pos=(2 * operation_id - 1, -job_id - 1),
                    # color=self.job_colors[job_id],
                    m_start_time=-1,
                    m_finish_time=-1,
                    machine_id=-1,
                )
                if operation_id != 0:
                    # add a conjunctive edge from last operation (task_id - 1) to this operation (task_id)
                    self.Graph.add_edge(
                        task_id - 1, task_id,
                        edge_type="conjunctive_arc",
                        weight=0,
                    )

    def get_operation_duration(self, job_id, operation_id, machine_id):
        op = self.jobs[job_id].operations[operation_id]
        machine_id_str = self.machine_info[machine_id].machine_id
        for em in op.eligible_machines:
            if em.machine_id == machine_id_str:
                return em.standard_duration
        return -1

    def build_and_save_output_json(self, output_path="Data/OutputData/output_data_example.json"):
        # Ensure output_path is absolute and under project root
        if not os.path.isabs(output_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_path = os.path.join(project_root, output_path)
        # Build output data structure as per Output Data Class Diagram
        # 1. Algorithm info
        algorithm_info = {
            "algorithm_name": "FjspMaEnv",
            "execution_time": float(getattr(self, 'execution_time', 0.0))
        }
        # 2. Schedule summary
        schedule_summary = {
            "total_makespan": float(self.final_makespan),
            "schedule_start_time": str(getattr(self, 'schedule_start_time', datetime.datetime.now().isoformat())),
            "schedule_end_time": str(getattr(self, 'schedule_end_time', datetime.datetime.now().isoformat())),
        }
        # 3. Gantt chart data
        machine_timelines = []
        for machine_id in range(self.n_machines):
            timeline_bars = []
            for route in self.machine_routes[machine_id]:
                job_id, operation_id = route
                job = self.jobs[job_id]
                op = job.operations[operation_id]
                # Find the start/end time for this operation on this machine
                start_time = self.result_start_time_for_jobs[job_id, operation_id, 1]
                end_time = self.result_finish_time_for_jobs[job_id, operation_id, 1]
                timeline_bars.append({
                    "operation_id": op.operation_id,
                    "job_id": job.job_id,
                    "product_id": job.product_id,
                    "order_id": job.order_id,
                    "start_time": float(start_time),
                    "end_time": float(end_time)
                })
            machine_timelines.append({
                "machine_id": self.machine_info[machine_id].machine_id,
                "timeline_bars": timeline_bars
            })
        gantt_chart_data = {
            "machine_timelines": machine_timelines
        }
        # 4. Top-level output
        output_data = {
            "scheduling_result": {
                "instance_id": self.scheduling_instance.instance_id if self.scheduling_instance else "unknown",
                "algorithm_info": algorithm_info,
                "schedule_summary": schedule_summary,
                "gantt_chart_data": gantt_chart_data
            }
        }
        # Save to JSON
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Scheduling result saved to {output_path}")






