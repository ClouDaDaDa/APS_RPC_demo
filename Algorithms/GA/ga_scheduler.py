import numpy as np
import random
import copy
import os
import sys

# add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from Environment.dfjspt_env import FjspMaEnv

class GAConfig:
    def __init__(self,
                 population_size=30,
                 generations=50,
                 crossover_rate=0.8,
                 mutation_rate=0.2,
                 elitism=2,
                 alpha=0.7,  # priority weight
                 seed=None):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.alpha = alpha
        self.seed = seed

class Chromosome:
    def __init__(self, job_seq, machine_seq):
        self.job_seq = job_seq  # operation scheduling sequence (global op index sequence)
        self.machine_seq = machine_seq  # machine assigned to each operation
        self.fitness = None
        self.schedule_result = None  # optional: store scheduling result

class GAScheduler:
    def __init__(self, env, config: GAConfig):
        self.env = env
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
            random.seed(config.seed)
        self.n_jobs = env.n_jobs
        self.n_operations = sum(env.n_operations_for_jobs)
        self.n_machines = env.n_machines
        self.job_priority = np.array(env.job_priority)
        # record available machines for each operation
        self.op_machine_choices = []  # [(job_id, op_id, [machine_ids])]
        for job_id, job in enumerate(env.jobs):
            for op_id, op in enumerate(job.operations):
                eligible = [env.machine_id_to_index[em.machine_id] for em in op.eligible_machines]
                self.op_machine_choices.append(eligible)
        # global operation index to (job_id, op_id)
        self.op_global_to_local = []
        for job_id, job in enumerate(env.jobs):
            for op_id, op in enumerate(job.operations):
                self.op_global_to_local.append((job_id, op_id))
        # number of operations for each job
        self.job_op_count = [len(job.operations) for job in env.jobs]
        # save input data path
        self.inputdata_json = getattr(env, 'inputdata_json', None)

    def heuristic_chromosome(self):
        env = FjspMaEnv({'train_or_eval_or_test': 'test', 'inputdata_json': self.inputdata_json})
        obs, info = env.reset()
        env.job_arrival_time = [0 for _ in env.job_arrival_time]
        for job_id in range(env.n_jobs):
            env.job_features[job_id, 2] = 0
        obs, info = env.reset()
        done = False
        op_seq = []
        machine_seq = []
        job_op_done = [0 for _ in range(env.n_jobs)]
        while not done:
            job_mask = obs['agent0']['action_mask']
            job_features = obs['agent0']['observation']
            available_jobs = np.where(job_mask == 1)[0]
            if len(available_jobs) == 0:
                break
            arrival_times = job_features[available_jobs, 2]
            job_action = available_jobs[np.argmin(arrival_times)]
            op_id = job_op_done[job_action]
            op_global = sum(env.n_operations_for_jobs[:job_action]) + op_id
            op_seq.append(op_global)
            obs, reward, terminated, truncated, info = env.step({'agent0': job_action})
            if terminated['__all__']:
                break
            machine_mask = obs['agent1']['action_mask']
            machine_features = obs['agent1']['observation']
            available_machines = np.where(machine_mask == 1)[0]
            idle_times = machine_features[available_machines, 3]
            machine_action = available_machines[np.argmin(idle_times)]
            machine_seq.append(machine_action)
            obs, reward, terminated, truncated, info = env.step({'agent1': machine_action})
            job_op_done[job_action] += 1
            done = terminated['__all__']
        return op_seq, machine_seq

    def initialize_population(self):
        population = []
        n_heuristic = max(1, int(0.2 * self.config.population_size))
        for _ in range(self.config.population_size - n_heuristic):
            job_op_queues = [[(job_id, op_id) for op_id in range(n_ops)] for job_id, n_ops in enumerate(self.job_op_count)]
            op_seq = []
            job_ptrs = [0 for _ in self.job_op_count]
            total_ops = sum(self.job_op_count)
            for _ in range(total_ops):
                available = [j for j in range(self.n_jobs) if job_ptrs[j] < self.job_op_count[j]]
                j = random.choice(available)
                op_id = job_ptrs[j]
                op_seq.append((j, op_id))
                job_ptrs[j] += 1
            op_seq_global = [sum(self.job_op_count[:j]) + op_id for j, op_id in op_seq]
            machine_seq = [random.choice(self.op_machine_choices[op]) for op in op_seq_global]
            population.append(Chromosome(op_seq_global, machine_seq))
        for _ in range(n_heuristic):
            op_seq, machine_seq = self.heuristic_chromosome()
            population.append(Chromosome(op_seq, machine_seq))
        return population

    def fitness(self, chrom: Chromosome):
        inputdata_json = self.env.inputdata_json if hasattr(self.env, 'inputdata_json') else 'Data/InputData/input_data_example_W3_O3_P10.json'
        env = FjspMaEnv({'train_or_eval_or_test': 'test', 'inputdata_json': inputdata_json})
        obs, info = env.reset()
        # force all job arrival times to be 0
        env.job_arrival_time = [0 for _ in env.job_arrival_time]
        for job_id in range(env.n_jobs):
            env.job_features[job_id, 2] = 0
        obs, info = env.reset()
        done = False
        op_seq = chrom.job_seq
        machine_seq = chrom.machine_seq
        total_ops = len(op_seq)
        scheduled = [False] * total_ops
        job_op_done = [0 for _ in range(self.n_jobs)]
        while not done and not all(scheduled):
            job_mask = obs['agent0']['action_mask']
            available_jobs = np.where(job_mask == 1)[0]
            progressed = False
            for idx in range(total_ops):
                if scheduled[idx]:
                    continue
                op_global = op_seq[idx]
                job_id, op_id = self.op_global_to_local[op_global]
                if job_id in available_jobs and op_id == job_op_done[job_id]:
                    obs, reward, terminated, truncated, info = env.step({'agent0': job_id})
                    if terminated['__all__']:
                        done = True
                        break
                    machine_mask = obs['agent1']['action_mask']
                    available_machines = np.where(machine_mask == 1)[0]
                    m = machine_seq[idx]
                    if m not in available_machines:
                        m = available_machines[0]
                    obs, reward, terminated, truncated, info = env.step({'agent1': m})
                    job_op_done[job_id] += 1
                    scheduled[idx] = True
                    progressed = True
                    done = terminated['__all__']
            if not progressed:
                # deadlock
                chrom.fitness = 1e6
                chrom.schedule_result = env
                return 1e6
        # if done is not triggered, force a dummy step to ensure environment completes scheduling
        if not getattr(env, 'schedule_done', False):
            try:
                obs, reward, terminated, truncated, info = env.step({'agent0': -1})
            except Exception:
                pass
        # force refresh makespan etc.
        start_times = env.result_start_time_for_jobs[:, :, 1].flatten()
        finish_times = env.result_finish_time_for_jobs[:, :, 1].flatten()
        valid_mask = (finish_times > 0)
        if np.any(valid_mask):
            min_start = np.min(start_times[valid_mask])
            max_finish = np.max(finish_times[valid_mask])
            env.final_makespan = round(max_finish - min_start, 1)
            env.schedule_start_time = min_start
            env.schedule_end_time = max_finish
        else:
            env.final_makespan = 0.0
            env.schedule_start_time = 0.0
            env.schedule_end_time = 0.0
        # calculate weighted objective
        job_finish = np.zeros(self.n_jobs)
        for job_id in range(self.n_jobs):
            finish_times = env.result_finish_time_for_jobs[job_id, :, 1]
            job_finish[job_id] = np.max(finish_times)
        norm_priority = self.job_priority / (self.job_priority.max() if self.job_priority.max() > 0 else 1)
        weighted_sum = np.sum(norm_priority * job_finish)
        chrom.fitness = weighted_sum
        chrom.schedule_result = env
        return weighted_sum

    def selection(self, population):
        selected = []
        for _ in range(len(population)):
            a, b = random.sample(population, 2)
            selected.append(a if a.fitness < b.fitness else b)
        return selected

    def crossover(self, parent1: Chromosome, parent2: Chromosome):
        size = len(parent1.job_seq)
        job_op_blocks = [[] for _ in range(self.n_jobs)]
        for idx in range(size):
            job_id, op_id = self.op_global_to_local[parent1.job_seq[idx]]
            job_op_blocks[job_id].append(parent1.job_seq[idx])
        jobs_from_p1 = set(random.sample(range(self.n_jobs), self.n_jobs // 2))
        child_job_seq = []
        for idx in range(size):
            job_id, op_id = self.op_global_to_local[parent1.job_seq[idx]]
            if job_id in jobs_from_p1:
                child_job_seq.append(parent1.job_seq[idx])
        for idx in range(size):
            job_id, op_id = self.op_global_to_local[parent2.job_seq[idx]]
            if job_id not in jobs_from_p1:
                child_job_seq.append(parent2.job_seq[idx])
        # uniform crossover
        child_machine_seq = [parent1.machine_seq[i] if random.random() < 0.5 else parent2.machine_seq[i] for i in range(size)]
        return Chromosome(child_job_seq, child_machine_seq)

    def mutate(self, chrom: Chromosome):
        size = len(chrom.job_seq)
        # multi-point swap
        for _ in range(random.randint(1, 3)):
            idx1, idx2 = random.sample(range(size), 2)
            job1, op1 = self.op_global_to_local[chrom.job_seq[idx1]]
            job2, op2 = self.op_global_to_local[chrom.job_seq[idx2]]
            if job1 != job2:
                chrom.job_seq[idx1], chrom.job_seq[idx2] = chrom.job_seq[idx2], chrom.job_seq[idx1]
        # multi-point machine mutation
        for _ in range(random.randint(1, 3)):
            idx = random.randint(0, size-1)
            chrom.machine_seq[idx] = random.choice(self.op_machine_choices[chrom.job_seq[idx]])

    def evolve(self):
        population = self.initialize_population()
        # evaluate initial population
        for chrom in population:
            self.fitness(chrom)
        best_chrom = min(population, key=lambda c: c.fitness)
        for gen in range(self.config.generations):
            # selection
            selected = self.selection(population)
            # crossover
            offspring = []
            for i in range(0, len(selected)-1, 2):
                if random.random() < self.config.crossover_rate:
                    c1 = self.crossover(selected[i], selected[i+1])
                    c2 = self.crossover(selected[i+1], selected[i])
                    offspring.extend([c1, c2])
                else:
                    offspring.extend([copy.deepcopy(selected[i]), copy.deepcopy(selected[i+1])])
            # mutation
            for c in offspring:
                if random.random() < self.config.mutation_rate:
                    self.mutate(c)
            # evaluate new population
            for c in offspring:
                self.fitness(c)
            # elitism
            total = population + offspring
            total.sort(key=lambda c: c.fitness)
            population = total[:self.config.population_size]
            if population[0].fitness < best_chrom.fitness:
                best_chrom = copy.deepcopy(population[0])
            print(f"Gen {gen+1}: Best weighted sum = {best_chrom.fitness:.2f}")
        return best_chrom

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GA Scheduler for FJSP with Priority')
    parser.add_argument('--input', type=str, default='Data/InputData/input_data_example_W3_O3_P10.json')
    parser.add_argument('--output', type=str, default='Data/OutputData/output_data_GA_example.json')
    parser.add_argument('--pop', type=int, default=100)
    parser.add_argument('--gen', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--test_rule', action='store_true', help='Test rule-based chromosome decoding')
    parser.add_argument('--main_debug', action='store_true', help='Debug main rule-based env state')
    args = parser.parse_args()

    args.input = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'Data', 'InputData', 'input_data_example_W3_O3_P10.json')

    # load environment
    env = FjspMaEnv({'train_or_eval_or_test': 'test', 'inputdata_json': args.input})
    config = GAConfig(population_size=args.pop, generations=args.gen, alpha=args.alpha, seed=args.seed)
    ga = GAScheduler(env, config)

    if args.main_debug:
        print('--- Main process rule-based env state after reset ---')
        obs, info = env.reset()
        print('[MAIN DEBUG] env.job_arrival_time:', env.job_arrival_time)
        print('[MAIN DEBUG] env.job_features:', env.job_features)
        print('[MAIN DEBUG] env.job_action_mask:', getattr(env, 'job_action_mask', None))
        print('[MAIN DEBUG] obs[agent0][action_mask]:', obs['agent0']['action_mask'])
    elif args.test_rule:
        # use rule-based method to generate chromosome, test decoding
        print('Testing rule-based chromosome decoding...')
        obs, info = env.reset()
        done = False
        op_seq = []
        machine_seq = []
        job_op_done = [0 for _ in range(env.n_jobs)]
        while not done:
            job_mask = obs['agent0']['action_mask']
            job_features = obs['agent0']['observation']
            available_jobs = np.where(job_mask == 1)[0]
            if len(available_jobs) == 0:
                break
            arrival_times = job_features[available_jobs, 2]
            job_action = available_jobs[np.argmin(arrival_times)]
            op_id = job_op_done[job_action]
            op_global = sum(env.n_operations_for_jobs[:job_action]) + op_id
            op_seq.append(op_global)
            obs, reward, terminated, truncated, info = env.step({'agent0': job_action})
            if terminated['__all__']:
                break
            machine_mask = obs['agent1']['action_mask']
            machine_features = obs['agent1']['observation']
            available_machines = np.where(machine_mask == 1)[0]
            idle_times = machine_features[available_machines, 3]
            machine_action = available_machines[np.argmin(idle_times)]
            machine_seq.append(machine_action)
            obs, reward, terminated, truncated, info = env.step({'agent1': machine_action})
            job_op_done[job_action] += 1
            done = terminated['__all__']
        chrom = Chromosome(op_seq, machine_seq)
        ga.fitness(chrom)
        print(f'Rule-based chromosome fitness: {chrom.fitness}')
        print(f'Rule-based chromosome scheduled ops: {len([s for s in chrom.job_seq if s is not None])}')
        chrom.schedule_result.build_and_save_output_json(args.output)
        print(f'Rule-based scheduling result saved to: {args.output}')
    else:
        best = ga.evolve()
        print(f"Best weighted sum: {best.fitness:.2f}")
        # Save only the final result
        best.schedule_result.build_and_save_output_json(args.output)
        print(f"GA scheduling result saved to: {args.output}")
        print(f"Final Makespan: {best.schedule_result.final_makespan}")
