"""
Baseline Algorithms for JSSPLA:
1. Priority Dispatching Heuristics: FIFO, SPT, LPT
2. Standard GA (Genetic Algorithm)
3. Standard PSO (Single Swarm with SPV)
4. Proposed CDPSO (Cooperative Discrete PSO from Paper Algorithm 1)
"""

from typing import List, Tuple, Dict, Optional
import time
import random
import numpy as np

from warehouse_env import WarehouseInstance, JSSPLADecoder, JSSPLASchedule


# ---------------------------------------------------------------------------
# 1. Deterministic Heuristics: FIFO, SPT, LPT
# ---------------------------------------------------------------------------

def solve_fifo(instance: WarehouseInstance) -> Tuple[float, JSSPLASchedule]:
    """FIFO: Orders scheduled in sequential index order."""
    decoder = JSSPLADecoder(instance)
    os_seq = []
    # Schedule operation by operation in order of job arrival
    max_ops = max(len(j.operations) for j in instance.jobs)
    for op_round in range(max_ops):
        for job in instance.jobs:
            if op_round < len(job.operations):
                os_seq.append(job.job_id)

    # Balance AA alternating between the 2 dedicated AMRs
    aa_seq = [i % 2 for i in range(len(os_seq))]
    sched = decoder.decode(os_seq, aa_seq)
    return sched.makespan, sched


def solve_spt(instance: WarehouseInstance) -> Tuple[float, JSSPLASchedule]:
    """SPT: Shortest Total Processing Time first."""
    decoder = JSSPLADecoder(instance)
    # Sort jobs by total duration ascending
    job_durations = [(j.job_id, sum(op.duration for op in j.operations)) for j in instance.jobs]
    job_durations.sort(key=lambda x: x[1])

    os_seq = []
    for job_id, _ in job_durations:
        n_ops = len(instance.jobs[job_id].operations)
        os_seq.extend([job_id] * n_ops)

    aa_seq = [i % 2 for i in range(len(os_seq))]
    sched = decoder.decode(os_seq, aa_seq)
    return sched.makespan, sched


def solve_lpt(instance: WarehouseInstance) -> Tuple[float, JSSPLASchedule]:
    """LPT: Longest Total Processing Time first."""
    decoder = JSSPLADecoder(instance)
    # Sort jobs by total duration descending
    job_durations = [(j.job_id, sum(op.duration for op in j.operations)) for j in instance.jobs]
    job_durations.sort(key=lambda x: x[1], reverse=True)

    os_seq = []
    for job_id, _ in job_durations:
        n_ops = len(instance.jobs[job_id].operations)
        os_seq.extend([job_id] * n_ops)

    aa_seq = [i % 2 for i in range(len(os_seq))]
    sched = decoder.decode(os_seq, aa_seq)
    return sched.makespan, sched


# ---------------------------------------------------------------------------
# 2. Standard Genetic Algorithm (GA)
# ---------------------------------------------------------------------------

class StandardGA:
    """Standard GA for JSSPLA with combined chromosome (OS, AA)."""
    def __init__(self, pop_size: int = 40, max_gen: int = 50, pc: float = 0.8, pm: float = 0.15, seed: int = 42):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.pc = pc
        self.pm = pm
        self.seed = seed

    def solve(self, instance: WarehouseInstance) -> Tuple[float, JSSPLASchedule, List[float]]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        decoder = JSSPLADecoder(instance)
        base_os = list(instance.base_sequence)
        total_ops = len(base_os)

        # Initialize population
        population = []
        for _ in range(self.pop_size):
            os_ind = list(base_os)
            random.shuffle(os_ind)
            aa_ind = [random.randint(0, 1) for _ in range(total_ops)]
            population.append((os_ind, aa_ind))

        history = []
        best_makespan = float("inf")
        best_sched = None

        for gen in range(self.max_gen):
            # Evaluate
            fits = []
            for os_ind, aa_ind in population:
                sched = decoder.decode(os_ind, aa_ind)
                fits.append(sched.makespan)
                if sched.makespan < best_makespan:
                    best_makespan = sched.makespan
                    best_sched = sched

            history.append(best_makespan)

            # Selection (Tournament)
            new_pop = []
            for _ in range(self.pop_size // 2):
                # Tournament 1
                i1, i2 = random.sample(range(self.pop_size), 2)
                p1 = population[i1] if fits[i1] < fits[i2] else population[i2]
                # Tournament 2
                i3, i4 = random.sample(range(self.pop_size), 2)
                p2 = population[i3] if fits[i3] < fits[i4] else population[i4]

                # Crossover
                c1_os, c1_aa = list(p1[0]), list(p1[1])
                c2_os, c2_aa = list(p2[0]), list(p2[1])

                if random.random() < self.pc:
                    # Uniform crossover for AA
                    for idx in range(total_ops):
                        if random.random() < 0.5:
                            c1_aa[idx], c2_aa[idx] = c2_aa[idx], c1_aa[idx]

                # Mutation
                if random.random() < self.pm:
                    # Swap mutation for OS
                    idx1, idx2 = random.sample(range(total_ops), 2)
                    c1_os[idx1], c1_os[idx2] = c1_os[idx2], c1_os[idx1]
                if random.random() < self.pm:
                    idx1, idx2 = random.sample(range(total_ops), 2)
                    c2_os[idx1], c2_os[idx2] = c2_os[idx2], c2_os[idx1]
                if random.random() < self.pm:
                    c1_aa[random.randint(0, total_ops - 1)] ^= 1
                if random.random() < self.pm:
                    c2_aa[random.randint(0, total_ops - 1)] ^= 1

                new_pop.extend([(c1_os, c1_aa), (c2_os, c2_aa)])

            population = new_pop

        return best_makespan, best_sched, history


# ---------------------------------------------------------------------------
# 3. Standard PSO (Single-Swarm with SPV)
# ---------------------------------------------------------------------------

class StandardPSO:
    """Standard Single-Swarm PSO for JSSPLA."""
    def __init__(self, num_particles: int = 40, max_iter: int = 50, seed: int = 42):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.seed = seed

    def solve(self, instance: WarehouseInstance) -> Tuple[float, JSSPLASchedule, List[float]]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        decoder = JSSPLADecoder(instance)
        base_os = list(instance.base_sequence)
        dim = len(base_os)

        # Positions and velocities
        X_os = np.random.uniform(-4.0, 4.0, (self.num_particles, dim))
        V_os = np.random.uniform(-1.5, 1.5, (self.num_particles, dim))
        X_aa = np.random.uniform(0.0, 1.0, (self.num_particles, dim))
        V_aa = np.random.uniform(-0.5, 0.5, (self.num_particles, dim))

        pbest_X_os = X_os.copy()
        pbest_X_aa = X_aa.copy()
        pbest_val = np.zeros(self.num_particles)

        gbest_val = float("inf")
        gbest_sched = None
        gbest_X_os = None
        gbest_X_aa = None

        def decode_particle(x_o, x_a):
            ranks = np.argsort(x_o)
            os_seq = [base_os[r] for r in ranks]
            aa_seq = [int(v > 0.5) for v in x_a]
            return os_seq, aa_seq

        # Initial evaluation
        for i in range(self.num_particles):
            os_s, aa_s = decode_particle(X_os[i], X_aa[i])
            sched = decoder.decode(os_s, aa_s)
            pbest_val[i] = sched.makespan
            if sched.makespan < gbest_val:
                gbest_val = sched.makespan
                gbest_sched = sched
                gbest_X_os = X_os[i].copy()
                gbest_X_aa = X_aa[i].copy()

        history = [gbest_val]

        for it in range(self.max_iter):
            w = 0.9 - 0.5 * (it / max(1, self.max_iter - 1))
            c1, c2 = 1.5, 1.5

            r1 = np.random.rand(self.num_particles, dim)
            r2 = np.random.rand(self.num_particles, dim)

            # Update OS
            V_os = w * V_os + c1 * r1 * (pbest_X_os - X_os) + c2 * r2 * (gbest_X_os - X_os)
            V_os = np.clip(V_os, -2.0, 2.0)
            X_os = np.clip(X_os + V_os, -4.0, 4.0)

            # Update AA
            V_aa = w * V_aa + c1 * r1 * (pbest_X_aa - X_aa) + c2 * r2 * (gbest_X_aa - X_aa)
            V_aa = np.clip(V_aa, -1.0, 1.0)
            X_aa = np.clip(X_aa + V_aa, 0.0, 1.0)

            for i in range(self.num_particles):
                os_s, aa_s = decode_particle(X_os[i], X_aa[i])
                sched = decoder.decode(os_s, aa_s)
                if sched.makespan < pbest_val[i]:
                    pbest_val[i] = sched.makespan
                    pbest_X_os[i] = X_os[i].copy()
                    pbest_X_aa[i] = X_aa[i].copy()
                    if sched.makespan < gbest_val:
                        gbest_val = sched.makespan
                        gbest_sched = sched
                        gbest_X_os = X_os[i].copy()
                        gbest_X_aa = X_aa[i].copy()

            history.append(gbest_val)

        return gbest_val, gbest_sched, history


# ---------------------------------------------------------------------------
# 4. Proposed CDPSO (Cooperative Discrete PSO from Paper Algorithm 1)
# ---------------------------------------------------------------------------

class CDPSO:
    """
    Cooperative Discrete PSO as proposed in the Paper (Algorithm 1):
    - Dual Swarms: Machine Swarm (X) and AMR Swarm (Y).
    - Cooperative Evaluation:
      X_i.fit = Decoder(X_i.seq, gbest2)
      Y_j.fit = Decoder(gbest1, Y_j.seq)
    - Self-adaptive diversity control:
      P_pbest(t) = 0.6 - 0.4 * (t / Tmax)
      P_gbest(t) = 0.2 + 0.5 * (t / Tmax)
    - Update via Swap sequences.
    """
    def __init__(self, pop_size: int = 40, max_iter: int = 50, seed: int = 42):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.seed = seed

    @staticmethod
    def apply_swap_attraction(seq: list, target: list, prob: float) -> list:
        """Attracts discrete sequence toward target sequence via swap sequence."""
        res = list(seq)
        n = len(res)
        num_swaps = int(n * prob * 0.35)
        for _ in range(num_swaps):
            idx1 = random.randint(0, n - 1)
            # Find where target's element is located in current
            val_target = target[idx1]
            if res[idx1] != val_target:
                # Find matching index
                for idx2 in range(idx1 + 1, n):
                    if res[idx2] == val_target:
                        res[idx1], res[idx2] = res[idx2], res[idx1]
                        break
        return res

    @staticmethod
    def apply_bit_attraction(seq: list, target: list, prob: float) -> list:
        """Attracts binary AMR assignment toward target."""
        res = list(seq)
        for i in range(len(res)):
            if random.random() < prob:
                res[i] = target[i]
        return res

    def solve(self, instance: WarehouseInstance) -> Tuple[float, JSSPLASchedule, List[float]]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        decoder = JSSPLADecoder(instance)
        base_os = list(instance.base_sequence)
        total_ops = len(base_os)

        # 1. Initialize Machine Swarm X (OS)
        X = []
        pbest1 = []
        for _ in range(self.pop_size):
            s = list(base_os)
            random.shuffle(s)
            X.append(s)
            pbest1.append(list(s))

        # 2. Initialize AMR Swarm Y (AA)
        Y = []
        pbest2 = []
        for _ in range(self.pop_size):
            y = [random.randint(0, 1) for _ in range(total_ops)]
            Y.append(y)
            pbest2.append(list(y))

        # Initial gbest
        gbest1 = list(X[0])
        gbest2 = list(Y[0])
        initial_sched = decoder.decode(gbest1, gbest2)
        gbest_val = initial_sched.makespan
        best_sched = initial_sched

        pbest1_val = [float("inf")] * self.pop_size
        pbest2_val = [float("inf")] * self.pop_size

        history = [gbest_val]

        # Optimization loop
        for t in range(self.max_iter):
            # Dynamic self-adaptive control probabilities (Eq 17, 18)
            p_pbest = 0.6 - 0.4 * (t / max(1, self.max_iter))
            p_gbest = 0.2 + 0.5 * (t / max(1, self.max_iter))

            # --- A. Machine Swarm Update ---
            for i in range(self.pop_size):
                sched = decoder.decode(X[i], gbest2)
                score = sched.makespan
                if score < pbest1_val[i]:
                    pbest1_val[i] = score
                    pbest1[i] = list(X[i])
                    if score < gbest_val:
                        gbest_val = score
                        best_sched = sched
                        gbest1 = list(X[i])

            # --- B. AMR Swarm Update ---
            for j in range(self.pop_size):
                sched = decoder.decode(gbest1, Y[j])
                score = sched.makespan
                if score < pbest2_val[j]:
                    pbest2_val[j] = score
                    pbest2[j] = list(Y[j])
                    if score < gbest_val:
                        gbest_val = score
                        best_sched = sched
                        gbest2 = list(Y[j])

            # --- C. Update via Swap Sequence ---
            for i in range(self.pop_size):
                # Attract Machine sequence toward pbest1 and gbest1
                if random.random() < p_pbest:
                    X[i] = self.apply_swap_attraction(X[i], pbest1[i], p_pbest)
                if random.random() < p_gbest:
                    X[i] = self.apply_swap_attraction(X[i], gbest1, p_gbest)

            for j in range(self.pop_size):
                # Attract AMR assignment toward pbest2 and gbest2
                if random.random() < p_pbest:
                    Y[j] = self.apply_bit_attraction(Y[j], pbest2[j], p_pbest)
                if random.random() < p_gbest:
                    Y[j] = self.apply_bit_attraction(Y[j], gbest2, p_gbest)

            history.append(gbest_val)

        return gbest_val, best_sched, history
