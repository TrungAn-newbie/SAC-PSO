"""
PSO RL Environment and Solver for JSSPLA with Simplified State (6-dim) and Action (3-dim: w, c1, c2).
"""

import sys
import os
from typing import List, Tuple, Dict, Optional, Any
import random
import numpy as np
import torch

from warehouse_env import WarehouseInstance, JSSPLADecoder, JSSPLASchedule
from sac_agent import SACAgent


class BasicWarehousePSORLEnv:
    """
    RL Environment modeling dynamic execution of PSO for JSSPLA.
    Simplified State Space: 6 features.
    Action Space: 3 continuous parameters (w, c1, c2).
    """
    def __init__(
        self,
        instance: WarehouseInstance,
        num_particles: int = 30,
        max_iter: int = 25,
        seed: int = 42,
    ):
        self.instance = instance
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.seed = seed
        self.decoder = JSSPLADecoder(instance)
        self.base_os = list(instance.base_sequence)
        self.dim = len(self.base_os)

        self.current_iter = 0
        self.stagnation = 0
        self.gbest_val = float("inf")
        self.gbest_sched: Optional[JSSPLASchedule] = None
        self.gbest_X_os: Optional[np.ndarray] = None
        self.gbest_X_aa: Optional[np.ndarray] = None
        self.initial_gbest = float("inf")
        self.prev_gbest = float("inf")
        self.prev_mean_pbest = float("inf")

    def _decode_particle(self, x_o: np.ndarray, x_a: np.ndarray) -> Tuple[List[int], List[int]]:
        """Transforms continuous SPV particles to discrete OS and AA sequences."""
        ranks = np.argsort(x_o)
        os_seq = [self.base_os[r] for r in ranks]
        aa_seq = [int(v > 0.5) for v in x_a]
        return os_seq, aa_seq

    def _generate_pipeline_schedule(self) -> Tuple[List[int], List[int]]:
        """Generates a structured pipelined schedule seed."""
        out_jobs = [j.job_id for j in self.instance.jobs if j.job_type == "OUTBOUND"]
        in_jobs = [j.job_id for j in self.instance.jobs if j.job_type == "INBOUND"]
        random.shuffle(out_jobs)
        random.shuffle(in_jobs)

        n_out = len(out_jobs)
        n_in = len(in_jobs)

        if n_out <= 20:
            bsz_out = max(1, n_out // 4)
            os_seq = []
            for b in range(0, n_out, bsz_out):
                b_out = out_jobs[b : b + bsz_out]
                b_in = in_jobs[b * n_in // max(1, n_out) : (b + bsz_out) * n_in // max(1, n_out)]
                os_seq.extend(b_out)
                os_seq.extend(b_in)
                os_seq.extend(b_out)
                os_seq.extend(b_out)
        elif n_out <= 30:
            bsz_out = 6
            os_seq = []
            for b in range(0, n_out, bsz_out):
                b_out = out_jobs[b : b + bsz_out]
                b_in = in_jobs[b * n_in // max(1, n_out) : (b + bsz_out) * n_in // max(1, n_out)]
                os_seq.extend(b_out)
                os_seq.extend(b_in)
                os_seq.extend(b_out)
                os_seq.extend(b_out)
        else:
            bsz_out = 16
            os_seq = []
            for b in range(0, n_out, bsz_out):
                b_out = out_jobs[b : b + bsz_out]
                b_in = in_jobs[b * n_in // max(1, n_out) : (b + bsz_out) * n_in // max(1, n_out)]
                os_seq.extend(b_out)
                os_seq.extend(b_in)
                os_seq.extend(b_out)
                os_seq.extend(b_out)

        cluster_counters = {c: 0 for c in range(4)}
        aa_seq = []
        for j_id in os_seq:
            c = self.instance.jobs[j_id].cluster_id
            aa_seq.append(cluster_counters[c] % 2)
            cluster_counters[c] += 1

        return os_seq, aa_seq

    def _sequence_to_spv(self, target_os_seq: List[int], target_aa_seq: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Maps discrete sequences into continuous SPV representations."""
        dim = self.dim
        spv_os = np.zeros(dim)
        spv_aa = np.zeros(dim)

        job_indices: Dict[int, List[int]] = {}
        for idx, j_id in enumerate(self.base_os):
            job_indices.setdefault(j_id, []).append(idx)

        job_occ = {j_id: 0 for j_id in job_indices}
        for pos, j_id in enumerate(target_os_seq):
            occ = job_occ[j_id]
            if occ < len(job_indices[j_id]):
                orig_idx = job_indices[j_id][occ]
                job_occ[j_id] += 1
                spv_os[orig_idx] = (pos / max(1, dim)) * 6.0 - 3.0 + np.random.uniform(-0.02, 0.02)
                spv_aa[orig_idx] = 0.85 if target_aa_seq[pos] == 1 else 0.15

        return spv_os, spv_aa

    def reset(self, new_seed: Optional[int] = None) -> np.ndarray:
        """Resets the swarm and returns the initial 6-dimensional state."""
        if new_seed is not None:
            self.seed = new_seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.current_iter = 0
        self.stagnation = 0

        P = self.num_particles
        dim = self.dim

        # Continuous particles initialization
        self.X_os = np.random.uniform(-4.0, 4.0, (P, dim))
        self.V_os = np.random.uniform(-1.5, 1.5, (P, dim))
        self.X_aa = np.random.uniform(0.0, 1.0, (P, dim))
        self.V_aa = np.random.uniform(-0.5, 0.5, (P, dim))

        self.pbest_X_os = self.X_os.copy()
        self.pbest_X_aa = self.X_aa.copy()
        self.pbest_val = np.zeros(P)

        self.gbest_val = float("inf")
        self.gbest_sched = None
        self.gbest_X_os = None
        self.gbest_X_aa = None

        # Evaluate initial swarm
        for i in range(P):
            o_s, a_s = self._decode_particle(self.X_os[i], self.X_aa[i])
            sc = self.decoder.decode(o_s, a_s)
            self.pbest_val[i] = sc.makespan
            if sc.makespan < self.gbest_val:
                self.gbest_val = sc.makespan
                self.gbest_sched = sc
                self.gbest_X_os = self.X_os[i].copy()
                self.gbest_X_aa = self.X_aa[i].copy()

        # Non-destructive pipeline anchor injection
        n_seeds = max(1, int(P * 0.25))
        worst_indices = np.argsort(self.pbest_val)[-n_seeds:]
        for wi in worst_indices:
            pip_os, pip_aa = self._generate_pipeline_schedule()
            sc_pip = self.decoder.decode(pip_os, pip_aa)
            if sc_pip.makespan < self.pbest_val[wi]:
                spv_o, spv_a = self._sequence_to_spv(pip_os, pip_aa)
                self.X_os[wi] = spv_o
                self.X_aa[wi] = spv_a
                self.pbest_val[wi] = sc_pip.makespan
                self.pbest_X_os[wi] = spv_o
                self.pbest_X_aa[wi] = spv_a
                if sc_pip.makespan < self.gbest_val:
                    self.gbest_val = sc_pip.makespan
                    self.gbest_sched = sc_pip
                    self.gbest_X_os = spv_o
                    self.gbest_X_aa = spv_a

        self.initial_gbest = self.gbest_val
        self.prev_gbest = self.gbest_val
        self.prev_mean_pbest = float(np.mean(self.pbest_val))

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Calculates the simplified 6-dimensional state:
        0. progress: t / T_max in [0, 1]
        1. gbest_ratio: gbest_t / gbest_0 in (0, 1]
        2. diversity: std(pbest) / (mean(pbest) + eps)
        3. stagnation_ratio: stagnation / T_max in [0, 1]
        4. step_improvement: (prev_gbest - gbest) / prev_gbest in [0, 1]
        5. mean_pbest_ratio: (mean(pbest) - gbest) / (mean(pbest) + eps) in [0, 1]
        """
        prog = self.current_iter / max(1, self.max_iter)
        gbest_norm = self.gbest_val / max(1e-5, self.initial_gbest)
        mean_pbest = float(np.mean(self.pbest_val))
        std_pbest = float(np.std(self.pbest_val))

        div = std_pbest / max(1e-5, mean_pbest)
        stag_norm = min(1.0, self.stagnation / max(1, self.max_iter))
        imp_step = max(0.0, (self.prev_gbest - self.gbest_val) / max(1e-5, self.prev_gbest))
        mean_dist_ratio = max(0.0, (mean_pbest - self.gbest_val) / max(1e-5, mean_pbest))

        state = np.array([
            prog,
            gbest_norm,
            div,
            stag_norm,
            imp_step,
            mean_dist_ratio,
        ], dtype=np.float32)
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one PSO iteration with SAC-tuned parameters (w, c1, c2).
        action: array of 3 continuous values in [-1, 1]
        """
        prog = self.current_iter / max(1, self.max_iter - 1)

        # Dynamic parameter modulation using action
        w_base = 0.90 - 0.50 * prog
        w = float(np.clip(w_base + 0.25 * float(action[0]), 0.35, 0.95))

        c1_base = 1.60 - 0.30 * prog
        c1 = float(np.clip(c1_base + 0.40 * float(action[1]), 0.80, 2.60))

        c2_base = 1.40 + 0.40 * prog
        c2 = float(np.clip(c2_base + 0.40 * float(action[2]), 0.80, 2.60))

        dim = self.dim
        P = self.num_particles

        r1 = np.random.rand(P, dim)
        r2 = np.random.rand(P, dim)

        # Swarm velocity and position updates for OS
        self.V_os = np.clip(
            w * self.V_os + c1 * r1 * (self.pbest_X_os - self.X_os) + c2 * r2 * (self.gbest_X_os - self.X_os),
            -2.0, 2.0
        )
        self.X_os = np.clip(self.X_os + self.V_os, -4.0, 4.0)

        # Swarm velocity and position updates for AA
        self.V_aa = np.clip(
            w * self.V_aa + c1 * r1 * (self.pbest_X_aa - self.X_aa) + c2 * r2 * (self.gbest_X_aa - self.X_aa),
            -1.0, 1.0
        )
        self.X_aa = np.clip(self.X_aa + self.V_aa, 0.0, 1.0)

        # Evaluate all particles
        self.prev_gbest = self.gbest_val
        self.prev_mean_pbest = float(np.mean(self.pbest_val))
        improved = False

        for i in range(P):
            o_s, a_s = self._decode_particle(self.X_os[i], self.X_aa[i])
            sc = self.decoder.decode(o_s, a_s)
            if sc.makespan < self.pbest_val[i]:
                self.pbest_val[i] = sc.makespan
                self.pbest_X_os[i] = self.X_os[i].copy()
                self.pbest_X_aa[i] = self.X_aa[i].copy()
                if sc.makespan < self.gbest_val:
                    self.gbest_val = sc.makespan
                    self.gbest_sched = sc
                    self.gbest_X_os = self.X_os[i].copy()
                    self.gbest_X_aa = self.X_aa[i].copy()
                    improved = True

        if improved:
            self.stagnation = 0
        else:
            self.stagnation += 1

        self.current_iter += 1
        done = self.current_iter >= self.max_iter

        # Dense Reward Shaping
        delta_gbest = (self.prev_gbest - self.gbest_val) / max(1e-5, self.initial_gbest)
        curr_mean_pbest = float(np.mean(self.pbest_val))
        delta_mean_pbest = (self.prev_mean_pbest - curr_mean_pbest) / max(1e-5, self.initial_gbest)

        reward = 15.0 * delta_gbest + 3.0 * delta_mean_pbest - 0.02 * (self.stagnation / self.max_iter)
        if done:
            total_imp = (self.initial_gbest - self.gbest_val) / max(1e-5, self.initial_gbest)
            reward += 6.0 * total_imp

        next_state = self._get_state()
        info = {
            "gbest": self.gbest_val,
            "makespan": self.gbest_val,
            "w": w,
            "c1": c1,
            "c2": c2,
        }
        return next_state, reward, done, info


class BasicSACPSOSolver:
    """
    Standard Swarm PSO solver guided dynamically by the trained Basic SAC Agent (w, c1, c2).
    Includes elitist memetic refinement.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_particles: int = 30,
        max_iter: int = 25,
        seed: int = 42,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.seed = seed
        self.agent = SACAgent(state_dim=6, action_dim=3, device=device)

        if model_path is not None and os.path.exists(model_path):
            try:
                self.agent.load(model_path)
                if verbose:
                    print(f"[BasicSACPSOSolver] Loaded trained weights from: {model_path}")
            except Exception as e:
                if verbose:
                    print(f"[BasicSACPSOSolver] Model load notice: {e}. Using active policy.")
        else:
            if verbose:
                print("[BasicSACPSOSolver] Initialized with active policy.")

    def _refine_solution(
        self,
        instance: WarehouseInstance,
        decoder: JSSPLADecoder,
        base_os: List[int],
        gbest_val: float,
        gbest_sched: JSSPLASchedule,
        gbest_X_os: np.ndarray,
        gbest_X_aa: np.ndarray,
    ) -> Tuple[float, JSSPLASchedule, np.ndarray, np.ndarray]:
        """
        Monotonic elitist memetic refinement:
        1. Greedy AMR Workload Balancing across 4 clusters.
        2. Critical-Path Job Shifting on bottleneck operations.
        """
        dim = len(base_os)
        cand_aa = [int(v > 0.5) for v in gbest_X_aa]
        ranks = list(np.argsort(gbest_X_os))
        best_val = gbest_val
        best_sched = gbest_sched

        # 1. Multi-pass Greedy AMR Workload Balancing
        for _ in range(4):
            improved = False
            for c in range(4):
                r1_id, r2_id = 2 * c, 2 * c + 1
                t1 = best_sched.amr_completion_times[r1_id]
                t2 = best_sched.amr_completion_times[r2_id]
                if abs(t1 - t2) > 3.0:
                    heavier = 0 if t1 > t2 else 1
                    lighter = 1 - heavier
                    cands = [
                        idx for idx in range(dim)
                        if instance.jobs[base_os[idx]].cluster_id == c
                        and cand_aa[idx] == heavier
                    ]
                    for cand_idx in cands:
                        test_aa = cand_aa.copy()
                        test_aa[cand_idx] = lighter
                        os_c = [base_os[rk] for rk in ranks]
                        sc_c = decoder.decode(os_c, test_aa)
                        if sc_c.makespan < best_val:
                            best_val = sc_c.makespan
                            best_sched = sc_c
                            cand_aa = test_aa
                            improved = True
            if not improved:
                break

        # 2. Critical Path Bottleneck Job Shift
        latest_jobs = sorted(best_sched.job_completion_times.items(), key=lambda x: x[1], reverse=True)
        k_evals = 0
        for j_id, _ in latest_jobs[:min(8, len(latest_jobs))]:
            pos_in_ranks = [i for i, rk in enumerate(ranks) if base_os[rk] == j_id]
            for p_idx in pos_in_ranks:
                for shift in [1, 2, 3]:
                    if p_idx >= shift:
                        cand_ranks = list(ranks)
                        cand_ranks[p_idx], cand_ranks[p_idx - shift] = cand_ranks[p_idx - shift], cand_ranks[p_idx]
                        cand_os = [base_os[rk] for rk in cand_ranks]
                        sc_cand = decoder.decode(cand_os, cand_aa)
                        k_evals += 1
                        if sc_cand.makespan < best_val:
                            best_val = sc_cand.makespan
                            best_sched = sc_cand
                            ranks = cand_ranks
                            break
            if k_evals >= 20:
                break

        new_X_os = np.empty(dim)
        new_X_os[ranks] = np.linspace(-3.0, 3.0, dim)
        new_X_aa = np.array([0.85 if v == 1 else 0.15 for v in cand_aa])

        return best_val, best_sched, new_X_os, new_X_aa

    def solve(
        self,
        instance: WarehouseInstance,
        use_refinement: bool = True,
    ) -> Tuple[float, JSSPLASchedule, List[float], List[Dict[str, float]]]:
        """
        Solves JSSPLA for an instance using Basic SAC-guided PSO.
        """
        env = BasicWarehousePSORLEnv(
            instance=instance,
            num_particles=self.num_particles,
            max_iter=self.max_iter,
            seed=self.seed,
        )
        state = env.reset()
        history = [env.gbest_val]
        actions_hist = []

        done = False
        while not done:
            action = self.agent.select_action(state, evaluate=True)
            state, reward, done, info = env.step(action)
            history.append(env.gbest_val)
            actions_hist.append({
                "w": info["w"],
                "c1": info["c1"],
                "c2": info["c2"],
            })

        if use_refinement and env.gbest_sched is not None and env.gbest_X_os is not None and env.gbest_X_aa is not None:
            ref_val, ref_sched, ref_X_os, ref_X_aa = self._refine_solution(
                instance=instance,
                decoder=env.decoder,
                base_os=env.base_os,
                gbest_val=env.gbest_val,
                gbest_sched=env.gbest_sched,
                gbest_X_os=env.gbest_X_os,
                gbest_X_aa=env.gbest_X_aa,
            )
            if ref_val < env.gbest_val:
                env.gbest_val = ref_val
                env.gbest_sched = ref_sched
                env.gbest_X_os = ref_X_os
                env.gbest_X_aa = ref_X_aa
                history[-1] = ref_val

        return env.gbest_val, env.gbest_sched, history, actions_hist
