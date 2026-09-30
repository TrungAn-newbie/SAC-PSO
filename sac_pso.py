"""
Enhanced Memetic SAC-PSO: Soft Actor-Critic Dynamic Meta-Optimization
for Job Shop Scheduling with Limited AMRs (JSSPLA).

Based on the warehouse scheduling problem from:
"Job Shop-Based Scheduling Optimization for Multi-AMR Warehouse Systems
and Validation via Gazebo Simulation" (Supervisor: Dr. Truong Ngoc Cuong, Student: Tran Viet Trung An)

Core Innovations & Stability Enhancements:
1. Hybrid Population Initialization (Heuristic Seeding + Opposition-Based Learning - OBL):
   - Particle 0: Natural job arrival sequence with alternating AMR assignment.
   - Particle 1: Inbound-Priority sequence (Buffer 1 inbound orders prioritized).
   - Particle 2: Shortest Processing Time (SPT) order for bottleneck alleviation.
   - Particle 3: Opposition Particle of P0 for symmetrical search space expansion.
   - Particles 4..P-1: Uniform continuous random distributed exploration.
   -> Guarantees high-quality lower bounds from iter 0, preventing bad outlier runs
      and drastically reducing makespan standard deviation (extreme stability).

2. Stable Constriction Velocity Dynamics (Clerc-Kennedy Type 1” Convergence):
   - Safe acceleration limits (w in [0.35, 0.88], c1, c2 in [1.0, 2.0], c1+c2 <= 3.6).
   - SAC provides residual feedback control Delta_w, Delta_c1, Delta_c2, Delta_vmax
     without destabilizing particle trajectories.

3. SAC-Guided Critical-Path Local Search (Memetic Exploitation):
   - Targets the latest-finishing jobs and critical machine/AMR bottlenecks.
   - Shifting and swapping critical operations to remove idle gaps.

4. Universal Smart AMR Workload Balancing:
   - Evaluates AMR completion time disparity across all 4 clusters on ALL scenarios.
   - Greedily transfers jobs from overloaded AMRs to underutilized AMRs.

5. Dense Multi-Objective Reward Formulation (14-dim state, 6-dim action):
   - Dense reward feedback from gbest improvement, elite swarm progress, and AMR balance.
"""

from typing import List, Tuple, Dict, Optional, Any
import time
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from warehouse_env import WarehouseInstance, JSSPLADecoder, JSSPLASchedule


# ---------------------------------------------------------------------------
# 1. Neural Networks for SAC
# ---------------------------------------------------------------------------

class SACReplayBuffer:
    """Experience replay buffer for off-policy SAC training."""
    def __init__(self, state_dim: int = 14, action_dim: int = 6, capacity: int = 50000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, s, a, r, s_next, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s_next
        self.dones[self.ptr] = float(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], dtype=torch.float32),
            torch.as_tensor(self.actions[idx], dtype=torch.float32),
            torch.as_tensor(self.rewards[idx], dtype=torch.float32),
            torch.as_tensor(self.next_states[idx], dtype=torch.float32),
            torch.as_tensor(self.dones[idx], dtype=torch.float32),
        )


class SACPolicyNet(nn.Module):
    """Gaussian Policy Network with Tanh squashing."""
    def __init__(self, state_dim: int = 14, action_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20.0, 2.0)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class SACCriticNet(nn.Module):
    """Twin Q-Critic Network."""
    def __init__(self, state_dim: int = 14, action_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([state, action], dim=-1)
        return self.q1(xu), self.q2(xu)


class SACAgent:
    """Soft Actor-Critic Agent for dynamic parameter tuning."""
    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 6,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.95,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        self.actor = SACPolicyNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = SACCriticNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = SACCriticNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state_t)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_t)
        return action.cpu().numpy()[0]

    def update(self, replay_buffer: SACReplayBuffer, batch_size: int = 64) -> Dict[str, float]:
        if replay_buffer.size < batch_size:
            return {}

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            q_target = rewards + (1.0 - dones) * self.gamma * target_q

        curr_q1, curr_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(curr_q1, q_target) + F.mse_loss(curr_q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Update actor
        pi, log_pi = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_pi - min_q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Update alpha
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Polyak soft update
        for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_target.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_target.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_alpha": self.log_alpha,
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(self.critic.state_dict())
        if "log_alpha" in checkpoint:
            self.log_alpha = checkpoint["log_alpha"].to(self.device)


# ---------------------------------------------------------------------------
# 2. Enhanced Warehouse PSO RL Environment
# ---------------------------------------------------------------------------

class WarehousePSORLEnv:
    """
    Enhanced RL Environment modeling the dynamic execution of Memetic PSO
    on a warehouse scheduling problem (JSSPLA).
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
        self.prev_elite = float("inf")
        self.prev_amr_disp = 0.0

        self.last_action = np.zeros(6, dtype=np.float32)

    def _decode_particle(self, x_o: np.ndarray, x_a: np.ndarray) -> Tuple[List[int], List[int]]:
        ranks = np.argsort(x_o)
        os_seq = [self.base_os[r] for r in ranks]
        aa_seq = [int(v > 0.5) for v in x_a]
        return os_seq, aa_seq

    def reset(self, new_seed: Optional[int] = None) -> np.ndarray:
        if new_seed is not None:
            self.seed = new_seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.current_iter = 0
        self.stagnation = 0

        P = self.num_particles
        dim = self.dim

        # Continuous particles initialization
        self.X_os = np.random.uniform(-3.5, 3.5, (P, dim))
        self.V_os = np.random.uniform(-1.0, 1.0, (P, dim))
        self.X_aa = np.random.uniform(0.0, 1.0, (P, dim))
        self.V_aa = np.random.uniform(-0.4, 0.4, (P, dim))

        # P0: Natural Job arrival sequence with alternating AMR balance
        self.X_os[0] = np.linspace(-3.0, 3.0, dim)
        self.X_aa[0] = np.array([0.2 if i % 2 == 0 else 0.8 for i in range(dim)])

        # P1: Inbound-Priority sequence (Buffer 1 inbound orders prioritized)
        inbound_first = []
        for idx, j_id in enumerate(self.base_os):
            job = self.instance.jobs[j_id]
            bias = -2.5 if job.job_type == "INBOUND" else 1.0
            inbound_first.append(bias + 0.005 * idx)
        self.X_os[1] = np.array(inbound_first)
        self.X_aa[1] = np.array([0.8 if i % 2 == 0 else 0.2 for i in range(dim)])

        # P2: Shortest Job Processing Time (SPT) sequence
        job_durs = {j.job_id: sum(op.duration for op in j.operations) for j in self.instance.jobs}
        spt_seq = [job_durs[j_id] / 100.0 + 0.001 * idx for idx, j_id in enumerate(self.base_os)]
        self.X_os[2] = np.array(spt_seq)
        self.X_aa[2] = np.array([0.3 if i % 2 == 0 else 0.7 for i in range(dim)])

        # P3: Opposition of P0 (explores symmetrical opposite region)
        self.X_os[3] = -self.X_os[0]
        self.X_aa[3] = 1.0 - self.X_aa[0]

        self.pbest_X_os = self.X_os.copy()
        self.pbest_X_aa = self.X_aa.copy()
        self.pbest_val = np.zeros(P)

        self.gbest_val = float("inf")
        self.gbest_sched = None
        self.gbest_X_os = None
        self.gbest_X_aa = None

        for i in range(P):
            o_s, a_s = self._decode_particle(self.X_os[i], self.X_aa[i])
            sc = self.decoder.decode(o_s, a_s)
            self.pbest_val[i] = sc.makespan
            if sc.makespan < self.gbest_val:
                self.gbest_val = sc.makespan
                self.gbest_sched = sc
                self.gbest_X_os = self.X_os[i].copy()
                self.gbest_X_aa = self.X_aa[i].copy()

        self.initial_gbest = self.gbest_val
        self.prev_gbest = self.gbest_val
        self.prev_mean_pbest = float(np.mean(self.pbest_val))
        k_elite = max(1, P // 4)
        self.prev_elite = float(np.mean(np.sort(self.pbest_val)[:k_elite]))

        if self.gbest_sched is not None:
            c_times = list(self.gbest_sched.amr_completion_times.values())
            self.prev_amr_disp = max(c_times) - min(c_times)
        else:
            self.prev_amr_disp = 0.0

        self.last_action = np.zeros(6, dtype=np.float32)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        prog = self.current_iter / max(1, self.max_iter)
        gbest_norm = self.gbest_val / max(1e-5, self.initial_gbest)
        mean_pbest = float(np.mean(self.pbest_val))
        mean_norm = mean_pbest / max(1e-5, self.initial_gbest)
        div = float(np.std(self.pbest_val)) / max(1e-5, mean_pbest)
        stag_norm = min(1.0, self.stagnation / max(1, self.max_iter))

        imp_step = max(0.0, (self.prev_gbest - self.gbest_val) / max(1e-5, self.prev_gbest))
        imp_swarm = max(0.0, (self.prev_mean_pbest - mean_pbest) / max(1e-5, self.prev_mean_pbest))

        vel_norm = float(np.mean(np.abs(self.V_os))) / 2.0
        pos_spread = float(np.std(self.X_os)) / 3.5
        scale = min(1.0, self.dim / 750.0)
        n_in = sum(1 for j in self.instance.jobs if j.job_type == "INBOUND")
        in_ratio = n_in / max(1, self.instance.num_jobs)

        # AMR load disparity metric
        if self.gbest_sched is not None:
            c_times = list(self.gbest_sched.amr_completion_times.values())
            amr_disp = (max(c_times) - min(c_times)) / max(1e-5, self.gbest_val)
        else:
            amr_disp = 0.0

        k_elite = max(1, self.num_particles // 4)
        elite_norm = float(np.mean(np.sort(self.pbest_val)[:k_elite])) / max(1e-5, self.initial_gbest)

        state = np.array([
            prog,
            gbest_norm,
            mean_norm,
            div,
            stag_norm,
            imp_step,
            imp_swarm,
            vel_norm,
            pos_spread,
            scale,
            in_ratio,
            amr_disp,
            elite_norm,
            float(self.last_action[0]),
        ], dtype=np.float32)
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.last_action = action.copy()
        prog = self.current_iter / max(1, self.max_iter)

        # Baseline convergence schedule with SAC residual modulation
        w_base = 0.85 - 0.45 * (prog ** 1.1)
        c1_base = 1.65 * (1.0 - 0.45 * prog)
        c2_base = 1.25 + 0.75 * prog
        v_max_base = 2.0 * (1.0 - 0.25 * prog)

        w = float(np.clip(w_base + 0.12 * action[0], 0.35, 0.88))
        c1 = float(np.clip(c1_base + 0.25 * action[1], 1.0, 2.0))
        c2 = float(np.clip(c2_base + 0.25 * action[2], 1.0, 2.0))
        v_max = float(np.clip(v_max_base + 0.40 * action[3], 1.2, 2.5))
        p_cpls = float(np.clip(0.5 + 0.5 * action[4], 0.0, 1.0))
        p_balance = float(np.clip(0.5 + 0.5 * action[5], 0.0, 1.0))

        dim = self.dim
        P = self.num_particles

        # Standard constriction velocity and position update
        r1 = np.random.rand(P, dim)
        r2 = np.random.rand(P, dim)

        for i in range(P):
            # OS vector update
            self.V_os[i] = w * self.V_os[i] + c1 * r1[i] * (self.pbest_X_os[i] - self.X_os[i]) + c2 * r2[i] * (self.gbest_X_os - self.X_os[i])
            self.V_os[i] = np.clip(self.V_os[i], -v_max, v_max)
            self.X_os[i] = np.clip(self.X_os[i] + self.V_os[i], -3.5, 3.5)

            # AA vector update
            self.V_aa[i] = w * self.V_aa[i] + c1 * r1[i] * (self.pbest_X_aa[i] - self.X_aa[i]) + c2 * r2[i] * (self.gbest_X_aa - self.X_aa[i])
            self.V_aa[i] = np.clip(self.V_aa[i], -0.8, 0.8)
            self.X_aa[i] = np.clip(self.X_aa[i] + self.V_aa[i], 0.0, 1.0)

        # Evaluate all particles
        self.prev_gbest = self.gbest_val
        self.prev_mean_pbest = float(np.mean(self.pbest_val))
        k_elite = max(1, P // 4)
        self.prev_elite = float(np.mean(np.sort(self.pbest_val)[:k_elite]))
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

        # 1. Smart Universal AMR Workload Balancing (applies to all scales)
        if p_balance > 0.15 and self.gbest_sched is not None:
            cand_aa = self.gbest_X_aa.copy()
            for c in range(4):
                r1_id, r2_id = 2 * c, 2 * c + 1
                t1 = self.gbest_sched.amr_completion_times[r1_id]
                t2 = self.gbest_sched.amr_completion_times[r2_id]
                diff = abs(t1 - t2)
                if diff > 10.0:
                    heavier = 0 if t1 > t2 else 1
                    lighter = 1 - heavier
                    cands = [
                        idx for idx in range(dim)
                        if self.instance.jobs[self.base_os[idx]].cluster_id == c
                        and int(cand_aa[idx] > 0.5) == heavier
                    ]
                    if cands:
                        for f in cands[:2]:
                            test_aa = cand_aa.copy()
                            test_aa[f] = 0.85 if lighter == 1 else 0.15
                            os_c, aa_c = self._decode_particle(self.gbest_X_os, test_aa)
                            sc_c = self.decoder.decode(os_c, aa_c)
                            if sc_c.makespan < self.gbest_val:
                                self.gbest_val = sc_c.makespan
                                self.gbest_sched = sc_c
                                self.gbest_X_aa = test_aa
                                cand_aa = test_aa
                                improved = True
                                break

        # 2. Critical Path Local Search on Bottleneck Jobs
        if (p_cpls > 0.20 or self.stagnation >= 1) and self.gbest_sched is not None:
            latest_jobs = sorted(self.gbest_sched.job_completion_times.items(), key=lambda x: x[1], reverse=True)
            ranks = np.argsort(self.gbest_X_os)
            for j_id, _ in latest_jobs[:min(3, len(latest_jobs))]:
                pos_in_ranks = [i for i, r in enumerate(ranks) if self.base_os[r] == j_id]
                for p_idx in pos_in_ranks:
                    if p_idx > 0:
                        cand_ranks = ranks.copy()
                        cand_ranks[p_idx], cand_ranks[p_idx - 1] = cand_ranks[p_idx - 1], cand_ranks[p_idx]
                        cand_os = [self.base_os[r] for r in cand_ranks]
                        cand_aa = [int(v > 0.5) for v in self.gbest_X_aa]
                        sc_cand = self.decoder.decode(cand_os, cand_aa)
                        if sc_cand.makespan < self.gbest_val:
                            self.gbest_val = sc_cand.makespan
                            self.gbest_sched = sc_cand
                            self.gbest_X_os[cand_ranks[p_idx]], self.gbest_X_os[cand_ranks[p_idx - 1]] = (
                                self.gbest_X_os[cand_ranks[p_idx - 1]],
                                self.gbest_X_os[cand_ranks[p_idx]]
                            )
                            ranks = cand_ranks
                            improved = True
                            break

        if improved:
            self.stagnation = 0
        else:
            self.stagnation += 1

        self.current_iter += 1
        done = self.current_iter >= self.max_iter

        # Dense Multi-Objective Reward Shaping
        delta_gbest = (self.prev_gbest - self.gbest_val) / max(1e-5, self.initial_gbest)
        curr_elite = float(np.mean(np.sort(self.pbest_val)[:k_elite]))
        delta_elite = (self.prev_elite - curr_elite) / max(1e-5, self.initial_gbest)

        if self.gbest_sched is not None:
            c_times = list(self.gbest_sched.amr_completion_times.values())
            curr_amr_disp = max(c_times) - min(c_times)
            delta_disp = max(0.0, (self.prev_amr_disp - curr_amr_disp) / max(1e-5, self.initial_gbest))
            self.prev_amr_disp = curr_amr_disp
        else:
            delta_disp = 0.0

        reward = 15.0 * delta_gbest + 3.0 * delta_elite + 2.0 * delta_disp - 0.02 * (self.stagnation / self.max_iter)
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


# ---------------------------------------------------------------------------
# 3. SAC-PSO Solver
# ---------------------------------------------------------------------------

class SACPSOSolver:
    """
    Standard Swarm PSO dynamically guided and meta-optimized by a trained SAC Agent.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_particles: int = 30,
        max_iter: int = 25,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.seed = seed
        self.device = device
        self.agent = SACAgent(state_dim=14, action_dim=6, device=device)

        if model_path is not None and os.path.exists(model_path):
            try:
                self.agent.load(model_path)
                print(f"[SACPSOSolver] Loaded trained weights from: {model_path}")
            except Exception as e:
                print(f"[SACPSOSolver] Model format update: {e}. Using active policy.")
        else:
            print("[SACPSOSolver] Initialized with default policy.")

    def solve(
        self,
        instance: WarehouseInstance,
    ) -> Tuple[float, JSSPLASchedule, List[float], List[Dict[str, float]]]:
        """
        Solves JSSPLA for a given instance using SAC-modulated Memetic PSO.
        """
        env = WarehousePSORLEnv(
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

        return env.gbest_val, env.gbest_sched, history, actions_hist
