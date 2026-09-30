"""
SAC-PSO: Soft Actor-Critic Dynamic Meta-Optimization for Standard PSO
applied to Job Shop Scheduling with Limited AMRs (JSSPLA).

Based on the warehouse scheduling problem from:
"Job Shop-Based Scheduling Optimization for Multi-AMR Warehouse Systems
and Validation via Gazebo Simulation" (Supervisor: Dr. Truong Ngoc Cuong, Student: Tran Viet Trung An)

Architecture:
1. Standard Single-Swarm PSO with continuous representation:
   - OS (Operation Sequence): SPV (Smallest Position Value) ranking on continuous vector X_os.
   - AA (AMR Assignment): Thresholding on continuous vector X_aa.
2. SAC (Soft Actor-Critic) Agent:
   - State (12-dim): Swarm convergence, diversity, progress, stagnation, scale.
   - Action (6-dim): Dynamically modulates:
     * w: Inertia weight [0.20, 0.95]
     * c1: Cognitive acceleration [0.40, 2.60]
     * c2: Social acceleration [0.40, 2.60]
     * v_max: Adaptive velocity clamp [1.0, 3.5]
     * p_perturb: Anti-stagnation diversity kick rate [0.0, 0.30]
     * p_balance: AMR workload rebalancing rate [0.0, 0.90]
3. Multi-instance RL training across small, medium, and large warehouse instances.
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
    def __init__(self, state_dim: int = 12, action_dim: int = 6, capacity: int = 50000):
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
    def __init__(self, state_dim: int = 12, action_dim: int = 6, hidden_dim: int = 128):
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
    def __init__(self, state_dim: int = 12, action_dim: int = 6, hidden_dim: int = 128):
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
        state_dim: int = 12,
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
# 2. Warehouse PSO RL Environment
# ---------------------------------------------------------------------------

class WarehousePSORLEnv:
    """
    RL Environment modeling the dynamic execution of Standard PSO
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

        # Continuous particles
        self.X_os = np.random.uniform(-4.0, 4.0, (self.num_particles, self.dim))
        self.V_os = np.random.uniform(-1.5, 1.5, (self.num_particles, self.dim))
        self.X_aa = np.random.uniform(0.0, 1.0, (self.num_particles, self.dim))
        self.V_aa = np.random.uniform(-0.5, 0.5, (self.num_particles, self.dim))

        self.pbest_X_os = self.X_os.copy()
        self.pbest_X_aa = self.X_aa.copy()
        self.pbest_val = np.zeros(self.num_particles)

        self.gbest_val = float("inf")
        self.gbest_sched = None
        self.gbest_X_os = None
        self.gbest_X_aa = None

        for i in range(self.num_particles):
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
        self.last_action = np.zeros(6, dtype=np.float32)

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        progress = self.current_iter / max(1, self.max_iter)
        gbest_norm = self.gbest_val / max(1e-5, self.initial_gbest)
        mean_pbest = float(np.mean(self.pbest_val))
        mean_norm = mean_pbest / max(1e-5, self.initial_gbest)
        div = float(np.std(self.pbest_val)) / max(1e-5, mean_pbest)
        stag_norm = min(1.0, self.stagnation / max(1, self.max_iter))

        imp_step = max(0.0, (self.prev_gbest - self.gbest_val) / max(1e-5, self.prev_gbest))
        imp_swarm = max(0.0, (self.prev_mean_pbest - mean_pbest) / max(1e-5, self.prev_mean_pbest))

        vel_norm = float(np.mean(np.abs(self.V_os))) / 2.0
        pos_spread = float(np.std(self.X_os)) / 4.0
        scale = min(1.0, self.dim / 750.0)
        n_in = sum(1 for j in self.instance.jobs if j.job_type == "INBOUND")
        in_ratio = n_in / max(1, self.instance.num_jobs)

        state = np.array([
            progress,
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
            float(self.last_action[0]),  # previous w
        ], dtype=np.float32)
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.last_action = action.copy()
        prog = self.current_iter / max(1, self.max_iter)

        # Adaptive baseline parameters with SAC residual modulation
        if self.stagnation > 1:
            w_base = 0.82
            c1_base = 1.90
            c2_base = 1.10
            v_max_base = 2.40
        else:
            w_base = 0.90 - 0.55 * (prog ** 1.1)
            c1_base = 1.80 * (1.0 - prog) + 0.50
            c2_base = 0.50 * (1.0 - prog) + 2.20
            v_max_base = 2.00

        w = float(np.clip(w_base + 0.15 * action[0], 0.20, 0.95))
        c1 = float(np.clip(c1_base + 0.35 * action[1], 0.40, 2.60))
        c2 = float(np.clip(c2_base + 0.35 * action[2], 0.40, 2.60))
        v_max = float(np.clip(v_max_base + 0.60 * action[3], 1.20, 3.20))
        p_perturb = float(0.15 + 0.15 * action[4])
        p_balance = float(0.45 + 0.45 * action[5])

        dim = self.dim
        P = self.num_particles

        # Update all particles
        r1 = np.random.rand(P, dim)
        r2 = np.random.rand(P, dim)

        for i in range(P):
            # OS velocity & position
            self.V_os[i] = w * self.V_os[i] + c1 * r1[i] * (self.pbest_X_os[i] - self.X_os[i]) + c2 * r2[i] * (self.gbest_X_os - self.X_os[i])
            self.V_os[i] = np.clip(self.V_os[i], -v_max, v_max)
            self.X_os[i] = np.clip(self.X_os[i] + self.V_os[i], -4.0, 4.0)

            # AA velocity & position
            self.V_aa[i] = w * self.V_aa[i] + c1 * r1[i] * (self.pbest_X_aa[i] - self.X_aa[i]) + c2 * r2[i] * (self.gbest_X_aa - self.X_aa[i])
            self.V_aa[i] = np.clip(self.V_aa[i], -1.0, 1.0)
            self.X_aa[i] = np.clip(self.X_aa[i] + self.V_aa[i], 0.0, 1.0)

        # Anti-stagnation high-precision adjacent perturbation
        if p_perturb > 0.06 and self.stagnation > 1:
            num_kick = max(1, min(P - 1, int(P * p_perturb)))
            for p_i in range(1, num_kick + 1):
                ranks = np.argsort(self.X_os[p_i])
                idx = random.randint(0, dim - 2)
                self.X_os[p_i, ranks[idx]], self.X_os[p_i, ranks[idx+1]] = (
                    self.X_os[p_i, ranks[idx+1]],
                    self.X_os[p_i, ranks[idx]],
                )
                if random.random() < 0.2:
                    self.X_aa[p_i, ranks[idx]] = 1.0 - self.X_aa[p_i, ranks[idx]]

        # Evaluate particles
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

        # Dynamic AMR Workload Balancing (active on medium and large instances)
        if dim > 50 and p_balance > 0.15 and self.gbest_sched is not None:
            cand_aa = self.gbest_X_aa.copy()
            changed = False
            diff_thresh = max(50.0, self.gbest_val * 0.03)

            for c in range(4):
                r1_id, r2_id = 2 * c, 2 * c + 1
                t1 = self.gbest_sched.amr_completion_times[r1_id]
                t2 = self.gbest_sched.amr_completion_times[r2_id]
                if abs(t1 - t2) > diff_thresh:
                    heavier = 0 if t1 > t2 else 1
                    lighter = 1 - heavier
                    cands = [
                        idx for idx in range(dim)
                        if self.instance.jobs[self.base_os[idx]].cluster_id == c
                        and int(cand_aa[idx] > 0.5) == heavier
                    ]
                    if cands:
                        num_flip = min(len(cands), random.randint(1, 2))
                        for f in random.sample(cands, num_flip):
                            cand_aa[f] = 0.85 if lighter == 1 else 0.15
                        changed = True

            if changed:
                o_c, a_c = self._decode_particle(self.gbest_X_os, cand_aa)
                sc_c = self.decoder.decode(o_c, a_c)
                if sc_c.makespan < self.gbest_val:
                    self.gbest_val = sc_c.makespan
                    self.gbest_sched = sc_c
                    self.gbest_X_aa = cand_aa
                    improved = True

        if improved:
            self.stagnation = 0
        else:
            self.stagnation += 1

        self.current_iter += 1
        done = self.current_iter >= self.max_iter

        # Reward shaping
        delta_gbest = (self.prev_gbest - self.gbest_val) / max(1e-5, self.initial_gbest)
        curr_mean = float(np.mean(self.pbest_val))
        delta_mean = (self.prev_mean_pbest - curr_mean) / max(1e-5, self.initial_gbest)

        reward = 12.0 * delta_gbest + 2.0 * delta_mean - 0.04 * (self.stagnation / self.max_iter)
        if done:
            total_imp = (self.initial_gbest - self.gbest_val) / max(1e-5, self.initial_gbest)
            reward += 5.0 * total_imp

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
    Standard Single-Swarm PSO guided dynamically by a trained SAC Agent.
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
        self.agent = SACAgent(state_dim=12, action_dim=6, device=device)

        if model_path is not None and os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"[SACPSOSolver] Loaded trained weights from: {model_path}")
        else:
            print("[SACPSOSolver] Initialized with default policy.")

    def solve(
        self,
        instance: WarehouseInstance,
    ) -> Tuple[float, JSSPLASchedule, List[float], List[Dict[str, float]]]:
        """
        Solves JSSPLA for a given instance using SAC-modulated Standard PSO.
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
