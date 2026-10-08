"""
Basic Soft Actor-Critic (SAC) Implementation.
Optimized for 6-dimensional State and 3-dimensional Continuous Action (w, c1, c2).
"""

import os
from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class SACReplayBuffer:
    """Experience replay buffer for off-policy SAC training."""
    def __init__(self, state_dim: int = 6, action_dim: int = 3, capacity: int = 50000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, s: np.ndarray, a: np.ndarray, r: float, s_next: np.ndarray, d: bool):
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
    """Gaussian Policy Network with Tanh Squashing."""
    def __init__(self, state_dim: int = 6, action_dim: int = 3, hidden_dim: int = 128):
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
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        # Log probability computation with tanh squashing correction
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class SACCriticNet(nn.Module):
    """Twin Q-Critic Network to prevent overestimation bias."""
    def __init__(self, state_dim: int = 6, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Q2 network
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
    """
    Soft Actor-Critic Agent for dynamic parameter tuning (w, c1, c2).
    """
    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.95,
        tau: float = 0.005,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        # Actor and Critic
        self.actor = SACPolicyNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = SACCriticNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = SACCriticNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy temperature tuning
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> torch.Tensor:
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

        # 1. Update Critic
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

        # 2. Update Actor
        pi, log_pi = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_pi - min_q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 3. Update Alpha (Entropy Temperature)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # 4. Polyak soft target update
        for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_target.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_target.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
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
