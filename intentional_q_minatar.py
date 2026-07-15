import os, pickle, argparse
import torch
import numpy as np
import torch.nn as nn

import gymnasium as gym
from minatar.gym import register_envs
register_envs()
from optimizer import IntentionalOptimizerValue as Optimizer
import torch.nn.functional as F
from normalization_wrappers import NormalizeObservation, ScaleReward
from sparse_init import sparse_init
from metrics_logger import MetricsLogger, has_nonfinite_value
from precision_utils import DTYPE_CHOICES, DEVICE_CHOICES, default_dtype_tag, resolve_device, resolve_dtype

def _tensor_has_nan(tensor):
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return False
    return torch.isnan(tensor).any().item()

def _tensor_has_inf(tensor):
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return False
    return torch.isinf(tensor).any().item()

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class LayerNormalization(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return F.layer_norm(input, input.size())
    def extra_repr(self) -> str:
        return "Layer Normalization"

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

class IntentionalQ(nn.Module):
    def __init__(self, n_channels=4, n_actions=3, hidden_size=128, epsilon_target=0.01, epsilon_start=1.0, exploration_fraction=0.1, total_steps=1_000_000, gamma=0.99, lamda=0.8, eta_value=0.5, dtype=torch.float32, device=torch.device("cpu")):
        super(IntentionalQ, self).__init__()
        self.dtype = dtype
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_target = epsilon_target
        self.epsilon = epsilon_start
        self.exploration_fraction = exploration_fraction
        self.total_steps = total_steps
        self.time_step = 0
        self.network = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, stride=1),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=0),
            nn.Linear(1024, hidden_size),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, n_actions)
        )
        self.apply(initialize_weights)
        self.to(device=self.device, dtype=self.dtype)
        self.optimizer = Optimizer(list(self.parameters()), gamma=gamma, lamda=lamda, eta=eta_value)

    def q(self, x):
        x = x.to(device=self.device, dtype=self.dtype)
        x = torch.moveaxis(x, -1, 0)
        return self.network(x)

    def sample_action(self, s):
        self.time_step += 1
        self.epsilon = linear_schedule(self.epsilon_start, self.epsilon_target, self.exploration_fraction * self.total_steps, self.time_step)
        if isinstance(s, np.ndarray):
            s = torch.tensor(np.array(s), dtype=self.dtype, device=self.device)
        if np.random.rand() < self.epsilon:
            q_values = self.q(s)
            greedy_action = torch.argmax(q_values, dim=-1).item()
            random_action = np.random.randint(0, self.n_actions)
            if greedy_action == random_action:
                return random_action, False
            else:
                return random_action, True
        else:
            q_values = self.q(s)
            return torch.argmax(q_values, dim=-1).item(), False

    def update_params(self, s, a, r, s_prime, done, is_nongreedy):
        done_mask = 0 if done else 1
        s, a, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=self.dtype, device=self.device), torch.tensor([a], dtype=torch.int, device=self.device).squeeze(0), \
                                         torch.tensor(np.array(r), dtype=self.dtype, device=self.device), torch.tensor(np.array(s_prime), dtype=self.dtype, device=self.device), \
                                         torch.tensor(np.array(done_mask), dtype=self.dtype, device=self.device)
        
        q_values = self.q(s)
        next_q_values = self.q(s_prime)
        q_sa = q_values[a]
        max_q_s_prime_a_prime = torch.max(next_q_values, dim=-1).values
        td_target = r + self.gamma * max_q_s_prime_a_prime * done_mask
        delta = td_target - q_sa

        self.optimizer.zero_grad()
        q_sa.backward()
        optimizer_stats = self.optimizer.step(delta.item(), reset=(done or is_nongreedy))
        stats = {
            "td_error": delta.item(),
            "q_sa": q_sa.item(),
            "q_mean": q_values.detach().float().mean().item(),
            "q_max": q_values.detach().float().max().item(),
            "max_next_q": max_q_s_prime_a_prime.item(),
            "td_target": td_target.item(),
            "epsilon": self.epsilon,
            "is_nongreedy": bool(is_nongreedy),
            "has_nan": any(_tensor_has_nan(x) for x in (s, r, s_prime, done_mask, q_values, next_q_values, q_sa, max_q_s_prime_a_prime, td_target, delta)),
            "has_inf": any(_tensor_has_inf(x) for x in (s, r, s_prime, done_mask, q_values, next_q_values, q_sa, max_q_s_prime_a_prime, td_target, delta)),
        }
        stats.update(optimizer_stats)
        stats["has_nan"] = bool(stats["has_nan"] or optimizer_stats["has_nan"])
        stats["has_inf"] = bool(stats["has_inf"] or optimizer_stats["has_inf"])
        return stats

def main(env_name, seed, gamma, lamda, total_steps, epsilon_target, epsilon_start, exploration_fraction, eta_value, debug, render=False, track=False, wandb_project="intentional-updates", log_metrics=False, log_interval=1000, log_dir="logs", dtype_name="fp32", device_name="auto", dtype_tag=None, stop_on_nonfinite=False):
    torch.manual_seed(seed); np.random.seed(seed)
    dtype = resolve_dtype(dtype_name)
    device = resolve_device(device_name)
    dtype_tag = default_dtype_tag(dtype_name, dtype_tag)
    env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NormalizeObservation(env)
    env = ScaleReward(env, gamma=gamma)
    agent = IntentionalQ(n_channels=env.observation_space.shape[-1], n_actions=env.action_space.n, gamma=gamma, lamda=lamda, epsilon_target=epsilon_target, epsilon_start=epsilon_start, exploration_fraction=exploration_fraction, total_steps=total_steps, eta_value=eta_value, dtype=dtype, device=device)
    if track:
        import wandb
        wandb.init(
            project=wandb_project,
            config={
                "env_name": env_name,
                "seed": seed,
                "gamma": gamma,
                "lamda": lamda,
                "total_steps": total_steps,
                "eta_value": eta_value,
                "epsilon_target": epsilon_target,
                "epsilon_start": epsilon_start,
                "exploration_fraction": exploration_fraction,
                "dtype": dtype_name,
                "device": str(device),
            },
            name=f"{env_name}-seed{seed}",
        )
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id), "dtype: {}".format(dtype_name), "device: {}".format(device))
    logger = None
    if log_metrics:
        device_tag = str(device).replace(":", "-")
        run_name = f"intentional_q_minatar_{env.spec.id}_seed{seed}_{dtype_tag}_{device_tag}"
        logger = MetricsLogger(log_dir, run_name)
        if debug:
            print("metrics log: {}".format(logger.path))
    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    episode_num = 1
    for t in range(1, total_steps+1):
        a, is_nongreedy = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        step_metrics = agent.update_params(s, a, r, s_prime, terminated or truncated, is_nongreedy)
        if logger and (t % log_interval == 0 or has_nonfinite_value(step_metrics)):
            logger.log({
                "event": "step",
                "global_step": t,
                "env_name": env.spec.id,
                "seed": seed,
                "dtype": dtype_name,
                "device": str(device),
                "dtype_tag": dtype_tag,
                **step_metrics,
            })
        if stop_on_nonfinite and has_nonfinite_value(step_metrics):
            if debug:
                print("Stopping on non-finite metric at step {}".format(t))
            break
        s = s_prime
        if terminated or truncated:
            episode_return = info["episode"]["r"]
            episode_length = info["episode"].get("l", None)
            if debug:
                print("Episodic Return: {}, Time Step {}, Episode Number {}, Epsilon {}".format(episode_return, t, episode_num, agent.epsilon))
            returns.append(episode_return)
            term_time_steps.append(t)
            s, _ = env.reset()
            episode_num += 1
            if logger:
                logger.log({
                    "event": "episode",
                    "global_step": t,
                    "env_name": env.spec.id,
                    "seed": seed,
                    "dtype": dtype_name,
                    "device": str(device),
                    "dtype_tag": dtype_tag,
                    "episodic_return": episode_return,
                    "episode_length": episode_length,
                    "epsilon": agent.epsilon,
                })
            if track:
                wandb.log({
                        "charts/episodic_return": episode_return,
                        "charts/episode_length": episode_length,
                        "charts/epsilon": agent.epsilon,
                        "global_step": t,
                    },step=t)
    env.close()
    save_dir = "data_intentional_q_{}_gamma{}_lamda{}_eta{}".format(env.spec.id, gamma, lamda, eta_value)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)
    if track:
        wandb.finish()
    if logger:
        logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Intentional Q(λ)')
    parser.add_argument('--env_name', type=str, default='MinAtar/Breakout-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--epsilon_target', type=float, default=0.01)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--exploration_fraction', type=float, default=0.2)
    parser.add_argument('--eta_value', type=float, default=0.25)
    parser.add_argument('--total_steps', type=int, default=5_000_000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--track", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="intentional-updates")
    parser.add_argument("--log_metrics", action="store_true", help="Write step and episode diagnostics to JSONL")
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="fp32")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--dtype_tag", type=str, default=None)
    parser.add_argument("--stop_on_nonfinite", action="store_true")
    args = parser.parse_args()
    main(args.env_name, args.seed, args.gamma, args.lamda, args.total_steps, args.epsilon_target, args.epsilon_start, args.exploration_fraction, args.eta_value, args.debug, args.render, args.track, args.wandb_project, args.log_metrics, args.log_interval, args.log_dir, args.dtype, args.device, args.dtype_tag, args.stop_on_nonfinite)
