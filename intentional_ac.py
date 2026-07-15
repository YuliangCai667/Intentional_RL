import os, pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import shimmy
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Normal
from optimizer import IntentionalOptimizerPolicy, IntentionalOptimizerValue
from normalization_wrappers import NormalizeObservation, ScaleReward
from sparse_init import sparse_init
from metrics_logger import MetricsLogger, has_nonfinite_value
from precision_utils import DTYPE_CHOICES, DEVICE_CHOICES, default_dtype_tag, resolve_device, resolve_dtype

def _prefix_stats(prefix, stats):
    return {f"{prefix}_{key}": value for key, value in stats.items()}

def _tensor_has_nan(tensor):
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return False
    return torch.isnan(tensor).any().item()

def _tensor_has_inf(tensor):
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return False
    return torch.isinf(tensor).any().item()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128):
        super(Actor, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_mu = nn.Linear(hidden_size, n_actions)
        self.linear_std = nn.Linear(hidden_size, n_actions)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        mu = self.linear_mu(x)
        pre_std = self.linear_std(x)
        std = F.softplus(pre_std)
        return mu, std

class Critic(nn.Module):
    def __init__(self, n_obs=11, hidden_size=128):
        super(Critic, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.hidden_layer  = nn.Linear(hidden_size, hidden_size)
        self.linear_layer  = nn.Linear(hidden_size, 1)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)      
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        return self.linear_layer(x)

class IntentionalAC(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, gamma=0.99, lamda=0.8, eta_policy=0.05, eta_value=0.5, dtype=torch.float32, device=torch.device("cpu")):
        super(IntentionalAC, self).__init__()
        self.dtype = dtype
        self.device = torch.device(device)
        self.gamma = gamma
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size)
        self.value_net = Critic(n_obs=n_obs, hidden_size=hidden_size)
        self.to(device=self.device, dtype=self.dtype)
        self.optimizer_policy = IntentionalOptimizerPolicy(self.policy_net.parameters(),  gamma=gamma, lamda=lamda, eta=eta_policy)
        self.optimizer_value = IntentionalOptimizerValue(self.value_net.parameters(), gamma=gamma, lamda=lamda, eta=eta_value)
        self.eta_value = eta_value
        self.eta_policy = eta_policy

    def pi(self, x):
        return self.policy_net(x)

    def v(self, x):
        return self.value_net(x)

    def sample_action(self, s):
        x = torch.from_numpy(s).to(device=self.device, dtype=self.dtype)
        mu, std = self.pi(x)
        dist = Normal(mu, std)
        return dist.sample().detach().cpu().numpy()

    def update_params(self, s, a, r, s_prime, done, terminated, entropy_coeff):
        termination_mask = 0 if terminated else 1
        s, a, r, s_prime, termination_mask = torch.tensor(np.array(s), dtype=self.dtype, device=self.device), torch.tensor(np.array(a), dtype=self.dtype, device=self.device), \
                                         torch.tensor(np.array(r), dtype=self.dtype, device=self.device), torch.tensor(np.array(s_prime), dtype=self.dtype, device=self.device), \
                                         torch.tensor(np.array(termination_mask), dtype=self.dtype, device=self.device)

        v_s, v_prime = self.v(s), self.v(s_prime)
        td_target = r + self.gamma * v_prime * termination_mask
        delta = td_target - v_s

        mu, std = self.pi(s)
        dist = Normal(mu, std)

        log_prob_pi = (dist.log_prob(a)).sum()
        entropy_pi = entropy_coeff * dist.entropy().sum() * torch.sign(delta).item()
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        v_s.backward()
        (log_prob_pi + entropy_pi).backward()
        policy_stats = self.optimizer_policy.step(delta.item(), reset=done)
        value_stats = self.optimizer_value.step(delta.item(), reset=done)
        stats = {
            "td_error": delta.item(),
            "value": v_s.item(),
            "next_value": v_prime.item(),
            "td_target": td_target.item(),
            "policy_mu_mean": mu.detach().float().mean().item(),
            "policy_std_mean": std.detach().float().mean().item(),
            "policy_log_prob": log_prob_pi.item(),
            "policy_entropy": dist.entropy().sum().item(),
            "has_nan": any(_tensor_has_nan(x) for x in (s, a, r, s_prime, v_s, v_prime, td_target, delta, mu, std, log_prob_pi, entropy_pi)),
            "has_inf": any(_tensor_has_inf(x) for x in (s, a, r, s_prime, v_s, v_prime, td_target, delta, mu, std, log_prob_pi, entropy_pi)),
        }
        stats.update(_prefix_stats("policy_optimizer", policy_stats))
        stats.update(_prefix_stats("value_optimizer", value_stats))
        stats["has_nan"] = bool(stats["has_nan"] or policy_stats["has_nan"] or value_stats["has_nan"])
        stats["has_inf"] = bool(stats["has_inf"] or policy_stats["has_inf"] or value_stats["has_inf"])
        return stats

def main(env_name, seed, gamma, lamda, total_steps, entropy_coeff, eta_policy, eta_value, debug, render=False, track=False, wandb_project="intentional-updates", log_metrics=False, log_interval=1000, log_dir="logs", dtype_name="fp32", device_name="auto", dtype_tag=None, stop_on_nonfinite=False):
    torch.manual_seed(seed); np.random.seed(seed)
    dtype = resolve_dtype(dtype_name)
    device = resolve_device(device_name)
    dtype_tag = default_dtype_tag(dtype_name, dtype_tag)
    env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = ScaleReward(env, gamma=gamma)
    env = NormalizeObservation(env)
    agent = IntentionalAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], gamma=gamma, lamda=lamda, eta_policy=eta_policy, eta_value=eta_value, dtype=dtype, device=device)
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
                "eta_policy": eta_policy,
                "eta_value": eta_value,
                "entropy_coeff": entropy_coeff,
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
        run_name = f"intentional_ac_{env.spec.id}_seed{seed}_{dtype_tag}_{device_tag}"
        logger = MetricsLogger(log_dir, run_name)
        if debug:
            print("metrics log: {}".format(logger.path))
    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    for t in range(1, total_steps+1):
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        step_metrics = agent.update_params(s, a, r, s_prime, done,  terminated, entropy_coeff)
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
        if done:
            episode_return = info["episode"]["r"]
            episode_length = info["episode"].get("l", None)
            if debug:
                print("Episodic Return: {}, Time Step {}".format(episode_return, t))
            returns.append(episode_return)
            term_time_steps.append(t)
            s, _ = env.reset()
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
                })
            if track:
                wandb.log({
                        "charts/episodic_return": episode_return,
                        "charts/episode_length": episode_length,
                        "global_step": t,
                    },step=t)
    env.close()
    save_dir = "data_intentional_ac_{}_gamma{}_lamda{}_entropy_coeff{}_eta_policy{}_eta_value{}".format(env.spec.id, gamma, lamda, entropy_coeff, eta_policy, eta_value)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)
    if track:
        wandb.finish()
    if logger:
        logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Intentional AC(λ)')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--total_steps', type=int, default=5_000_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--eta_policy', type=float, default=0.05)
    parser.add_argument('--eta_value', type=float, default=0.5)
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
    main(args.env_name, args.seed, args.gamma, args.lamda, args.total_steps, args.entropy_coeff, args.eta_policy, args.eta_value, args.debug, args.render, args.track, args.wandb_project, args.log_metrics, args.log_interval, args.log_dir, args.dtype, args.device, args.dtype_tag, args.stop_on_nonfinite)
