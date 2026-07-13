import torch, math

def _tensor_has_nan(tensor):
    return torch.isnan(tensor).any().item()


def _tensor_has_inf(tensor):
    return torch.isinf(tensor).any().item()


class IntentionalOptimizer(torch.optim.Optimizer):
    """
    Unified intentional-update optimizer for both policy and value networks.
    
    Args:
        params: Model parameters
        gamma: Discount factor
        lamda: Eligibility trace decay
        eta: Step size coefficient (default: 0.5 for value, use 0.05 for policy)
        beta2: RMSProp decay
        normalize_delta: If True, normalize delta after clipping (for policy).
                        If False, only clip delta (for value).
        
        Ablation flags (all default to True for full algorithm):
        use_adaptive_clip: If True, use EMA-based adaptive clipping. If False, clip to [-1, 1].
        use_rmsprop: If True, use RMSProp normalization. If False, no gradient normalization.
        use_sigma: If True, use sigma normalization in step size. If False, step_size = eta / ||z||.
    """
    def __init__(self, params, gamma=0.99, lamda=0.0, eta=0.5, beta2=0.999, 
                 normalize_delta=False, clip_mult=20.0, beta_clip=0.9998, beta_norm=0.9998,
                 use_adaptive_clip=True, use_rmsprop=True, use_sigma=True):
        defaults = dict(gamma=gamma, lamda=lamda, beta2=beta2)
        super().__init__(params, defaults)

        self.gamma = gamma
        self.lamda = lamda
        self.eta = eta
        self.beta2 = beta2
        self.normalize_delta = normalize_delta
        
        # Ablation flags
        self.use_adaptive_clip = use_adaptive_clip
        self.use_rmsprop = use_rmsprop
        self.use_sigma = use_sigma

        self.sigma = 0.0
        self.t_step = 0

        # state for delta clipping (only used if use_adaptive_clip=True)
        self.clip_mult = clip_mult   # C in the algorithm
        self.beta_clip = beta_clip    # β_clip in the algorithm
        self.clip_t = 0
        self.clip_ema_sq = 0.0       # δ̂_t in the algorithm
        
        # normalization state (only used if normalize_delta=True)
        self.beta_norm = beta_norm    # β_norm in the algorithm
        self.norm_t = 0
        self.delta_abs_ema = 0.0     # Ā_t in the algorithm
        self.safe_delta = 1.0

    def _process_delta(self, delta: float) -> float:
        """Clip delta and optionally normalize."""
        if self.use_adaptive_clip:
            # Update EMA with raw squared value first
            self.clip_t += 1
            self.clip_ema_sq = self.beta_clip * self.clip_ema_sq + (1.0 - self.beta_clip) * delta * delta

            # Clip using updated EMA
            cap = self.clip_mult * math.sqrt(self.clip_ema_sq / (1.0 - self.beta_clip ** self.clip_t))
            clipped = math.copysign(min(abs(delta), cap), delta)
        else:
            # Base algorithm: simple [-1, 1] clipping
            clipped = max(-1.0, min(1.0, delta))

        if not self.normalize_delta:
            return clipped

        # Normalize delta (for policy)
        self.norm_t += 1
        self.delta_abs_ema = self.beta_norm * self.delta_abs_ema + (1.0 - self.beta_norm) * abs(clipped)
        delta_abs_ema_corrected = self.delta_abs_ema / (1.0 - self.beta_norm ** self.norm_t)

        return clipped / max(delta_abs_ema_corrected, 1e-12)

    def step(self, delta, reset=False):
        self.t_step += 1
        stats = {
            "delta_raw": float(delta),
            "has_nan": not math.isfinite(float(delta)) and math.isnan(float(delta)),
            "has_inf": not math.isfinite(float(delta)) and math.isinf(float(delta)),
        }

        # 1) Entrywise RMSProp stats and norm_grad
        norm_grad = 0.0
        grad_l2_sq = 0.0
        param_l2_sq_before = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                param_l2_sq_before += p.data.detach().float().square().sum().item()
                if _tensor_has_nan(p.data):
                    stats["has_nan"] = True
                if _tensor_has_inf(p.data):
                    stats["has_inf"] = True
                if p.grad is None:
                    continue
                state = self.state[p]
                grad_l2_sq += p.grad.detach().float().square().sum().item()
                if _tensor_has_nan(p.grad):
                    stats["has_nan"] = True
                if _tensor_has_inf(p.grad):
                    stats["has_inf"] = True
                if self.use_rmsprop:
                    if len(state) == 0:
                        state["entrywise_squared_grad"] = torch.zeros_like(p.data)
                    v = state["entrywise_squared_grad"]
                    v.mul_(self.beta2).addcmul_(p.grad, p.grad, value=1.0 - self.beta2)
                    v_hat = (v / (1.0 - self.beta2 ** self.t_step)).sqrt() + 1e-8
                    state["rmsprop_v_hat"] = v_hat
                    norm_grad += (p.grad.square() / v_hat).sum().item()
                else:
                    # No RMSProp: v_hat = 1
                    state["rmsprop_v_hat"] = torch.ones_like(p.data)
                    norm_grad += p.grad.square().sum().item()

        # 2) Eligibility traces and z_sum
        z_sum = 0.0
        eligibility_trace_l2_sq = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                e.mul_(self.gamma * self.lamda).add_(p.grad, alpha=1.0)
                eligibility_trace_l2_sq += e.detach().float().square().sum().item()
                if _tensor_has_nan(e) or _tensor_has_nan(state["rmsprop_v_hat"]):
                    stats["has_nan"] = True
                if _tensor_has_inf(e) or _tensor_has_inf(state["rmsprop_v_hat"]):
                    stats["has_inf"] = True
                if self.use_rmsprop:
                    z_sum += (e.square() / state["rmsprop_v_hat"]).sum().item()
                else:
                    # Base algorithm: L2 norm of eligibility trace
                    z_sum += e.square().sum().item()

        # 3) Global normalizer (sigma) and step size
        if self.use_sigma:
            self.sigma += (1 - self.gamma * self.lamda) * (norm_grad - self.sigma)
            sigma_bc = self.sigma / (1 - (self.gamma * self.lamda) ** self.t_step)
            u = math.sqrt(sigma_bc * z_sum)
        else:
            # Base algorithm: step_size = eta / ||z||
            u = z_sum
        
        step_size = self.eta / max(u, 1e-8)

        # 4) Delta clipping (and optional normalization)
        self.safe_delta = self._process_delta(delta)

        # 5) Apply update
        update_l2_sq = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                e = state["eligibility_trace"]
                
                update_direction = e / state["rmsprop_v_hat"]
                update_alpha = self.safe_delta * step_size
                update_l2_sq += update_direction.detach().float().square().sum().item() * (update_alpha ** 2)
                p.data.add_(update_direction, alpha=update_alpha)
                if _tensor_has_nan(p.data):
                    stats["has_nan"] = True
                if _tensor_has_inf(p.data):
                    stats["has_inf"] = True
                
                if reset:
                    e.zero_()

        stats.update({
            "delta_processed": float(self.safe_delta),
            "grad_norm": math.sqrt(max(grad_l2_sq, 0.0)),
            "rmsprop_grad_norm": math.sqrt(max(norm_grad, 0.0)),
            "eligibility_trace_norm": math.sqrt(max(eligibility_trace_l2_sq, 0.0)),
            "rmsprop_trace_norm": math.sqrt(max(z_sum, 0.0)),
            "step_size": float(step_size),
            "update_norm": math.sqrt(max(update_l2_sq, 0.0)),
            "param_norm": math.sqrt(max(param_l2_sq_before, 0.0)),
            "sigma": float(self.sigma),
        })
        stats["has_nan"] = bool(stats["has_nan"] or any(math.isnan(v) for v in stats.values() if isinstance(v, float)))
        stats["has_inf"] = bool(stats["has_inf"] or any(math.isinf(v) for v in stats.values() if isinstance(v, float)))
        return stats


def IntentionalOptimizerPolicy(params, gamma=0.99, lamda=0.0, eta=0.05, beta2=0.999,
              clip_mult=20.0, beta_clip=0.9998, beta_norm=0.9998, normalize_delta=True,
              use_adaptive_clip=True, use_rmsprop=True, use_sigma=True):
    """Intentional optimizer configured for policy networks (with delta normalization)."""
    return IntentionalOptimizer(params, gamma=gamma, lamda=lamda, eta=eta, 
               beta2=beta2, normalize_delta=normalize_delta,
               clip_mult=clip_mult, beta_clip=beta_clip, beta_norm=beta_norm,
               use_adaptive_clip=use_adaptive_clip, use_rmsprop=use_rmsprop,
               use_sigma=use_sigma)


def IntentionalOptimizerValue(params, gamma=0.99, lamda=0.0, eta=0.5, beta2=0.999,
             clip_mult=20.0, beta_clip=0.9998, normalize_delta=False,
             use_adaptive_clip=True, use_rmsprop=True, use_sigma=True):
    """Intentional optimizer configured for value networks (without delta normalization)."""
    return IntentionalOptimizer(params, gamma=gamma, lamda=lamda, eta=eta, 
               beta2=beta2, normalize_delta=normalize_delta,
               clip_mult=clip_mult, beta_clip=beta_clip,
               use_adaptive_clip=use_adaptive_clip, use_rmsprop=use_rmsprop,
               use_sigma=use_sigma)

