import torch
import os
import numpy as np

EPS = 1e-6
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Action_Distribution:
    def __init__(self):
        super().__init__()

    def get_act_dist(self, logits):
        act_dist_cls = getattr(self, "action_distribution_cls")
        has_act_lim = hasattr(self, "act_high_lim")

        act_dist = act_dist_cls(logits)
        if has_act_lim:
            act_dist.act_high_lim = getattr(self, "act_high_lim")
            act_dist.act_low_lim = getattr(self, "act_low_lim")

        return act_dist


class TanhGaussDistribution:
    def __init__(self, logits, act_low_lim=None, act_high_lim=None):
        self.logits = logits
        self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(self.mean, self.std),
            reinterpreted_batch_ndims=1,
        )
        device = logits.device
        dtype = logits.dtype
        if act_low_lim is not None and act_high_lim is not None:
            self.act_low_lim = torch.as_tensor(act_low_lim, dtype=dtype, device=device)
            self.act_high_lim = torch.as_tensor(act_high_lim, dtype=dtype, device=device)
        else:
            action_dim = self.mean.shape[-1]
            self.act_high_lim = torch.ones(action_dim, dtype=dtype, device=device)
            self.act_low_lim = -torch.ones(action_dim, dtype=dtype, device=device)

    def sample(self, train=True):
        action, _, log_prob = self.sample_with_raw(train=train)
        return action.to('cpu'), log_prob.to('cpu')

    def _scale(self):
        return (self.act_high_lim - self.act_low_lim) / 2

    def _bias(self):
        return (self.act_high_lim + self.act_low_lim) / 2

    def sample_with_raw(self, train=True):
        raw_action = self.gauss_distribution.sample() if train else self.mean
        action_limited = self._scale() * torch.tanh(raw_action) + self._bias()
        return action_limited, raw_action, self.log_prob_from_raw(raw_action)

    def rsample(self):
        action, _, log_prob = self.rsample_with_raw()
        return action, log_prob

    def rsample_with_raw(self):
        raw_action = self.gauss_distribution.rsample()
        action_limited = self._scale() * torch.tanh(raw_action) + self._bias()
        return action_limited, raw_action, self.log_prob_from_raw(raw_action)

    def log_prob_from_raw(self, raw_action) -> torch.Tensor:
        # The transform is raw -> tanh(raw) -> scale * value + bias.
        tanh_log_det = torch.log(
            torch.clamp(1.0 - torch.tanh(raw_action).pow(2), min=EPS)
        ).sum(dim=-1)
        scale_log_det = torch.log(torch.clamp(self._scale().abs(), min=EPS)).sum()
        return self.gauss_distribution.log_prob(raw_action) - tanh_log_det - scale_log_det

    def log_prob(self, action_limited) -> torch.Tensor:
        normalized = (action_limited - self._bias()) / self._scale()
        normalized = torch.clamp(normalized, -1.0 + EPS, 1.0 - EPS)
        return self.log_prob_from_raw(torch.atanh(normalized))

    def entropy(self):
        return self.gauss_distribution.entropy()

    def mode(self):
        return (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(self.mean) + (
                self.act_high_lim + self.act_low_lim
        ) / 2

    def kl_divergence(self, other: "GaussDistribution") -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(
            self.gauss_distribution, other.gauss_distribution
        )


class GaussDistribution:
    def __init__(self, logits):
        self.logits = logits
        self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(self.mean, self.std),
            reinterpreted_batch_ndims=1,
        )
        self.act_high_lim = torch.tensor([4.0])
        self.act_low_lim = torch.tensor([0.0])

    def sample(self):
        action = self.gauss_distribution.sample()
        action = action.to('cpu')
        # action_limited = (self.act_high_lim - self.act_low_lim)/2*action + (self.act_high_lim + self.act_low_lim)/2
        action = action.to('cuda:0')
        log_prob = self.gauss_distribution.log_prob(action)
        action = action.to('cpu')

        log_prob = log_prob.to('cpu')
        return action, log_prob

    def rsample(self):
        action = self.gauss_distribution.rsample()
        log_prob = self.gauss_distribution.log_prob(action)
        return action, log_prob

    def log_prob(self, action) -> torch.Tensor:
        log_prob = self.gauss_distribution.log_prob(action)
        return log_prob

    def entropy(self):
        return self.gauss_distribution.entropy()

    def mode(self):
        return torch.clamp(self.mean, self.act_low_lim, self.act_high_lim)

    def kl_divergence(self, other: "GaussDistribution") -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(
            self.gauss_distribution, other.gauss_distribution
        )
