"""PyTorch DSAC-FDPI-Dual agent for the existing Epre encoder stack."""

from collections import deque
from copy import deepcopy
import math
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import Config
from epre_dsac.Epre_dsac import Epre_dsac_agent
from epre_dsac.Epre_dsac_model import PolicyNet, SafetyCritic
from epre_dsac.parameters import agent_par
from utilsa.act_distribution_cls import TanhGaussDistribution


class EpreDSACFDPIAgent(Epre_dsac_agent):
    """Adds FDPI safety critics and a constrained risk-seeking dual actor."""

    def __init__(self, ModelPath, ModelPath_temp, hrl=False, h_rl=1e-5):
        algorithm_mode = str(agent_par.get("algorithm_mode", "dsac_fdpi")).lower()
        state_encoder = str(agent_par.get("fdpi_state_encoder", "frenet")).lower()
        if algorithm_mode == "dsac_fdpi_stt":
            state_encoder = "stt"
        elif algorithm_mode == "dsac_fdpi":
            state_encoder = "frenet"
        super().__init__(
            ModelPath,
            ModelPath_temp,
            hrl=hrl,
            h_rl=h_rl,
            state_encoder=state_encoder,
        )
        self.fdpi_enabled = bool(agent_par.get("fdpi_enabled", False))
        self.global_step = int(agent_par.get("train_data", {}).get("global_step", 0))
        self.episode = int(agent_par.get("train_data", {}).get("episode", 0))
        self.feasible_ratio_history = deque(
            maxlen=int(agent_par.get("fdpi_feasible_window", 1000))
        )
        self.dual_active = False
        self._build_fdpi_networks()

    def _build_fdpi_networks(self):
        self.main_policy = self.policy
        self.main_policy_target = self.policy_target
        self.dual_policy = PolicyNet(self.q_state_size, self.action_number).to(self.device)
        self.dual_policy.load_state_dict(self.main_policy.state_dict())

        def critic_pair():
            first = SafetyCritic(self.q_state_size, self.action_number).to(self.device)
            second = SafetyCritic(self.q_state_size, self.action_number).to(self.device)
            return first, second, deepcopy(first), deepcopy(second)

        self.g1, self.g2, self.g1_target, self.g2_target = critic_pair()
        self.gr1, self.gr2, self.gr1_target, self.gr2_target = critic_pair()
        (
            self.dual_g1,
            self.dual_g2,
            self.dual_g1_target,
            self.dual_g2_target,
        ) = critic_pair()

        lr = Config.lr
        self.g1_optimizer = optim.Adam(self.g1.parameters(), lr=lr, weight_decay=1e-4)
        self.g2_optimizer = optim.Adam(self.g2.parameters(), lr=lr, weight_decay=1e-4)
        self.gr1_optimizer = optim.Adam(self.gr1.parameters(), lr=lr, weight_decay=1e-4)
        self.gr2_optimizer = optim.Adam(self.gr2.parameters(), lr=lr, weight_decay=1e-4)
        self.dual_g1_optimizer = optim.Adam(self.dual_g1.parameters(), lr=lr, weight_decay=1e-4)
        self.dual_g2_optimizer = optim.Adam(self.dual_g2.parameters(), lr=lr, weight_decay=1e-4)
        self.dual_policy_optimizer = optim.Adam(
            self.dual_policy.parameters(), lr=lr, weight_decay=1e-4
        )

        multiplier_lr = float(agent_par.get("fdpi_lambda_lr", 3e-4))
        cg_init = max(float(agent_par.get("fdpi_cg_init", 0.01)), 1e-8)
        self.log_cg = torch.nn.Parameter(
            torch.tensor(math.log(cg_init), dtype=torch.float32, device=self.device)
        )
        self.log_lambda1 = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
        self.log_lambda2 = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
        self.log_lambda3 = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
        self.log_lambda4 = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
        self.cg_optimizer = optim.Adam([self.log_cg], lr=multiplier_lr)
        self.lambda1_optimizer = optim.Adam([self.log_lambda1], lr=multiplier_lr)
        self.lambda2_optimizer = optim.Adam([self.log_lambda2], lr=multiplier_lr)
        self.lambda3_optimizer = optim.Adam([self.log_lambda3], lr=multiplier_lr)
        self.lambda4_optimizer = optim.Adam([self.log_lambda4], lr=multiplier_lr)

    @property
    def mean_feasible_ratio(self):
        return float(np.mean(self.feasible_ratio_history)) if self.feasible_ratio_history else 0.0

    def refresh_dual_active(self):
        self.dual_active = bool(
            agent_par.get("fdpi_dual_enabled", True)
            and self.global_step >= int(agent_par.get("fdpi_warmup_steps", 10000))
            and self.feasible_ratio_history
            and self.mean_feasible_ratio > float(agent_par.get("fdpi_dual_threshold", 0.90))
        )
        return self.dual_active

    def _distribution(self, policy, encoded_state):
        return TanhGaussDistribution(
            policy(encoded_state),
            act_low_lim=self.action_low_limit,
            act_high_lim=self.action_high_limit,
        )

    def take_fdpi_action(self, encoded_state, behavior_policy, train=True):
        """Sample once and score the exact same pre-tanh action under both actors."""
        if not train:
            behavior_policy = "main"
        if behavior_policy not in ("main", "dual"):
            raise ValueError("behavior_policy must be 'main' or 'dual'")
        if behavior_policy == "dual" and not self.fdpi_enabled:
            behavior_policy = "main"

        self.main_policy.eval()
        self.dual_policy.eval()
        with torch.no_grad():
            main_dist = self._distribution(self.main_policy, encoded_state)
            dual_dist = self._distribution(self.dual_policy, encoded_state)
            behavior_dist = main_dist if behavior_policy == "main" else dual_dist
            action, raw_action, _ = behavior_dist.sample_with_raw(train=train)
            logp_main = main_dist.log_prob_from_raw(raw_action)
            logp_dual = dual_dist.log_prob_from_raw(raw_action)

        action_np = action.detach().cpu().numpy()
        main_np = logp_main.detach().cpu().numpy()
        dual_np = logp_dual.detach().cpu().numpy()
        if action_np.shape[0] == 1:
            action_np = action_np[0]
            main_np = main_np[0]
            dual_np = dual_np[0]
        return {
            "action": action_np,
            "logp_main": main_np,
            "logp_dual": dual_np,
            "behavior_policy": behavior_policy,
        }

    def _tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _encode_batch(self, batch):
        if "encoded_state" in batch:
            state = self._tensor(batch["encoded_state"])
            next_state = self._tensor(batch["encoded_next_state"])
            return state, next_state, False

        raw_state = self._tensor(batch["state"])
        raw_next_state = self._tensor(batch["next_state"])
        if self.state_encoder == "frenet":
            return (
                self.encode_policy_state(raw_state),
                self.encode_policy_state(raw_next_state),
                False,
            )
        env_input = self._tensor(batch["env_input"])
        next_env_input = self._tensor(batch["next_env_input"])
        env_map = self._tensor(batch["env_map"])
        next_env_map = self._tensor(batch["next_env_map"])
        state = self.encode_policy_state(raw_state, env_input, env_map)
        with torch.no_grad():
            next_state = self.encode_policy_state(
                raw_next_state, next_env_input, next_env_map, target=True
            )
        return state, next_state, True

    @staticmethod
    def _set_trainable(networks, value):
        for network in networks:
            for parameter in network.parameters():
                parameter.requires_grad_(value)

    def _finite(self, loss, loss_name, action, logp, weight):
        if torch.isfinite(loss).all():
            return True
        bad_mask = ~torch.isfinite(weight.reshape(-1))
        if logp.numel() == bad_mask.numel():
            bad_mask = bad_mask | ~torch.isfinite(logp.reshape(-1))
        if action.ndim > 1 and action.shape[0] == bad_mask.numel():
            bad_mask = bad_mask | ~torch.isfinite(action).all(dim=-1)
        bad = torch.where(bad_mask)[0].detach().cpu().tolist()
        if not bad:
            bad = list(range(action.shape[0] if action.ndim > 1 else 1))
        print(
            "[fdpi][NONFINITE] network={} loss={} batch_index={} action={} "
            "log_probability={} importance_weight={}".format(
                loss_name.split("_")[0],
                loss_name,
                bad,
                action.detach().cpu().numpy(),
                logp.detach().cpu().numpy(),
                weight.detach().cpu().numpy(),
            ),
            flush=True,
        )
        return False

    @staticmethod
    def _zero_and_step(loss, optimizers, parameters):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss).all():
            print("[fdpi][NONFINITE] skip optimizer step for non-finite loss", flush=True)
            return False
        loss.backward()
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            print("[fdpi][NONFINITE] skip optimizer step for non-finite gradient", flush=True)
            return False
        torch.nn.utils.clip_grad_norm_(parameters, 40.0)
        for optimizer in optimizers:
            optimizer.step()
        return True

    def _q_values(self, network, state, action):
        output = network(state, action)
        return output[..., 0], output[..., 1]

    def update(self, batch, iteration=0, logger=None):
        """Perform one update in the FDPI order described by the algorithm."""
        state, next_state, trains_encoder = self._encode_batch(batch)
        action = self._tensor(batch["action"])
        reward = self._tensor(batch["reward"]).reshape(-1)
        cost = self._tensor(batch["cost"]).reshape(-1).clamp(0.0, 1.0)
        terminated = self._tensor(batch["terminated"]).reshape(-1)
        bootstrap_mask = 1.0 - terminated
        main_weight = torch.exp(self._tensor(batch["log_is_to_main"]).reshape(-1)).detach()
        dual_weight = torch.exp(self._tensor(batch["log_is_to_dual"]).reshape(-1)).detach()
        gamma = float(Config.discount_factor)
        cost_gamma = float(agent_par.get("fdpi_cost_gamma", 0.97))
        alpha = self.log_alpha.exp().detach()

        # 1-2. Distributional DSAC reward critics (and the shared encoder).
        with torch.no_grad():
            next_dist = self._distribution(self.main_policy, next_state)
            next_action, next_logp = next_dist.rsample()
            target_q1_mean, target_q1_std = self._q_values(
                self.q1_target, next_state, next_action
            )
            target_q2_mean, target_q2_std = self._q_values(
                self.q2_target, next_state, next_action
            )
            target_q_mean = torch.minimum(target_q1_mean, target_q2_mean)
            z1 = torch.randn_like(target_q1_mean).clamp(-3.0, 3.0)
            z2 = torch.randn_like(target_q2_mean).clamp(-3.0, 3.0)
            target_q_sample = torch.where(
                target_q1_mean <= target_q2_mean,
                target_q1_mean + z1 * target_q1_std,
                target_q2_mean + z2 * target_q2_std,
            )
            q_backup = reward + bootstrap_mask * gamma * (
                target_q_mean - alpha * next_logp
            )
            q_backup_sample = reward + bootstrap_mask * gamma * (
                target_q_sample - alpha * next_logp
            )

        q1_mean, q1_std = self._q_values(self.q1, state, action)
        q2_mean, q2_std = self._q_values(self.q2, state, action)
        if self.mean_std1 == -1.0:
            self.mean_std1 = q1_std.detach().mean()
            self.mean_std2 = q2_std.detach().mean()
        else:
            self.mean_std1 = (1.0 - self.tau_b) * self.mean_std1 + self.tau_b * q1_std.detach().mean()
            self.mean_std2 = (1.0 - self.tau_b) * self.mean_std2 + self.tau_b * q2_std.detach().mean()

        def distributional_loss(mean, std, moving_std):
            bias = 0.1
            bound = mean.detach() + (q_backup_sample - mean.detach()).clamp(
                -3.0 * moving_std, 3.0 * moving_std
            )
            mean_coef = (q_backup - mean.detach()) / (std.detach().pow(2) + bias)
            std_coef = (
                (mean.detach() - bound).pow(2) - std.detach().pow(2)
            ) / (std.detach().pow(3) + bias)
            return (moving_std.pow(2) + bias) * (-mean_coef * mean - std_coef * std)

        q1_per_sample = distributional_loss(q1_mean, q1_std, self.mean_std1)
        q2_per_sample = distributional_loss(q2_mean, q2_std, self.mean_std2)
        q1_loss = (main_weight * q1_per_sample).mean()
        q2_loss = (main_weight * q2_per_sample).mean()
        q_loss = q1_loss + q2_loss
        q_optimizers = [self.q1_optimizer, self.q2_optimizer]
        q_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
        if trains_encoder:
            q_optimizers.append(self.h_optimizer)
            q_parameters += list(self.h.parameters())
        if self._finite(q_loss, "q_loss", action, next_logp, main_weight):
            self._zero_and_step(q_loss, q_optimizers, q_parameters)
        state = state.detach()
        next_state = next_state.detach()

        # 3. Main feasibility critics.
        with torch.no_grad():
            target_g = torch.maximum(
                self.g1_target(next_state, next_action),
                self.g2_target(next_state, next_action),
            ).clamp(0.0, 1.0)
            g_backup = cost + bootstrap_mask * (1.0 - cost) * cost_gamma * target_g
        g1_value = self.g1(state, action)
        g2_value = self.g2(state, action)
        g1_loss = (main_weight * (g1_value - g_backup).pow(2)).mean()
        g2_loss = (main_weight * (g2_value - g_backup).pow(2)).mean()
        g_loss = g1_loss + g2_loss
        if self._finite(g_loss, "g_loss", action, next_logp, main_weight):
            self._zero_and_step(
                g_loss,
                [self.g1_optimizer, self.g2_optimizer],
                list(self.g1.parameters()) + list(self.g2.parameters()),
            )

        # 4. Recovery critics.
        with torch.no_grad():
            target_gr = torch.minimum(
                self.gr1_target(next_state, next_action),
                self.gr2_target(next_state, next_action),
            ).clamp(0.0, 1.0)
            gr_backup = (1.0 - cost) + bootstrap_mask * cost * cost_gamma * target_gr
        gr1_value = self.gr1(state, action)
        gr2_value = self.gr2(state, action)
        gr1_loss = (main_weight * (gr1_value - gr_backup).pow(2)).mean()
        gr2_loss = (main_weight * (gr2_value - gr_backup).pow(2)).mean()
        gr_loss = gr1_loss + gr2_loss
        if self._finite(gr_loss, "gr_loss", action, next_logp, main_weight):
            self._zero_and_step(
                gr_loss,
                [self.gr1_optimizer, self.gr2_optimizer],
                list(self.gr1.parameters()) + list(self.gr2.parameters()),
            )

        # 5. Freeze all main critics while updating the main actor.
        main_critics = [self.q1, self.q2, self.g1, self.g2, self.gr1, self.gr2]
        self._set_trainable(main_critics, False)
        main_dist = self._distribution(self.main_policy, state)
        new_action, new_logp = main_dist.rsample()
        q = torch.minimum(
            self._q_values(self.q1, state, new_action)[0],
            self._q_values(self.q2, state, new_action)[0],
        )
        g = torch.maximum(self.g1(state, new_action), self.g2(state, new_action))
        gr = torch.minimum(self.gr1(state, new_action), self.gr2(state, new_action))
        if agent_par.get("fdpi_full_policy_loss", True):
            pf = float(agent_par.get("fdpi_pf", 0.10))
            cg = self.log_cg.exp().detach()
            lambda1 = self.log_lambda1.exp().detach()
            lambda2 = self.log_lambda2.exp().detach()
            vio = cost > 0.0
            fea = (g < pf) & (~vio)
            cri = fea & (g >= pf - cg)
            per_policy_loss = (
                ((~cri) & fea).float() * (-q)
                + cri.float() * (-q + lambda1 * g) / (lambda1 + 1.0)
                + ((~fea) & (~vio)).float() * (-q + lambda2 * g) / (lambda2 + 1.0)
                + vio.float() * (-gr)
                + alpha * new_logp
            )
            main_policy_loss = (main_weight * per_policy_loss).mean()
        else:
            pf = float(agent_par.get("fdpi_pf", 0.10))
            main_policy_loss = (alpha * new_logp - q).mean()
        if self._finite(main_policy_loss, "main_policy_loss", new_action, new_logp, main_weight):
            self._zero_and_step(
                main_policy_loss,
                [self.policy_optimizer],
                list(self.main_policy.parameters()),
            )
        self._set_trainable(main_critics, True)

        # 6. Entropy temperature.
        alpha_loss = -self.log_alpha * (new_logp.detach() + self.target_entropy).mean()
        if self._finite(alpha_loss, "alpha_loss", new_action, new_logp, main_weight):
            self._zero_and_step(alpha_loss, [self.alpha_optimizer], [self.log_alpha])

        # 7. Non-negative boundary and feasibility multipliers.
        critical_violation = torch.where(g.detach() >= pf, g.detach() - pf, torch.zeros_like(g)).mean()
        infeasible_increment = torch.where(g.detach() >= pf, g.detach() - pf, torch.zeros_like(g)).mean()
        cg_loss = self.log_cg.exp() * (critical_violation - self.log_cg.exp().detach())
        lambda1_loss = -self.log_lambda1.exp() * critical_violation
        lambda2_loss = -self.log_lambda2.exp() * infeasible_increment
        for loss, optimizer, parameter in (
            (cg_loss, self.cg_optimizer, self.log_cg),
            (lambda1_loss, self.lambda1_optimizer, self.log_lambda1),
            (lambda2_loss, self.lambda2_optimizer, self.log_lambda2),
        ):
            self._zero_and_step(loss, [optimizer], [parameter])

        # 8. Dual safety critics.
        with torch.no_grad():
            dual_next_dist = self._distribution(self.dual_policy, next_state)
            dual_next_action, _ = dual_next_dist.rsample()
            dual_target_g = torch.minimum(
                self.dual_g1_target(next_state, dual_next_action),
                self.dual_g2_target(next_state, dual_next_action),
            ).clamp(0.0, 1.0)
            dual_g_backup = cost + bootstrap_mask * (1.0 - cost) * cost_gamma * dual_target_g
        dual_g1_value = self.dual_g1(state, action)
        dual_g2_value = self.dual_g2(state, action)
        dual_g1_loss = (dual_weight * (dual_g1_value - dual_g_backup).pow(2)).mean()
        dual_g2_loss = (dual_weight * (dual_g2_value - dual_g_backup).pow(2)).mean()
        dual_g_loss = dual_g1_loss + dual_g2_loss
        if self._finite(dual_g_loss, "dual_g_loss", action, new_logp, dual_weight):
            self._zero_and_step(
                dual_g_loss,
                [self.dual_g1_optimizer, self.dual_g2_optimizer],
                list(self.dual_g1.parameters()) + list(self.dual_g2.parameters()),
            )

        # 9. Risk-seeking dual actor with bidirectional Gaussian-space KL.
        self._set_trainable([self.dual_g1, self.dual_g2], False)
        detached_main_dist = TanhGaussDistribution(
            self.main_policy(state).detach(),
            act_low_lim=self.action_low_limit,
            act_high_lim=self.action_high_limit,
        )
        dual_dist = self._distribution(self.dual_policy, state)
        dual_action, _, dual_logp = dual_dist.rsample_with_raw()
        dual_g = torch.minimum(
            self.dual_g1(state, dual_action), self.dual_g2(state, dual_action)
        )
        kl_dual_to_main = dual_dist.kl_divergence(detached_main_dist).mean()
        kl_main_to_dual = detached_main_dist.kl_divergence(dual_dist).mean()
        lambda3 = self.log_lambda3.exp().detach()
        lambda4 = self.log_lambda4.exp().detach()
        dual_policy_loss = (
            (dual_weight * (-dual_g)).mean()
            + lambda3 * kl_dual_to_main
            + lambda4 * kl_main_to_dual
        )
        if self._finite(dual_policy_loss, "dual_policy_loss", dual_action, dual_logp, dual_weight):
            self._zero_and_step(
                dual_policy_loss,
                [self.dual_policy_optimizer],
                list(self.dual_policy.parameters()),
            )
        self._set_trainable([self.dual_g1, self.dual_g2], True)

        # 10. KL multipliers. Minimising this expression increases lambda
        # whenever KL exceeds the target and decreases it below the target.
        target_kl = float(agent_par.get("fdpi_target_kl", 5.0))
        lambda3_loss = self.log_lambda3.exp() * (target_kl - kl_dual_to_main.detach())
        lambda4_loss = self.log_lambda4.exp() * (target_kl - kl_main_to_dual.detach())
        self._zero_and_step(lambda3_loss, [self.lambda3_optimizer], [self.log_lambda3])
        self._zero_and_step(lambda4_loss, [self.lambda4_optimizer], [self.log_lambda4])

        # 11. All target networks use the same Polyak update.
        targets = (
            (self.q1, self.q1_target), (self.q2, self.q2_target),
            (self.h, self.h_target), (self.main_policy, self.main_policy_target),
            (self.g1, self.g1_target), (self.g2, self.g2_target),
            (self.gr1, self.gr1_target), (self.gr2, self.gr2_target),
            (self.dual_g1, self.dual_g1_target), (self.dual_g2, self.dual_g2_target),
        )
        with torch.no_grad():
            for source, target in targets:
                for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                    target_parameter.mul_(1.0 - self.tau).add_(source_parameter, alpha=self.tau)

        with torch.no_grad():
            feasible_mask = (torch.maximum(g1_value, g2_value) < pf) & (cost <= 0.0)
            feasible_ratio = feasible_mask.float().mean()
            self.feasible_ratio_history.append(float(feasible_ratio.cpu()))
            self.global_step += 1
            self.refresh_dual_active()
            action_gap = torch.linalg.vector_norm(new_action - dual_action, dim=-1).mean()

        info = {
            "train/q1_loss": float(q1_loss.detach().cpu()),
            "train/q2_loss": float(q2_loss.detach().cpu()),
            "train/main_policy_loss": float(main_policy_loss.detach().cpu()),
            "train/alpha": float(self.log_alpha.exp().detach().cpu()),
            "train/g1_loss": float(g1_loss.detach().cpu()),
            "train/g2_loss": float(g2_loss.detach().cpu()),
            "train/gr1_loss": float(gr1_loss.detach().cpu()),
            "train/gr2_loss": float(gr2_loss.detach().cpu()),
            "train/dual_g1_loss": float(dual_g1_loss.detach().cpu()),
            "train/dual_g2_loss": float(dual_g2_loss.detach().cpu()),
            "train/dual_policy_loss": float(dual_policy_loss.detach().cpu()),
            "fdpi/feasible_ratio": float(feasible_ratio.cpu()),
            "fdpi/dual_active": float(self.dual_active),
            "fdpi/main_episode_ratio": float(
                np.mean(np.asarray(batch.get("behavior_policy", [])) == "main")
                if len(batch.get("behavior_policy", [])) else 1.0
            ),
            "fdpi/dual_episode_ratio": float(
                np.mean(np.asarray(batch.get("behavior_policy", [])) == "dual")
                if len(batch.get("behavior_policy", [])) else 0.0
            ),
            "fdpi/kl_dual_to_main": float(kl_dual_to_main.detach().cpu()),
            "fdpi/kl_main_to_dual": float(kl_main_to_dual.detach().cpu()),
            "fdpi/lambda1": float(self.log_lambda1.exp().detach().cpu()),
            "fdpi/lambda2": float(self.log_lambda2.exp().detach().cpu()),
            "fdpi/lambda3": float(self.log_lambda3.exp().detach().cpu()),
            "fdpi/lambda4": float(self.log_lambda4.exp().detach().cpu()),
            "fdpi/cg": float(self.log_cg.exp().detach().cpu()),
            "fdpi/main_is_weight_mean": float(main_weight.mean().cpu()),
            "fdpi/dual_is_weight_mean": float(dual_weight.mean().cpu()),
            "fdpi/main_dual_action_l2_gap": float(action_gap.cpu()),
            "fdpi/cost_rate": float(cost.mean().cpu()),
        }
        if logger is not None:
            logger.add(iteration, **info)
        return info

    def checkpoint_state(self):
        networks = (
            "h", "h_target", "q1", "q2", "q1_target", "q2_target",
            "main_policy", "main_policy_target", "dual_policy", "g1", "g2",
            "g1_target", "g2_target", "gr1", "gr2", "gr1_target", "gr2_target",
            "dual_g1", "dual_g2", "dual_g1_target", "dual_g2_target",
        )
        optimizers = (
            "h_optimizer", "q1_optimizer", "q2_optimizer", "policy_optimizer",
            "alpha_optimizer", "g1_optimizer", "g2_optimizer", "gr1_optimizer",
            "gr2_optimizer", "dual_g1_optimizer", "dual_g2_optimizer",
            "dual_policy_optimizer", "cg_optimizer", "lambda1_optimizer",
            "lambda2_optimizer", "lambda3_optimizer", "lambda4_optimizer",
        )
        state = {name: getattr(self, name).state_dict() for name in networks}
        state["optimizers"] = {name: getattr(self, name).state_dict() for name in optimizers}
        state.update(
            state_encoder=self.state_encoder,
            q_state_size=self.q_state_size,
            log_alpha=self.log_alpha.detach().cpu(),
            log_cg=self.log_cg.detach().cpu(),
            lambda1=self.log_lambda1.detach().cpu(),
            lambda2=self.log_lambda2.detach().cpu(),
            log_lambda3=self.log_lambda3.detach().cpu(),
            log_lambda4=self.log_lambda4.detach().cpu(),
            global_step=self.global_step,
            episode=self.episode,
            feasible_ratio_history=list(self.feasible_ratio_history),
            mean_std1=(
                self.mean_std1.detach().cpu()
                if torch.is_tensor(self.mean_std1) else self.mean_std1
            ),
            mean_std2=(
                self.mean_std2.detach().cpu()
                if torch.is_tensor(self.mean_std2) else self.mean_std2
            ),
        )
        return state

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.checkpoint_state(), path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        saved_encoder = checkpoint.get("state_encoder", self.state_encoder)
        saved_size = int(checkpoint.get("q_state_size", self.q_state_size))
        if saved_encoder != self.state_encoder or saved_size != self.q_state_size:
            raise ValueError(
                "Checkpoint state encoder mismatch: saved {} ({} dims), current {} "
                "({} dims). Select the matching algorithm_mode.".format(
                    saved_encoder, saved_size, self.state_encoder, self.q_state_size
                )
            )
        missing_fdpi = "dual_policy" not in checkpoint
        for name in (
            "h", "h_target", "q1", "q2", "q1_target", "q2_target",
            "main_policy", "main_policy_target", "dual_policy", "g1", "g2",
            "g1_target", "g2_target", "gr1", "gr2", "gr1_target", "gr2_target",
            "dual_g1", "dual_g2", "dual_g1_target", "dual_g2_target",
        ):
            if name in checkpoint:
                getattr(self, name).load_state_dict(checkpoint[name])
        if missing_fdpi:
            self.dual_policy.load_state_dict(self.main_policy.state_dict())
            warnings.warn(
                "Legacy DSAC checkpoint has no FDPI networks; dual_policy was "
                "initialised from main_policy and safety critics remain random.",
                RuntimeWarning,
            )
        for key, parameter in (
            ("log_alpha", self.log_alpha), ("log_cg", self.log_cg),
            ("lambda1", self.log_lambda1), ("lambda2", self.log_lambda2),
            ("log_lambda3", self.log_lambda3), ("log_lambda4", self.log_lambda4),
        ):
            if key in checkpoint:
                parameter.data.copy_(checkpoint[key].to(self.device))
        for name, state in checkpoint.get("optimizers", {}).items():
            if hasattr(self, name):
                getattr(self, name).load_state_dict(state)
        self.global_step = int(checkpoint.get("global_step", 0))
        self.episode = int(checkpoint.get("episode", 0))
        self.feasible_ratio_history.clear()
        self.feasible_ratio_history.extend(checkpoint.get("feasible_ratio_history", []))
        self.mean_std1 = checkpoint.get("mean_std1", -1.0)
        self.mean_std2 = checkpoint.get("mean_std2", -1.0)
        if torch.is_tensor(self.mean_std1):
            self.mean_std1 = self.mean_std1.to(self.device)
        if torch.is_tensor(self.mean_std2):
            self.mean_std2 = self.mean_std2.to(self.device)
        self.refresh_dual_active()

    def save_tep(self, name):
        super().save_tep(name)
        self.save_checkpoint(name + "_fdpi_checkpoint.pth")

    def restore(self, policy_logs_path, q1_logs_path, q2_logs_path, h_logs_path):
        try:
            super().restore(policy_logs_path, q1_logs_path, q2_logs_path, h_logs_path)
        except RuntimeError as error:
            warnings.warn(
                "Legacy DSAC checkpoint state dimension is incompatible with "
                "the current {} encoder ({} dims); keeping freshly initialised "
                "FDPI networks. Details: {}".format(
                    self.state_encoder, self.q_state_size, error
                ),
                RuntimeWarning,
            )
            return
        self.dual_policy.load_state_dict(self.main_policy.state_dict())
        warnings.warn(
            "Loaded a legacy four-file DSAC model; dual_policy was initialised "
            "from main_policy and safety critics remain random.",
            RuntimeWarning,
        )

    def save_model(self, name):
        super().save_model(name)
        self.save_checkpoint(name + "_fdpi_checkpoint.pth")

    def continue_train_model(self, folder_path, best_h_rl=None):
        checkpoints = []
        if os.path.isdir(folder_path):
            checkpoints = [
                os.path.join(folder_path, filename)
                for filename in os.listdir(folder_path)
                if filename.endswith("_fdpi_checkpoint.pth")
            ]
        if checkpoints:
            self.load_checkpoint(max(checkpoints, key=os.path.getmtime))
            print("successfully load FDPI checkpoint")
            return
        try:
            super().continue_train_model(folder_path, best_h_rl=best_h_rl)
        except RuntimeError as error:
            warnings.warn(
                "Legacy model could not be loaded into the {}-D {} state "
                "networks; training will start from fresh FDPI weights. "
                "Details: {}".format(self.q_state_size, self.state_encoder, error),
                RuntimeWarning,
            )
            return
        self.dual_policy.load_state_dict(self.main_policy.state_dict())
        warnings.warn(
            "No FDPI checkpoint found; loaded legacy DSAC weights and initialised "
            "dual_policy from main_policy. Safety critics remain random.",
            RuntimeWarning,
        )

    def continue_test_model(self, folder_path, e):
        checkpoints = []
        if os.path.isdir(folder_path):
            checkpoints = [
                os.path.join(folder_path, filename)
                for filename in os.listdir(folder_path)
                if filename.endswith("_fdpi_checkpoint.pth")
            ]
        if checkpoints:
            self.load_checkpoint(max(checkpoints, key=os.path.getmtime))
            print("successfully load FDPI checkpoint for main-policy evaluation")
            return
        try:
            super().continue_test_model(folder_path, e)
        except RuntimeError as error:
            warnings.warn(
                "Legacy evaluation model is incompatible with the current "
                "{} encoder; using the freshly initialised main policy. "
                "Details: {}".format(self.state_encoder, error),
                RuntimeWarning,
            )
            return
        self.dual_policy.load_state_dict(self.main_policy.state_dict())
        warnings.warn(
            "Evaluation loaded legacy DSAC main_policy; dual networks are not used.",
            RuntimeWarning,
        )


# A short alias is convenient in existing worker imports.
Epre_dsac_fdpi_agent = EpreDSACFDPIAgent
