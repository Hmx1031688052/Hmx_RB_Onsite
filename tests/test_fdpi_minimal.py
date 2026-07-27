import gc
import math
import os
import tempfile
import unittest

import numpy as np
import torch

from epre_dsac.Epre_dsac import Epre_dsac_agent
from epre_dsac.Epre_dsac_fdpi import EpreDSACFDPIAgent
from epre_dsac.epre_reply_buffer import Reply_Buffer
from epre_dsac.parameters import agent_par
from utilsa.act_distribution_cls import TanhGaussDistribution


class FDPIMinimalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_enabled = agent_par["fdpi_enabled"]
        agent_par["fdpi_enabled"] = True
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.agent = EpreDSACFDPIAgent(cls.tempdir.name, cls.tempdir.name)

    @classmethod
    def tearDownClass(cls):
        agent_par["fdpi_enabled"] = cls.previous_enabled
        del cls.agent
        cls.tempdir.cleanup()
        gc.collect()

    def test_distribution_raw_log_prob_and_bounds(self):
        logits = torch.cat((torch.zeros(16, 2), torch.ones(16, 2)), dim=-1)
        distribution = TanhGaussDistribution(logits, [0.0, -5.0], [8.0, 5.0])
        action, raw, logp = distribution.rsample_with_raw()
        self.assertTrue(torch.all(action >= torch.tensor([0.0, -5.0])))
        self.assertTrue(torch.all(action <= torch.tensor([8.0, 5.0])))
        self.assertTrue(torch.allclose(logp, distribution.log_prob_from_raw(raw), atol=1e-6))
        self.assertTrue(torch.allclose(logp, distribution.log_prob(action), atol=2e-5))

    def test_policy_shapes_bounds_and_deterministic_main_only(self):
        encoded = torch.randn(8, self.agent.q_state_size)
        self.assertEqual(self.agent.main_policy(encoded).shape, (8, 4))
        self.assertEqual(self.agent.dual_policy(encoded).shape, (8, 4))
        main = self.agent.take_fdpi_action(encoded, "main", train=True)
        dual = self.agent.take_fdpi_action(encoded, "dual", train=True)
        evaluation = self.agent.take_fdpi_action(encoded, "dual", train=False)
        for result in (main, dual, evaluation):
            self.assertEqual(result["action"].shape, (8, 2))
            self.assertTrue(np.all(result["action"] >= np.array([0.0, -5.0])))
            self.assertTrue(np.all(result["action"] <= np.array([8.0, 5.0])))
        self.assertEqual(evaluation["behavior_policy"], "main")

    def test_plain_fdpi_uses_only_frenet_state(self):
        self.assertEqual(self.agent.state_encoder, "frenet")
        self.assertEqual(self.agent.q_state_size, agent_par["frenet_state_dim"])
        full_state = torch.arange(77, dtype=torch.float32).reshape(1, 77)
        encoded = self.agent.encode_policy_state(full_state)
        start = agent_par["frenet_state_start"]
        end = start + agent_par["frenet_state_dim"]
        self.assertTrue(torch.equal(encoded, full_state[:, start:end]))

    def test_fdpi_stt_encoder_interface_is_available(self):
        previous_mode = agent_par["algorithm_mode"]
        agent_par["algorithm_mode"] = "dsac_fdpi_stt"
        with tempfile.TemporaryDirectory() as directory:
            stt_agent = EpreDSACFDPIAgent(directory, directory)
            self.assertEqual(stt_agent.state_encoder, "stt")
            self.assertEqual(stt_agent.q_state_size, 128 * 7)
        agent_par["algorithm_mode"] = previous_mode

    def test_importance_weight_convention_and_clipping(self):
        logp_main, logp_dual = -2.0, -3.0
        main_step = (0.0, logp_dual - logp_main)
        dual_step = (logp_main - logp_dual, 0.0)
        self.assertEqual(main_step[0], 0.0)
        self.assertEqual(main_step[1], -1.0)
        self.assertEqual(dual_step[0], 1.0)
        self.assertEqual(dual_step[1], 0.0)
        lower = math.log(agent_par["fdpi_min_is_weight"])
        upper = math.log(agent_par["fdpi_max_is_weight"])
        clipped = np.clip(np.array([-100.0, 100.0]), lower, upper)
        weights = np.exp(clipped)
        self.assertTrue(np.all(weights >= agent_par["fdpi_min_is_weight"] - 1e-6))
        self.assertTrue(np.all(weights <= agent_par["fdpi_max_is_weight"] + 1e-6))

    def test_episode_policy_scheduler(self):
        from epre_dsac.fdpi_sampling import (
            accumulate_importance_weights,
            select_episode_behavior_policy,
        )

        self.assertEqual(select_episode_behavior_policy(True, True, False, 1.0), "main")
        behavior = select_episode_behavior_policy(True, True, True, 1.0, random_value=0.5)
        self.assertEqual(behavior, "dual")
        weights = accumulate_importance_weights(
            behavior, -2.0, -3.0, 0.0, 0.0, 0.5, 0.1, 10.0
        )
        self.assertEqual(behavior, "dual")
        self.assertEqual(weights[1], 0.0)

    def test_replay_schema_and_capacity(self):
        replay = Reply_Buffer(2, fdpi_enabled=True)
        row = {
            "state": np.zeros(3), "env_input": np.zeros(2), "env_map": np.zeros(2),
            "action": np.zeros(2), "reward": 1.0, "cost": 0.0,
            "next_state": np.ones(3), "next_env_input": np.ones(2),
            "next_env_map": np.ones(2), "terminated": False, "truncated": True,
            "behavior_policy": "main", "logp_main": -1.0, "logp_dual": -2.0,
            "log_is_to_main": 0.0, "log_is_to_dual": -1.0,
        }
        replay.append(row)
        replay.append(row)
        replay.append(row)
        self.assertEqual(len(replay), 2)
        batch = replay.sample(2)
        self.assertEqual(set(batch), set(row))
        self.assertEqual(batch["action"].shape, (2, 2))

    def test_one_encoded_update_is_finite(self):
        batch_size = 4
        batch = {
            "encoded_state": np.random.randn(batch_size, self.agent.q_state_size).astype(np.float32),
            "encoded_next_state": np.random.randn(batch_size, self.agent.q_state_size).astype(np.float32),
            "action": np.column_stack((
                np.random.uniform(0.0, 8.0, batch_size),
                np.random.uniform(-5.0, 5.0, batch_size),
            )).astype(np.float32),
            "reward": np.random.randn(batch_size).astype(np.float32),
            "cost": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "terminated": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "truncated": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "log_is_to_main": np.zeros(batch_size, dtype=np.float32),
            "log_is_to_dual": np.zeros(batch_size, dtype=np.float32),
            "behavior_policy": np.array(["main", "dual", "main", "dual"]),
        }
        info = self.agent.update(batch)
        self.assertTrue(all(np.isfinite(value) for value in info.values()))
        self.assertIn("train/dual_policy_loss", info)
        self.assertIn("fdpi/feasible_ratio", info)

    def test_checkpoint_restores_all_fdpi_networks(self):
        path = os.path.join(self.tempdir.name, "fdpi_test.pth")
        reference = next(self.agent.dual_g1.parameters()).detach().clone()
        self.agent.save_checkpoint(path)
        with torch.no_grad():
            next(self.agent.dual_g1.parameters()).add_(10.0)
        self.agent.load_checkpoint(path)
        self.assertTrue(torch.equal(reference, next(self.agent.dual_g1.parameters())))
        checkpoint = self.agent.checkpoint_state()
        required_networks = {
            "h", "h_target", "q1", "q2", "q1_target", "q2_target",
            "main_policy", "main_policy_target", "dual_policy", "g1", "g2",
            "g1_target", "g2_target", "gr1", "gr2", "gr1_target", "gr2_target",
            "dual_g1", "dual_g2", "dual_g1_target", "dual_g2_target",
        }
        self.assertTrue(required_networks.issubset(checkpoint))
        self.assertIn("optimizers", checkpoint)


class LegacyDSACCompatibilityTest(unittest.TestCase):
    def test_fdpi_disabled_legacy_agent_initializes_and_has_trainable_loss(self):
        previous = agent_par["fdpi_enabled"]
        agent_par["fdpi_enabled"] = False
        with tempfile.TemporaryDirectory() as directory:
            agent = Epre_dsac_agent(directory, directory)
            encoded = torch.randn(2, agent.q_state_size)
            action = torch.tensor([[1.0, 0.0], [2.0, -1.0]])
            loss = agent.q1(encoded, action).mean()
            agent.q1_optimizer.zero_grad()
            loss.backward()
            agent.q1_optimizer.step()
            self.assertTrue(torch.isfinite(loss))
        agent_par["fdpi_enabled"] = previous


if __name__ == "__main__":
    unittest.main()
