import os

agent_par = {}
agent_par["train"] = True

agent_par["state_dim"] = 53
agent_par["action_dim"] = 2
agent_par["action_low_limit"] = [0.0, -5.0]
agent_par["action_high_limit"] = [8.0, 5.0]
agent_par["rl_device"] = os.environ.get("RL_DEVICE", "cpu")
agent_par["nb_ego_states"] = 19
agent_par["nb_states_per_vehicle"] = 15
agent_par["sensor_nb_vehicles"] = 6

agent_par["safe_rule"] = False

agent_par["cnn"] = False
agent_par["cnn_state"] = False
if agent_par["cnn"]:
    agent_par["cnn_state"] = True

agent_par["use_quantile"] = True
agent_par["N_QUANT"] = 200
agent_par["use_cvar"] = True
agent_par["cvar_eta"] = 0.7
if agent_par["use_cvar"]:
    agent_par["use_quantile"] = True


agent_par["use_spare_reward"] = False
agent_par["use_her"] = False
agent_par["change_episode"] = 20000
if agent_par["use_spare_reward"]:
    agent_par["use_her"] = True

agent_par["learning_rate"] = 1e-4
agent_par["lr_min"] = 1e-5
agent_par["lr_decay"] = 0.94

agent_par["memory_size"] = 100000
agent_par["Prioritized"] = True
agent_par["prioritized_replay_beta0"] = 0.4
agent_par["prioritized_replay_alpha"] = 0.6

agent_par["N"] = 51
agent_par["Vmin"] = -100
agent_par["Vmax"] = 100
agent_par["seed"] = 914
agent_par["tau"] = 0.01

agent_par["batch_size"] = 32
agent_par["gamma"] = 0.995
agent_par["epsilon"] = 0.8
agent_par["epsilon_min"] = 0.02
agent_par["e_decay"] = 5e5



#edsac
agent_par['epre_dsac'] = True
agent_par['reset'] = False
agent_par['predict_map'] = False  #预测出旁车未来可通行路点
agent_par['all_car_map'] = True 
agent_par['use_other_direction'] = False #添加旁车位置信息（0:自车，1：左前，2：正前，3：右前，4：左后，5：正后，6：右后）
agent_par['reset_model'] = False
agent_par['new_best'] = False
agent_par['save_episode'] = 100

agent_par['two_buffer'] = False

agent_par['three_class'] = True
agent_par['three_same_lane'] = True
agent_par['no_tracker'] = True
agent_par['podar'] = False
agent_par['two_agent'] = False
agent_par['check_comfortable'] = False
agent_par['shushidu'] = False

# Feasible Dual Policy Iteration is independent from ``two_agent``.  The
# latter selects separate intersection/straight agents, while FDPI uses one
# environment and fixes one behaviour policy for an entire episode.
# Algorithm/encoder are separate choices:
#   original_dsac   -> legacy Agent (selected by use_epre_dsac=False)
#   stt_dsac        -> Epre_dsac_agent with Transformer HNet
#   dsac_fdpi       -> FDPI with the 24-D Frenet state (current default)
#   dsac_fdpi_stt   -> FDPI with Transformer HNet (reserved/available interface)
agent_par["algorithm_mode"] = "dsac_fdpi"
agent_par["fdpi_state_encoder"] = "frenet"
agent_par["fdpi_enabled"] = agent_par["algorithm_mode"] in (
    "dsac_fdpi",
    "dsac_fdpi_stt",
)
agent_par["fdpi_dual_enabled"] = True
agent_par["fdpi_full_policy_loss"] = True

agent_par["fdpi_pf"] = 0.10
agent_par["fdpi_cost_gamma"] = 0.97
agent_par["fdpi_dual_threshold"] = 0.90
agent_par["fdpi_dual_sample_ratio"] = 0.50
agent_par["fdpi_feasible_window"] = 1000

agent_par["fdpi_beta"] = 0.50
agent_par["fdpi_target_kl"] = 5.0
agent_par["fdpi_min_is_weight"] = 0.10
agent_par["fdpi_max_is_weight"] = 10.0

agent_par["fdpi_cg_init"] = 0.01
agent_par["fdpi_lambda_lr"] = 3e-4
agent_par["fdpi_warmup_steps"] = 10000

agent_par["global_path_future_points"] = 5
agent_par["frenet_pose_dim"] = 3
agent_par["frenet_state_dim"] = 9 + agent_par["global_path_future_points"] * agent_par["frenet_pose_dim"]
agent_par["frenet_state_start"] = 7
agent_par["global_path_state_dim"] = agent_par["frenet_state_dim"]
agent_par["history_feature"] = 8 + (1 if agent_par['use_other_direction'] else 0) + 3
agent_par["state_dim"] = 53 + agent_par["frenet_state_dim"]

# agent_par['only_dangerous_vehicle'] = False #去掉不影响自车的车辆

agent_par['continue_train'] = False
agent_par['train_data'] = {}
agent_par['train_data']['folder_path'] = "samples_epre_wutfsd/model/highway/model_temp"

agent_par['train_data']['folder_path_intersection'] = "samples_epre_wutfsd/model/town/model_temp_intersection"
agent_par['train_data']['folder_path_straight'] = "samples_epre_wutfsd/model/town/model_temp_straight"
agent_par['train_data']['episode'] = 0
agent_par['train_data']['update_time'] = 0

agent_par['train_data']['global_step'] = 0
# agent_par['train_data']['h_rl'] = 2e-5
