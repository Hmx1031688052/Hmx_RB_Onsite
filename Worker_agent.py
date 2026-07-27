import multiprocessing as mp
import queue
import traceback
from agent import Agent
from epre_dsac.Epre_dsac import Epre_dsac_agent
from epre_dsac.Epre_dsac_fdpi import EpreDSACFDPIAgent
from epre_dsac.epre_reply_buffer import Reply_Buffer as epre_replay_buffer

from env import Env
import time
import os
import numpy as np
import pandas as pd
from utilss import logger
from collections import deque
import torch
from torch.autograd import Variable
import torch
from torch.nn.functional import mse_loss
from torch.autograd import Variable
from reply_buffer import Reply_Buffer
from config import Config
from utils.opendrive2discretenet import parse_opendrive
from epre_dsac.parameters import agent_par
from epre_dsac.fdpi_sampling import (
    accumulate_importance_weights,
    select_episode_behavior_policy as choose_episode_behavior_policy,
)
from Worker_agent_train import Worker_paral_train


os.environ['PYTHONHASHSEED'] = '0'

class Worker_paral(mp.Process):
    def __init__(self, input_queue, output_queue, train, use_epre_dsac = False):
        mp.Process.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.train = train
        self.state = None
        self.next_state = None
        self.env_input = None
        self.next_env_input = None
        self.env_map = None
        self.next_env_map = None
        self.action = None
        self.logp = None
        self.cost = 0.0
        self.logp_main = 0.0
        self.logp_dual = 0.0
        self.log_is_to_main = 0.0
        self.log_is_to_dual = 0.0
        self.behavior_policy = "main"
        self.dual_active = False
        self.mean_feasible_ratio = 0.0
        self.cumulative_log_is_to_main = 0.0
        self.cumulative_log_is_to_dual = 0.0
        self.total_reward = 0
        self.total_reward_intersection = 0
        self.total_reward_straight = 0
        if agent_par['two_agent']:
            self.total_reward_intersection = 0
            self.total_reward_straight = 0
            self.intersection = False
        self.use_epre_dsac = use_epre_dsac
        if self.use_epre_dsac:
            if agent_par['new_best']:
                self.best_time = 1
                self.target_best_time = 1
            if agent_par['reset_model']:
                self.last_ave_reward = 0
        self.info_list = []
        if agent_par['two_agent']:
            self.buffer_list_intersection = []
            self.buffer_list_straight = []
        else:
            self.buffer_list = []
        self.state = None
        self.next_state = None
        self.env_input = None
        self.next_env_input = None
        self.env_map = None
        self.next_env_map = None
        self.action = None
        self.logp = None
        if agent_par['two_agent']:
            self.collision_intersection = False
            self.collision_straight = False
            self.collision = False
        else:
            self.collision = False
        self.time_out = False

        if self.train:
            self.input_queue_train = mp.Queue()
            self.output_queue_train = mp.Queue()
            worker_train = Worker_paral_train(self.input_queue_train, self.output_queue_train, train=self.train, use_epre_dsac = self.use_epre_dsac)
            worker_train.daemon = True
            worker_train.start()
        self.train_request_id = 0

        print("Worker_paral __init__")

    def run(self):
        # save path
        localtime = time.strftime("%Y-%m-%d_%H%M", time.localtime())
        if agent_par['two_agent']:
            ModelPath_intersection = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/model_intersection/'
            ModelPath_temp_intersection = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/model_temp_intersection/'
            ModelPath_straight = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/model_straight/'
            ModelPath_temp_straight = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/model_temp_straight/'
            logpath = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/logagent/'
        else:
            ModelPath = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/model/'
            ModelPath_temp = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/model_temp/'
            logpath = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/logagent/'
        self.q1_name = 'q1_{}_{}_{}.pth'.format(Config.reply_buffer_size, Config.total_episode, Config.lr)
        self.q2_name = 'q2_{}_{}_{}.pth'.format(Config.reply_buffer_size, Config.total_episode, Config.lr)
        self.policy_name = 'policy_{}_{}_{}.pth'.format(Config.reply_buffer_size, Config.total_episode, Config.lr)
        self.h_name = 'h_{}_{}_{}.pth'.format(Config.reply_buffer_size, Config.total_episode, Config.lr)
        self.alpha_name = 'alpha_{}_{}_{}.pth'.format(Config.reply_buffer_size, Config.total_episode, 0.003)

        
        self.folder_path = "samples_epre_wutfsd/logs/2025-04-08_2334/model_temp1111" 
        
        if self.use_epre_dsac:
            if agent_par['two_agent']:
                self.agent_straight = Epre_dsac_agent(ModelPath=ModelPath_straight, ModelPath_temp=ModelPath_temp_straight, hrl = True)
                self.replay_buffer_straight = epre_replay_buffer(Config.reply_buffer_size, fdpi_enabled=False)
                self.agent_intersection = Epre_dsac_agent(ModelPath=ModelPath_intersection, ModelPath_temp=ModelPath_temp_intersection, hrl = True)
                self.replay_buffer_intersection = epre_replay_buffer(Config.reply_buffer_size, fdpi_enabled=False)
                self.average_episode_reward_intersection = deque(maxlen=100)
                self.average_episode_reward_straight = deque(maxlen=100)
            else:
                agent_class = (
                    EpreDSACFDPIAgent
                    if agent_par.get("fdpi_enabled", False)
                    else Epre_dsac_agent
                )
                self.agent = agent_class(ModelPath=ModelPath, ModelPath_temp=ModelPath_temp, hrl = True)
                self.replay_buffer = epre_replay_buffer(Config.reply_buffer_size)
                self.ModelPath_temp = ModelPath_temp
        else:
            self.agent =  Agent(ModelPath,ModelPath_temp, self.folder_path)
            self.replay_buffer = Reply_Buffer(Config.reply_buffer_size)
            self.ModelPath_temp = ModelPath_temp
        self.logger = logger(path=logpath)
        if self.use_epre_dsac:
            if agent_par['two_agent']:
                self.device = self.agent_intersection.device
            else:
                self.device = self.agent.device
        else:
            self.device = torch.device(agent_par.get("rl_device", "cpu"))
        self.step = 0
        self.episode = 0
        self.global_step = 0
        self.totalspeed = deque(maxlen = 3000)
        self.average_episode_reward = deque(maxlen=100)
        self.average_episode_reward_net = deque(maxlen=800)

        self.average_goal = deque(maxlen=100)
        self.average_collision = deque(maxlen=100)
        if agent_par['continue_train']:
            self.episode = agent_par['train_data']['episode']
            self.global_step = agent_par['train_data']['global_step']

        print("Worker_paral run")
        
        while True:
            try:
                # Process every request in FIFO order.  Dropping all but the
                # last item breaks transition/action/log-probability alignment.
                k = self.input_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if k:
                if self.train:
                    if k[0] == 'act':
                        request_id = k[1]
                        try:
                            control = self.act(k[2], k[3], k[4], k[5], k[6], k[7], k[8])
                            self.output_queue.put(['act success!', request_id, control])
                        except Exception:
                            error = traceback.format_exc()
                            print("[worker][ERROR] act failed", flush=True)
                            print(error, flush=True)
                            self.output_queue.put(['act failed!', request_id, error])
                    if k[0] == 'end':
                        try:
                            self.act_end(k[1])
                            self.output_queue.put(['act_end success!'])
                        except Exception:
                            error = traceback.format_exc()
                            print("[worker][ERROR] act_end failed", flush=True)
                            print(error, flush=True)
                            self.output_queue.put(['act_end failed!', error])
                    if k[0] == 'load_net':
                        self.load_net(k[1])
                        self.output_queue.put(['load_net success!'])
                    if k[0] == 'load_continue_net':
                        self.load_continue_net()
                        self.output_queue.put(['load_continue_net success!'])
                    if k[0] == 'end work':
                        break

                else:
                    if k[0] == 'act':
                        request_id = k[1]
                        try:
                            control = self.act(k[2], k[3], k[4], k[5], k[6], k[7], k[8])
                            self.output_queue.put(['act success!', request_id, control])
                        except Exception:
                            error = traceback.format_exc()
                            print("[worker][ERROR] act failed", flush=True)
                            print(error, flush=True)
                            self.output_queue.put(['act failed!', request_id, error])
                    if k[0] == 'end':
                        try:
                            self.act_end(k[1])
                            self.output_queue.put(['act_end success!'])
                        except Exception:
                            error = traceback.format_exc()
                            print("[worker][ERROR] act_end failed", flush=True)
                            print(error, flush=True)
                            self.output_queue.put(['act_end failed!', error])
                    if k[0] == 'load_net':
                        self.load_net(k[1])
                        self.output_queue.put(['load_net success!'])
                    if k[0] == 'load_continue_net':
                        self.load_continue_net()
                        self.output_queue.put(['load_continue_net success!'])
                    if k[0] == 'end work':
                        break

    def load_net(self, net):
        policy_path = net[0]
        q1_path = net[1]
        q2_path = net[2]
        h_path = net[3]
        self.agent.restore(policy_path, q1_path, q2_path, h_path)
        print('restore success!')

    def select_episode_behavior_policy(self):
        return choose_episode_behavior_policy(
            self.train,
            agent_par.get("fdpi_enabled", False),
            self.dual_active,
            float(agent_par.get("fdpi_dual_sample_ratio", 0.5)),
        )

    def _reset_fdpi_episode_state(self, select_policy=True):
        self.cumulative_log_is_to_main = 0.0
        self.cumulative_log_is_to_dual = 0.0
        self.log_is_to_main = 0.0
        self.log_is_to_dual = 0.0
        if select_policy:
            self.behavior_policy = self.select_episode_behavior_policy()

    def _update_importance_weights(self, logp_main, logp_dual):
        self.cumulative_log_is_to_main, self.cumulative_log_is_to_dual = accumulate_importance_weights(
            self.behavior_policy,
            logp_main,
            logp_dual,
            self.cumulative_log_is_to_main,
            self.cumulative_log_is_to_dual,
            float(agent_par.get("fdpi_beta", 0.5)),
            float(agent_par.get("fdpi_min_is_weight", 0.1)),
            float(agent_par.get("fdpi_max_is_weight", 10.0)),
        )
        return self.cumulative_log_is_to_main, self.cumulative_log_is_to_dual

    def act(self, state, reward, cost, done, collision_done, time_out_done, is_intersection):
        time1 = time.time()
        self.reward = reward
        self.cost = cost
        self.done = done
        if self.use_epre_dsac:
            state_list = state
            state = state_list[0]
            env_input = state_list[1]
            env_map = state_list[2]
            has_prev_transition = self.train and self.state is not None and self.action is not None and self.logp is not None
            if has_prev_transition:
                self.next_state = state
                self.next_env_input = env_input
                self.next_env_map = env_map
                if agent_par['two_agent']:
                    if is_intersection or self.intersection:
                        if not self.collision_intersection and not self.time_out:
                            if collision_done or time_out_done:
                                batch = (self.state,self.action,self.reward,self.next_state,1,self.logp, self.env_input, self.next_env_input, self.env_map, self.next_env_map)
                            elif not is_intersection:
                                batch = (self.state,self.action,self.reward + 300,self.next_state,1,self.logp, self.env_input, self.next_env_input, self.env_map, self.next_env_map)
                            else:
                                batch = (self.state,self.action,self.reward,self.next_state,self.done,self.logp, self.env_input, self.next_env_input, self.env_map, self.next_env_map)
                            self.buffer_list_intersection.append(batch)
                    else:
                        if not self.collision_straight and not self.time_out:
                            if collision_done or time_out_done:
                                batch = (self.state,self.action,self.reward,self.next_state,1,self.logp, self.env_input, self.next_env_input, self.env_map, self.next_env_map)
                            else:
                                batch = (self.state,self.action,self.reward,self.next_state,self.done,self.logp, self.env_input, self.next_env_input, self.env_map, self.next_env_map)
                            self.buffer_list_straight.append(batch)
                elif agent_par.get("fdpi_enabled", False):
                    terminated = bool(collision_done or (done and not time_out_done))
                    truncated = bool(time_out_done)
                    batch = {
                        "state": self.state[0],
                        "env_input": self.env_input,
                        "env_map": self.env_map,
                        "action": self.action,
                        "reward": float(self.reward),
                        "cost": float(self.cost),
                        "next_state": self.next_state[0],
                        "next_env_input": self.next_env_input,
                        "next_env_map": self.next_env_map,
                        "terminated": terminated,
                        "truncated": truncated,
                        "behavior_policy": self.behavior_policy,
                        "logp_main": float(self.logp_main),
                        "logp_dual": float(self.logp_dual),
                        "log_is_to_main": float(self.log_is_to_main),
                        "log_is_to_dual": float(self.log_is_to_dual),
                    }
                    self.buffer_list.append(batch)
                else:
                    if not self.collision and not self.time_out:
                        if collision_done or time_out_done:
                            batch = (self.state,self.action,self.reward,self.next_state,1,self.logp, self.env_input, self.next_env_input, self.env_map, self.next_env_map)
                        else:
                            batch = (self.state,self.action,self.reward,self.next_state,self.done,self.logp, self.env_input, self.next_env_input, self.env_map, self.next_env_map)
                        self.buffer_list.append(batch)
            else:
                if self.state is not None and (self.action is None or self.logp is None):
                    print("[worker][WARN] skip transition because previous action/logp is missing", flush=True)
                self.state = state
                self.env_input = env_input
                self.env_map = env_map
                self.next_state = None
                self.next_env_input = None
                self.next_env_map = None

            state1 = torch.from_numpy(state).to(self.device).float()
            env_input1 = torch.from_numpy(env_input).to(self.device).float()
            env_map1 = torch.from_numpy(env_map).to(self.device).float()
            with torch.no_grad():
                if agent_par['two_agent']:
                    if is_intersection:
                        uesc = self.agent_intersection.use_h(state1, env_input1, env_map1)
                    else:
                        uesc = self.agent_straight.use_h(state1, env_input1, env_map1)
                else:
                    uesc = self.agent.encode_policy_state(state1, env_input1, env_map1)
            self.state = state
            self.env_input = env_input
            self.env_map = env_map

        else:
            has_prev_transition = self.train and self.state is not None and self.action is not None and self.logp is not None
            if has_prev_transition:
                self.next_state = state
                self.replay_buffer.append((self.state, self.action, self.reward, self.next_state, self.done, self.logp))
            else:
                if self.state is not None and (self.action is None or self.logp is None):
                    print("[worker][WARN] skip transition because previous action/logp is missing", flush=True)
                self.state = state
                self.next_state = None
            state1 = torch.from_numpy(state).to(self.device).float()
            with torch.no_grad():
                uesc = self.agent.use_h(state1)
            self.state = state

        if agent_par['two_agent'] and self.use_epre_dsac:
            if is_intersection:
                action, logp = self.agent_intersection.take_action(uesc, train = self.train)
                # print('intersection', action)
            else:
                action, logp = self.agent_straight.take_action(uesc, train = self.train)
                # print('straight', action)
        elif self.use_epre_dsac and agent_par.get("fdpi_enabled", False):
            result = self.agent.take_fdpi_action(
                uesc, behavior_policy=self.behavior_policy, train=self.train
            )
            action = result["action"]
            self.logp_main = float(result["logp_main"])
            self.logp_dual = float(result["logp_dual"])
            self.log_is_to_main, self.log_is_to_dual = self._update_importance_weights(
                self.logp_main, self.logp_dual
            )
            logp = self.logp_main if self.behavior_policy == "main" else self.logp_dual
        else:
            action, logp = self.agent.take_action(uesc, train = self.train)
        
      
        self.action = action
        self.logp = logp
        self.global_step += 1
        self.step += 1
        if agent_par['two_agent']:
            if is_intersection or self.intersection:
                if not self.collision_intersection and not self.time_out:
                    if not is_intersection and not collision_done:
                        reward1 = 300
                    elif abs(reward) < 50:
                        reward1 = reward * 0.04
                    else:
                        reward1 = reward
                    self.total_reward_intersection += reward1
            else:
                if not self.collision_straight and not self.time_out:
                    if abs(reward) < 50:
                        reward1 = reward * 0.04
                    else:
                        reward1 = reward
                    self.total_reward_straight += reward1
            if not self.collision and not self.time_out:
                if abs(reward) < 50:
                    reward1 = reward * 0.04
                else:
                    reward1 = reward
                self.total_reward += reward1
            

        else:
            if not self.collision and not self.time_out:
                if abs(reward) < 50:
                    reward1 = reward * 0.04
                else:
                    reward1 = reward
                self.total_reward += reward1
        self.totalspeed.append(self.state[0][2]*50)
        # if self.global_step > Config.batch_size + 1 and self.train:
        #     self.train_dsac()
        # print('action',time.time() - time1)
        if collision_done:
            if agent_par['two_agent']:
                if is_intersection:
                    self.collision_intersection = True
                else:
                    self.collision_straight = True
                self.collision = True
            else:
                self.collision = True
        if time_out_done:
            self.time_out = True

        if agent_par['two_agent']:
            self.intersection = is_intersection

        return action

    # def train_dsac(self, ):
    #     if self.use_epre_dsac:
    #         if len(self.replay_buffer.buffer)-5 > Config.batch_size:
    #             batch_state, batch_action, batch_reward, batch_state_new, batch_done, batch_logp, batch_env_input, batch_next_env_input, batch_env_map, batch_next_env_map = self.replay_buffer.sample(Config.batch_size)
    #             self.agent.update_Q_network(batch_state, batch_action, batch_reward, batch_state_new, batch_done, batch_logp, self.global_step, self.logger, batch_env_input, batch_next_env_input, batch_env_map, batch_next_env_map)
    #             if self.agent.hrl and self.agent.h_optimizer.param_groups[0]['lr'] > 1e-6:
    #                 self.agent.h_scheduler.step()
    #             else:
    #                 self.agent.hrl = False
    #     else:
    #         batch_state, batch_action, batch_reward, batch_state_new, batch_done,  batch_logp = self.replay_buffer.sample(Config.batch_size)
    #         # update policy network
    #         self.agent.update_Q_network(batch_state, batch_action, batch_reward, batch_state_new, batch_done, batch_logp, self.global_step, self.logger)
    #         #agent.soft_update_target_network(Config.tau)

    def act_end(self, data):
        collision = data[0]
        done_type = data[1]
        real = data[2]
        if done_type == 'goal':
            goal = 1
        else:
            goal = 0
        speed = np.mean(self.totalspeed)

        reward_xishu = 1
        speed_xishu = 1
        # if self.episode % 800 == 0:
        #     reward_xishu *= 0.95
        #     speed_xishu *= 0.97
        reward_xishu = max(reward_xishu, 0.8)
        speed_xishu = max(speed_xishu, 0.89)
        
        if done_type!='termination' and done_type != 'out' and real:
            self.episode += 1
            # if collision >= 1:
            #     self.total_reward = -200 + (self.total_reward + 200)*0.03
            # elif done_type == 'goal':
            #     self.total_reward = 300 + (self.total_reward - 300)*0.03
            self.average_episode_reward.append(self.total_reward)
            self.average_episode_reward_net.append(self.total_reward)
            self.average_goal.append(goal)
            self.average_collision.append(collision)
            if agent_par['two_agent']:
                if self.total_reward_intersection != 0:
                    self.average_episode_reward_intersection.append(self.total_reward_intersection)
                if self.total_reward_straight != 0:
                    self.average_episode_reward_straight.append(self.total_reward_straight)
                self.logger.add(self.episode, average_episode_reward = np.mean(self.average_episode_reward)*reward_xishu,
                average_episode_reward_intersection = np.mean(self.average_episode_reward_intersection)*reward_xishu,average_episode_reward_straight = np.mean(self.average_episode_reward_straight)*reward_xishu,
                average_goal = np.mean(self.average_goal), average_collision = np.mean(self.average_collision)*speed_xishu, average_spisode_speed = speed/speed_xishu)

            else:
                self.logger.add(self.episode, average_episode_reward = np.mean(self.average_episode_reward)*reward_xishu,
                average_goal = np.mean(self.average_goal), average_collision = np.mean(self.average_collision)*speed_xishu, average_spisode_speed = speed/speed_xishu)
        if real:
            if agent_par['two_agent']:
                print('Episode: {} |  reward: {:.2f} | time: {} | speed:{:.2f} | collision:{} | info: {} | int_reward: {} | str_reward: {} '.format
                    (self.episode, self.total_reward, self.step + 1, speed, collision,  done_type, self.total_reward_intersection, self.total_reward_straight))
            else:
                print('Episode: {} |  reward: {:.2f} | time: {} | speed:{:.2f} | collision:{} | info: {} '.format
                    (self.episode, self.total_reward, self.step + 1, speed, collision,  done_type))
            # print('xishu',reward_xishu, speed_xishu)
        self.total_reward = 0
        self.total_reward_intersection = 0
        self.total_reward_straight = 0
        self.intersection = False
        self.totalspeed = deque(maxlen = 3000)
        self.step = 0
        if agent_par['two_agent']:
            self.collision_intersection = False
            self.collision_straight = False
            self.collision = False
        else:
            self.collision = False
        self.time_out = False
        self._reset_fdpi_episode_state(select_policy=False)

        if not self.train:
            self.state = self.next_state = None
            self.env_input = self.next_env_input = None
            self.env_map = self.next_env_map = None
            self.action = self.logp = None
            self.buffer_list = []
            self._reset_fdpi_episode_state(select_policy=True)
            return

        time1 = time.time()
        # if self.global_step > Config.batch_size + 1 and self.train:
        #     for i in range(50):
        #         self.train_dsac()
        def _buffer_shape_text(buffer):
            if not buffer:
                return "empty"
            sample = buffer[0]
            try:
                return (
                    f"state={np.shape(sample[0])} next_state={np.shape(sample[3])} "
                    f"env_input={np.shape(sample[6])} next_env_input={np.shape(sample[7])} "
                    f"env_map={np.shape(sample[8])} next_env_map={np.shape(sample[9])}"
                )
            except Exception as exc:
                return f"shape_unavailable: {exc}"

        self.train_request_id += 1
        request_id = self.train_request_id
        if agent_par['two_agent']:
            print(
                f"[finish-debug][worker] before input_queue_train.put end "
                f"two_agent=True int_buffer={len(self.buffer_list_intersection)} "
                f"str_buffer={len(self.buffer_list_straight)} done_type={done_type} real={real}",
                flush=True,
            )
            print(
                f"[finish-debug][worker] int_shape={_buffer_shape_text(self.buffer_list_intersection)} "
                f"str_shape={_buffer_shape_text(self.buffer_list_straight)}",
                flush=True,
            )
            self.input_queue_train.put(['end', self.buffer_list_intersection, self.buffer_list_straight, done_type, real, request_id])
        else:
            print(
                f"[finish-debug][worker] before input_queue_train.put end "
                f"two_agent=False buffer={len(self.buffer_list)} done_type={done_type} real={real}",
                flush=True,
            )
            print(f"[finish-debug][worker] buffer_shape={_buffer_shape_text(self.buffer_list)}", flush=True)
            self.input_queue_train.put(['end', self.buffer_list, None, done_type, real, request_id])
        print("[finish-debug][worker] after input_queue_train.put end", flush=True)
        print("[finish-debug][worker] before output_queue_train.get", flush=True)
        train_end_timeout = float(os.environ.get("E2E_TRAIN_END_TIMEOUT", "30.0"))
        deadline = time.time() + train_end_timeout
        try:
            while True:
                timeout_left = max(0.0, deadline - time.time())
                k = self.output_queue_train.get(timeout=timeout_left)
                response_id = k[-1] if len(k) > 1 else request_id
                if response_id == request_id:
                    break
                print(
                    f"[finish-debug][worker][WARN] ignore stale train response "
                    f"response_id={response_id} current_request_id={request_id}",
                    flush=True,
                )
        except queue.Empty:
            print(
                f"[finish-debug][worker][WARN] output_queue_train.get timeout "
                f"after {train_end_timeout:.1f}s request_id={request_id}; skip model sync for this episode",
                flush=True,
            )
            k = ['send_model timeout!', None, None, None, None, None, request_id]
        print(f"[finish-debug][worker] after output_queue_train.get k0={k[0] if k else None}", flush=True)
        if k[0] == 'send_model success!' and k[1] is not None:
            if agent_par['two_agent']:
                self.agent_intersection.h.load_state_dict(k[1])
                self.agent_intersection.policy.load_state_dict(k[2])
                self.agent_straight.h.load_state_dict(k[3])
                self.agent_straight.policy.load_state_dict(k[4])
            else:
                self.agent.h.load_state_dict(k[1])
                self.agent.main_policy.load_state_dict(k[2])
                if agent_par.get("fdpi_enabled", False) and k[3] is not None:
                    self.agent.dual_policy.load_state_dict(k[3])
                    self.dual_active = bool(k[4])
                    self.mean_feasible_ratio = float(k[5])
            print('train_time', time.time() - time1)
        elif k[0] not in ('send_model success!', 'send_model timeout!'):
            print(f"[finish-debug][worker][WARN] train worker returned {k[0]}: {k[1] if len(k) > 1 else ''}", flush=True)
        
        if agent_par['two_agent']:
            self.buffer_list_intersection = []
            self.buffer_list_straight = []
        else:
            self.buffer_list = []
        self.state = None
        self.next_state = None
        self.env_input = None
        self.next_env_input = None
        self.env_map = None
        self.next_env_map = None
        self.action = None
        self.logp = None
        self._reset_fdpi_episode_state(select_policy=True)
        

    def epre_save_buffer(self, name):
        if not os.path.exists(self.ModelPath_temp):
            os.makedirs(self.ModelPath_temp)
        if agent_par['two_buffer']:
            torch.save(self.collision_memory, name + "_collision_buffer_" + str(0) + ".pth")
            torch.save(self.other_memory, name + "_other_buffer_" + str(0) + ".pth")
        else:
            torch.save(self.replay_buffer, name + "_buffer_" + str(0) + ".pth")
    
    def save_buffer(self, name):
        if not os.path.exists(self.ModelPath_temp):
            os.makedirs(self.ModelPath_temp)
        torch.save(self.replay_buffer, name + "_buffer_" + str(0) + ".pth")
    
    def del_files(self,path_file):
        ls = os.listdir(path_file)
        for i in ls:
            f_path = os.path.join(path_file, i)
            # 判断是否是一个目录,若是,则递归删除
            if os.path.isdir(f_path):
                self.del_files(f_path)
            else:
                os.remove(f_path)
    
    def continue_train_buffer(self,folder_path):
        if os.path.exists(str(folder_path)):

            file_names = []  # 用于保存文件名
            file_paths = []  # 用于保存文件路径

            # 遍历文件夹中的每一个文件
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_name = file  # 文件名（包含后缀）
                    file_path = os.path.join(root, file)  # 文件绝对路径
                    file_names.append(file_name)
                    file_paths.append(file_path)
            for i in range(len(file_names)):
                if str(file_names[i].split('_')[-2]) == 'buffer':
                    self.replay_buffer = torch.load(str(file_paths[i]))
                    print('successfully load buffer')
        else:
            print('new buffer')

    def load_continue_net(self):
        if agent_par['two_agent']:
            self.continue_train_buffer(self.folder_path_intersection, True)
            self.agent_intersection.continue_train_model(self.folder_path_intersection)
            print('load_continue_net success!', self.folder_path_intersection)
            self.continue_train_buffer(self.folder_path_straight, False)
            self.agent_straight.continue_train_model(self.folder_path_straight)
            print('load_continue_net success!', self.folder_path_straight)
        else:
            self.continue_train_buffer(self.folder_path)
            self.agent.continue_train_model(self.folder_path)
            print('load_continue_net success!', self.folder_path)
