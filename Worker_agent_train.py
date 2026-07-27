import multiprocessing as mp
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


os.environ['PYTHONHASHSEED'] = '0'

class Worker_paral_train(mp.Process):
    def __init__(self, input_queue, output_queue, train, use_epre_dsac = False):
        mp.Process.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.train = train
        self.state = None
        self.total_reward = 0
        self.use_epre_dsac = use_epre_dsac
        if self.use_epre_dsac:
            if agent_par['new_best']:
                self.best_time = 1
                self.target_best_time = 1
            if agent_par['reset_model']:
                self.last_ave_reward = 0
        self.info_list = []
        self.train_time = 0

        print("Worker_train_paral __init__")

    def run(self):
        # save path
        localtime = time.strftime("%Y-%m-%d_%H%M", time.localtime())
        if agent_par['two_agent']:
            ModelPath_intersection = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/model_intersection/'
            ModelPath_temp_intersection = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/model_temp_intersection/'
            logpath_intersection = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/logtrain_intersection/'
            ModelPath_straight = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/model_straight/'
            ModelPath_temp_straight = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/model_temp_straight/'
            logpath_straight = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/logtrain_straight/'
            self.ModelPath_temp_intersection = ModelPath_temp_intersection
            self.ModelPath_temp_straight = ModelPath_temp_straight
        else:
            ModelPath = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/model/'
            ModelPath_temp = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/model_temp/'
            logpath = os.path.dirname(os.path.realpath(
                __file__)) + '/logs/' + str(localtime) + '/logtrain/'
            self.ModelPath_temp = ModelPath_temp
            
        self.q1_name = 'q1_{}_{}_{}.pth'.format(Config.reply_buffer_size, Config.total_episode, Config.lr)
        self.q2_name = 'q2_{}_{}_{}.pth'.format(Config.reply_buffer_size, Config.total_episode, Config.lr)
        self.policy_name = 'policy_{}_{}_{}.pth'.format(Config.reply_buffer_size, Config.total_episode, Config.lr)
        self.h_name = 'h_{}_{}_{}.pth'.format(Config.reply_buffer_size, Config.total_episode, Config.lr)
        self.alpha_name = 'alpha_{}_{}_{}.pth'.format(Config.reply_buffer_size, Config.total_episode, 0.003)

        if agent_par['continue_train']:
            if agent_par['two_agent']:
                self.folder_path_intersection = agent_par['train_data']['folder_path_intersection']
                self.folder_path_straight = agent_par['train_data']['folder_path_straight']
            else:
                self.folder_path = agent_par['train_data']['folder_path']
        
        if self.use_epre_dsac:
            if agent_par['two_agent']:
                self.agent_straight = Epre_dsac_agent(ModelPath=ModelPath_straight, ModelPath_temp=ModelPath_temp_straight, hrl = True)
                self.replay_buffer_straight = epre_replay_buffer(Config.reply_buffer_size, fdpi_enabled=False)
                self.agent_intersection = Epre_dsac_agent(ModelPath=ModelPath_intersection, ModelPath_temp=ModelPath_temp_intersection, hrl = True)
                self.replay_buffer_intersection = epre_replay_buffer(Config.reply_buffer_size, fdpi_enabled=False)
            else:
                agent_class = (
                    EpreDSACFDPIAgent
                    if agent_par.get("fdpi_enabled", False)
                    else Epre_dsac_agent
                )
                self.agent = agent_class(ModelPath=ModelPath, ModelPath_temp=ModelPath_temp, hrl = True)
                self.replay_buffer = epre_replay_buffer(Config.reply_buffer_size)
                
        else:
            self.agent =  Agent(ModelPath,ModelPath_temp, self.folder_path)
            self.replay_buffer = Reply_Buffer(Config.reply_buffer_size)
        if agent_par['two_agent']:
            self.logger_intersection = logger(path=logpath_intersection)
            self.logger_straight = logger(path=logpath_straight)
        else:
            self.logger = logger(path=logpath)
        self.step = 0
        self.episode = 0
        self.global_step = 0
        self.update_time = 0

        self.totalspeed = deque(maxlen = 3000)
        self.average_episode_reward = deque(maxlen=100)
        self.average_episode_reward_net = deque(maxlen=800)

        self.average_goal = deque(maxlen=100)
        self.average_collision = deque(maxlen=100)
        

        if agent_par['continue_train']:
            self.load_continue_net()
            self.episode = agent_par['train_data']['episode']
            self.update_time = agent_par['train_data']['update_time']

        print("Worker_train_paral run")
        
        while True:
            while True:
                try:
                    k = self.input_queue.get(timeout=0.001)
                    print(k[0])
                    self.info_list.append(k)
                    break
                except:
                    if self.train_time <= 500:
                        try:
                            self.update_model()
                        except Exception:
                            print("[worker-train][ERROR] update_model failed", flush=True)
                            print(traceback.format_exc(), flush=True)
                            self.train_time = 501
                    else:
                        time.sleep(0.5)
                    break
            if len(self.info_list) > 0:
                k = self.info_list[-1]
                self.info_list = []
                if self.train:
                    print(k[0])
                    if k[0] == 'end':
                        request_id = k[5] if len(k) > 5 else None
                        print(
                            f"[finish-debug][worker-train] received end "
                            f"batch={len(k[1]) if k[1] is not None else None} "
                            f"batch_straight={len(k[2]) if k[2] is not None else None} "
                            f"done_type={k[3]} real={k[4]} request_id={request_id}",
                            flush=True,
                        )
                        try:
                            models = self.send_model(k[1], k[2], k[3], k[4])
                            self.output_queue.put(['send_model success!', *models, request_id])
                            print('send_model success!')
                        except Exception:
                            error = traceback.format_exc()
                            print("[finish-debug][worker-train][ERROR] send_model failed", flush=True)
                            print(error, flush=True)
                            self.output_queue.put(['send_model failed!', error, None, None, None, None, request_id])

    # def add_buffer(self, batch):
    #     self.replay_buffer.append(batch)
    
    def update_model(self):
        if agent_par['two_agent']:
            train_ok = len(self.replay_buffer_intersection.buffer)-5 > Config.batch_size + 1 and len(self.replay_buffer_straight.buffer)-5 > Config.batch_size + 1 and self.train
        else:
            train_ok = len(self.replay_buffer.buffer)-5 > Config.batch_size + 1 and self.train
        if train_ok:
            time1 = time.time()
            self.train_dsac()
            self.update_time += 1
            # print('update', time.time()-time1)
            self.train_time += 1
            if agent_par['train'] and agent_par['reset'] and self.update_time%2e5 == 0: #重新初始化部分网络
                print('reeeeeeeset')
                if agent_par['two_agent']:
                    self.agent_intersection.policy.reset()
                    self.agent_intersection.policy_target.reset()
                    self.agent_straight.policy.reset()
                    self.agent_straight.policy_target.reset()
                else:
                    self.agent.policy.reset()
                    self.agent.policy_target.reset()

        if self.update_time % 2000 == 0 and self.update_time != 0:
            if agent_par['reset_model']:
                if self.last_ave_reward != 0:
                    if np.mean(self.average_episode_reward_net) >= self.last_ave_reward:
                        self.last_ave_reward = np.mean(self.average_episode_reward_net)
                        self.last_h_rl = self.agent.h_optimizer.param_groups[0]['lr']
                        print('h_rl',self.last_h_rl)
                        if os.path.exists(self.agent.ModelPath_temp):
                            self.del_files(self.agent.ModelPath_temp)
                        # agent.lr_step()
                        name = '{}_{}_{}'.format(Config.reply_buffer_size, self.episode, self.update_time)
                        self.agent.save_tep(name=self.agent.ModelPath_temp+'{}_'.format(self.episode)+name)
                        self.epre_save_buffer(name=self.agent.ModelPath_temp+'{}_'.format(self.episode)+name)
                        self.agent.save_model(name=self.agent.ModelPath+'{}_'.format(self.episode)+name)
                        if agent_par['new_best']:
                            self.best_time = 1
                            self.target_best_time = 1
                    elif agent_par['new_best']:
                        if self.best_time >= self.target_best_time:
                            self.agent.continue_train_model(self.agent.ModelPath_temp, best_h_rl=self.last_h_rl)
                            self.target_best_time*=2
                            self.best_time = 1
                            print('重新加载模型save', self.target_best_time)
                        else:
                            self.best_time += 1
                            print('继续迭代模型save', self.target_best_time, self.best_time)
                    else:
                        self.agent.continue_train_model(self.agent.ModelPath_temp, best_h_rl=self.last_h_rl)
                        
                else:
                    self.last_ave_reward = np.mean(self.average_episode_reward_net)
                    self.last_h_rl = self.agent.h_optimizer.param_groups[0]['lr']
                    print('h_rl',self.last_h_rl)
                    if os.path.exists(self.agent.ModelPath_temp):
                        self.del_files(self.agent.ModelPath_temp)
                    # agent.lr_step()
                    name = '{}_{}_{}'.format(Config.reply_buffer_size, self.episode, self.update_time)
                    self.agent.save_tep(name=self.agent.ModelPath_temp+'{}_'.format(self.episode)+name)
                    self.epre_save_buffer(name=self.agent.ModelPath_temp+'{}_'.format(self.episode)+name)
                    self.agent.save_model(name=self.agent.ModelPath+'{}_'.format(self.episode)+name)
            else:
                time1 = time.time()
                if agent_par['two_agent']:
                    if os.path.exists(self.agent_intersection.ModelPath_temp):
                        self.del_files(self.agent_intersection.ModelPath_temp)
                    # agent.lr_step()
                    name = '{}_{}_{}'.format(Config.reply_buffer_size, self.episode, self.update_time)
                    self.agent_intersection.save_tep(name=self.agent_intersection.ModelPath_temp+'{}_'.format(self.episode)+name)
                    self.epre_save_buffer(name=self.agent_intersection.ModelPath_temp+'{}_'.format(self.episode)+name, is_intersection=True)
                    self.agent_intersection.save_model(name=self.agent_intersection.ModelPath+'{}_'.format(self.episode)+name)

                    if os.path.exists(self.agent_straight.ModelPath_temp):
                        self.del_files(self.agent_straight.ModelPath_temp)
                    # agent.lr_step()
                    name = '{}_{}_{}'.format(Config.reply_buffer_size, self.episode, self.update_time)
                    self.agent_straight.save_tep(name=self.agent_straight.ModelPath_temp+'{}_'.format(self.episode)+name)
                    self.epre_save_buffer(name=self.agent_straight.ModelPath_temp+'{}_'.format(self.episode)+name, is_intersection=False)
                    self.agent_straight.save_model(name=self.agent_straight.ModelPath+'{}_'.format(self.episode)+name)
                else:
                    if os.path.exists(self.agent.ModelPath_temp):
                        self.del_files(self.agent.ModelPath_temp)
                    # agent.lr_step()
                    name = '{}_{}_{}'.format(Config.reply_buffer_size, self.episode, self.update_time)
                    self.agent.save_tep(name=self.agent.ModelPath_temp+'{}_'.format(self.episode)+name)
                    self.epre_save_buffer(name=self.agent.ModelPath_temp+'{}_'.format(self.episode)+name)
                    self.agent.save_model(name=self.agent.ModelPath+'{}_'.format(self.episode)+name)
                print('模型保存时间',time.time()-time1)

        
    def send_model(self, batchs, batchs_straight, done_type, real):
        # int_list = []
        # for i in batchs:
        #     int_list.append([i[2], i[4]])
        # str_list = []
        # for i in batchs_straight:
        #     str_list.append([i[2], i[4]])
        # print(int_list)
        # print(11111)
        # print(str_list)
        keep_fdpi_episode = agent_par.get("fdpi_enabled", False) and real
        if keep_fdpi_episode or (done_type!='termination' and done_type != 'out' and real):
            if agent_par['two_agent']:
                for batch in batchs:
                    self.replay_buffer_intersection.append(batch)
                if batchs_straight is not None:
                    for batch in batchs_straight:
                        self.replay_buffer_straight.append(batch)
                
                print('buffer', len(self.replay_buffer_intersection.buffer), len(self.replay_buffer_straight.buffer),  self.update_time)
            else:
                for batch in batchs:
                    self.replay_buffer.append(batch)  
                print('buffer', len(self.replay_buffer.buffer), self.update_time)
                    
            self.episode += 1
            if not agent_par['two_agent'] and agent_par.get("fdpi_enabled", False):
                self.agent.episode = self.episode
        # print(reward_list, reward_total)
        if agent_par['two_agent']:
            h_model = self.agent_intersection.h.state_dict()
            policy_model = self.agent_intersection.policy.state_dict()
            h_model_straight = self.agent_straight.h.state_dict()
            policy_model_straight = self.agent_straight.policy.state_dict()
            self.train_time = 0
        else:
            h_model = self.agent.h.state_dict()
            policy_model = self.agent.main_policy.state_dict()
            # q1_model = self.agent.q1.state_dict()
            # q2_model = self.agent.q2.state_dict()
            # h_target_model = self.agent.h_target.state_dict()
            # policy_target_model = self.agent.policy_target.state_dict()
            # q1_target_model = self.agent.q1_target.state_dict()
            # q2_target_model = self.agent.q2_target.state_dict()
            self.train_time = 0
            if agent_par.get("fdpi_enabled", False):
                return (
                    h_model,
                    policy_model,
                    self.agent.dual_policy.state_dict(),
                    self.agent.dual_active,
                    self.agent.mean_feasible_ratio,
                )
            h_model_straight = policy_model_straight = None
            
        return h_model, policy_model, h_model_straight, policy_model_straight




    def train_dsac(self, ):
        if self.use_epre_dsac:
            if agent_par['two_agent']:
                if self.update_time %1000 == 0:
                    print('hrlllll',self.agent_intersection.h_optimizer.param_groups[0]['lr'],self.agent_straight.h_optimizer.param_groups[0]['lr'])
                if len(self.replay_buffer_intersection.buffer)-5 > Config.batch_size:
                    batch_state, batch_action, batch_reward, batch_state_new, batch_done, batch_logp, batch_env_input, batch_next_env_input, batch_env_map, batch_next_env_map = self.replay_buffer_intersection.sample(Config.batch_size)
                    self.agent_intersection.update_Q_network(batch_state, batch_action, batch_reward, batch_state_new, batch_done, batch_logp, self.update_time, self.logger_intersection, batch_env_input, batch_next_env_input, batch_env_map, batch_next_env_map)
                    if self.agent_intersection.hrl and self.agent_intersection.h_optimizer.param_groups[0]['lr'] > 1e-6:
                        self.agent_intersection.h_scheduler.step()
                    else:
                        self.agent_intersection.hrl = False
                if len(self.replay_buffer_straight.buffer)-5 > Config.batch_size:
                    batch_state, batch_action, batch_reward, batch_state_new, batch_done, batch_logp, batch_env_input, batch_next_env_input, batch_env_map, batch_next_env_map = self.replay_buffer_straight.sample(Config.batch_size)
                    self.agent_straight.update_Q_network(batch_state, batch_action, batch_reward, batch_state_new, batch_done, batch_logp, self.update_time, self.logger_straight, batch_env_input, batch_next_env_input, batch_env_map, batch_next_env_map)
                    if self.agent_straight.hrl and self.agent_straight.h_optimizer.param_groups[0]['lr'] > 1e-6:
                        self.agent_straight.h_scheduler.step()
                    else:
                        self.agent_straight.hrl = False

            else:
                if self.update_time %1000 == 0:
                    print('hrlllll',self.agent.h_optimizer.param_groups[0]['lr'])
                if len(self.replay_buffer.buffer)-5 > Config.batch_size:
                    batch = self.replay_buffer.sample(Config.batch_size)
                    if agent_par.get("fdpi_enabled", False):
                        self.agent.update(batch, self.update_time, self.logger)
                    else:
                        batch_state, batch_action, batch_reward, batch_state_new, batch_done, batch_logp, batch_env_input, batch_next_env_input, batch_env_map, batch_next_env_map = batch
                        self.agent.update_Q_network(batch_state, batch_action, batch_reward, batch_state_new, batch_done, batch_logp, self.update_time, self.logger, batch_env_input, batch_next_env_input, batch_env_map, batch_next_env_map)
                    if self.agent.hrl and self.agent.h_optimizer.param_groups[0]['lr'] > 1e-6:
                        self.agent.h_scheduler.step()
                    else:
                        self.agent.hrl = False
        else:
            batch_state, batch_action, batch_reward, batch_state_new, batch_done,  batch_logp = self.replay_buffer.sample(Config.batch_size)
            # update policy network
            self.agent.update_Q_network(batch_state, batch_action, batch_reward, batch_state_new, batch_done, batch_logp, self.update_time, self.logger)
            #agent.soft_update_target_network(Config.tau)


    def epre_save_buffer(self, name, is_intersection = False):
        if agent_par['two_agent']:
            if is_intersection:
                if not os.path.exists(self.ModelPath_temp_intersection):
                    os.makedirs(self.ModelPath_temp_intersection)
                if agent_par['two_buffer']:
                    torch.save(self.collision_memory, name + "_collision_buffer_" + str(0) + ".pth")
                    torch.save(self.other_memory, name + "_other_buffer_" + str(0) + ".pth")
                else:
                    torch.save(self.replay_buffer_intersection, name + "_buffer_" + str(0) + ".pth")
            else:
                if not os.path.exists(self.ModelPath_temp_straight):
                    os.makedirs(self.ModelPath_temp_straight)
                if agent_par['two_buffer']:
                    torch.save(self.collision_memory, name + "_collision_buffer_" + str(0) + ".pth")
                    torch.save(self.other_memory, name + "_other_buffer_" + str(0) + ".pth")
                else:
                    torch.save(self.replay_buffer_straight, name + "_buffer_" + str(0) + ".pth")

        else:
            if not os.path.exists(self.ModelPath_temp):
                os.makedirs(self.ModelPath_temp)
            if agent_par['two_buffer']:
                torch.save(self.collision_memory, name + "_collision_buffer_" + str(0) + ".pth")
                torch.save(self.other_memory, name + "_other_buffer_" + str(0) + ".pth")
            else:
                torch.save(self.replay_buffer, name + "_buffer_" + str(0) + ".pth")
    
    # def save_buffer(self, name):
    #     if not os.path.exists(self.ModelPath_temp):
    #         os.makedirs(self.ModelPath_temp)
    #     torch.save(self.replay_buffer, name + "_buffer_" + str(0) + ".pth")
    
    def del_files(self,path_file):
        ls = os.listdir(path_file)
        for i in ls:
            f_path = os.path.join(path_file, i)
            # 判断是否是一个目录,若是,则递归删除
            if os.path.isdir(f_path):
                self.del_files(f_path)
            else:
                os.remove(f_path)
    
    def continue_train_buffer(self,folder_path, is_intersection = False):
        if os.path.exists(str(folder_path)):
            print('okokokookok')
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
                    if agent_par['two_agent']:
                        if is_intersection:
                            self.replay_buffer_intersection = torch.load(str(file_paths[i]))
                            print('successfully load buffer_intersection')
                        else:
                            self.replay_buffer_straight = torch.load(str(file_paths[i]))
                            print('successfully load buffer_straight')

                    else:
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
