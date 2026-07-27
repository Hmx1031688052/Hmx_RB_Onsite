import torch
from torch.nn.functional import mse_loss
from torch.autograd import Variable
import torch.optim as optim
import random
import glob
import os
from config import Config
from epre_dsac.Epre_dsac_model import QNet, PolicyNet, HNet
import numpy as np
from epre_dsac.epre_reply_buffer import Reply_Buffer
from torch.optim import lr_scheduler
from torch.distributions import Normal
from copy import deepcopy
from utilsa.act_distribution_cls import TanhGaussDistribution, GaussDistribution
from epre_dsac.parameters import agent_par

class Epre_dsac_agent:
    def __init__(self, ModelPath, ModelPath_temp, hrl=False, h_rl=1e-5, state_encoder="stt"):
        self.action_number = int(agent_par.get("action_dim", 2))
        self.action_low_limit = agent_par.get("action_low_limit", [0.0, -5.0])
        self.action_high_limit = agent_par.get("action_high_limit", [8.0, 5.0])
        self.ModelPath_temp = ModelPath_temp
        self.epsilon = Config.initial_epsilon
        self.hrl = hrl

        self. set_state_size = 36
        self.state_size = 39
        self.state_encoder = str(state_encoder).lower()
        if self.state_encoder not in ("stt", "frenet"):
            raise ValueError("state_encoder must be 'stt' or 'frenet'")
        # STT uses seven 128-D actor embeddings. Plain FDPI uses only the
        # explicitly constructed Frenet state and bypasses HNet.
        self.q_state_size = (
            128 * 7
            if self.state_encoder == "stt"
            else int(agent_par.get("frenet_state_dim", 24))
        )
        self.h_rl = h_rl

        self.device = torch.device(agent_par.get("rl_device", "cpu"))
        self.build_network()
        if agent_par['two_buffer']:
            self.collision_memory = Reply_Buffer(Config.reply_buffer_size)
            self.other_memory = Reply_Buffer(Config.reply_buffer_size)
        else:
            self.memory = Reply_Buffer(Config.reply_buffer_size)
            


        self.ModelPath = ModelPath
        self.epsilon_min = Config.min_epsilon
        self.epsilon_decay = (self.epsilon-self.epsilon_min)/Config.epsilon_decay2
        self.lr = Config.lr
        self.tau = Config.tau
        self.mean_std1 = -1.0
        self.mean_std2 = -1.0
        self.gamma = Config.discount_factor
        self.auto_alpha = True
        self.alpha = 0.2
        self.target_entropy = -self.action_number
        self.delay_update = 2
        self.tau_b = self.tau


        # self.folder_path = "C:/Users/24957/Desktop/epre-dsac-wwr-mlp-hrl-reset/epre_model_temp"  # 文件夹路径
        # if mode == 'REPLAY':
        #     self.folder_path = "C:/Users/24957/Desktop/epre-dsac-wwr-mlp-hrl-reset/epre_model_temp"
        # if mode == 'FRAGMENT':
        #     self.folder_path = "kkkplanner/CVaR/logs/2024-04-06_2119/model_temp"
        # if mode == 'SERIAL':
        #     self.folder_path = "kkkplanner/CVaR/logs/2024-04-06_2119/model_temp"
            
        # self.continue_train_buffer(self.folder_path)
        # if agent_par['train']:
        #     self.continue_train_model(self.folder_path)
        # else:
        #     self.continue_test_model(self.folder_path,4)


    def build_network(self):
        self.q1 = QNet(self.q_state_size, self.action_number).to(self.device)
        self.q2 = QNet(self.q_state_size, self.action_number).to(self.device)
        self.h = HNet(self.set_state_size).to(self.device)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        self.h_target = deepcopy(self.h)
        self.main_policy = PolicyNet(self.q_state_size, self.action_number).to(self.device)
        self.main_policy_target = deepcopy(self.main_policy)
        # Backward-compatible names used by legacy checkpoints and callers.
        self.policy = self.main_policy
        self.policy_target = self.main_policy_target
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=self.device)
        )

        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=Config.lr,weight_decay=1e-4)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=Config.lr,weight_decay=1e-4)

        if self.hrl:
            self.h_optimizer = optim.Adam(self.h.parameters(), lr=self.h_rl,weight_decay=1e-4)
            self.h_scheduler = optim.lr_scheduler.StepLR(self.h_optimizer, step_size=50000, gamma=0.8)
        else:
            self.h_optimizer = optim.Adam(self.h.parameters(), lr=Config.lr,weight_decay=1e-4)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=Config.lr,weight_decay=1e-4)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=0.0003,weight_decay=1e-4)
        # self.q1_scheduler = lr_scheduler.ExponentialLR(self.q1_optimizer, gamma=0.94)
        # self.q2_scheduler = lr_scheduler.ExponentialLR(self.q2_optimizer, gamma=0.94)

    def soft_update_target_network(self, tau):
        for target_param, local_param in zip(self.target_network.parameters(), self.Q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        # copy current_network to target network
        # self.target_network.load_state_dict(self.Q_network.state_dict())
    def hard_update_target_network(self, tau):
        for target_param, local_param in zip(self.target_network.parameters(), self.Q_network.parameters()):
            target_param.data.copy_(local_param.data)
    
    def update_Q_network(self, state, action, reward, state_new, terminal, logp, iteration, logger, env_input, next_env_input, env_map, next_env_map):
        # print(111,action)
        # print(222,reward)
        # print(333,terminal)
        # print(444,iteration)
        state = torch.from_numpy(state)
        action = torch.from_numpy(action)
        state_new = torch.from_numpy(state_new)
        terminal = torch.from_numpy(terminal)
        reward = torch.from_numpy(reward)
        env_input = torch.from_numpy(env_input)
        next_env_input = torch.from_numpy(next_env_input)
        env_map = torch.from_numpy(env_map)
        next_env_map = torch.from_numpy(next_env_map)
        state = Variable(state).to(self.device).float()
        action = Variable(action).to(self.device).float()
        state_new = Variable(state_new).to(self.device).float()
        terminal = Variable(terminal).to(self.device).float()
        reward = Variable(reward).to(self.device).float()
        env_input = Variable(env_input).to(self.device).float()
        next_env_input = Variable(next_env_input).to(self.device).float()
        env_map = Variable(env_map).to(self.device).float()
        next_env_map = Variable(next_env_map).to(self.device).float()
        # self.q1.eval()
        # self.q2.eval()
        # self.q1_target.eval()
        # self.q2_target.eval()
        # self.policy.eval()
        # self.policy_target.eval()
        
        # use current network to evaluate action argmax_a' Q_current(s', a')_

        uesc = self.use_h(state, env_input, env_map)
        uesc_new = self.use_h(state_new, next_env_input, next_env_map)
        uesc_target = self.use_target_h(state, env_input, env_map)
        uesc_new_target = self.use_target_h(state_new, next_env_input, next_env_map)
        logits = self.policy(uesc)
        logits_mean, logits_std = torch.chunk(logits, chunks=2, dim=-1)
        policy_mean = torch.tanh(logits_mean).mean().item()
        policy_std = logits_std.mean().item()

        act_dist = TanhGaussDistribution(
            logits,
            act_low_lim=self.action_low_limit,
            act_high_lim=self.action_high_limit,
        )
        new_act, new_log_prob = act_dist.rsample()
        new_log_prob = new_log_prob.to(self.device)
        new_act = new_act.to(self.device)
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.h_optimizer.zero_grad()
        loss_q = self.__compute_loss_q(uesc, action, reward, uesc_new, terminal, uesc_target, uesc_new_target)
        loss_q.backward(retain_graph=True)

        for p in self.q1.parameters():
            p.requires_grad = False
        for p in self.q2.parameters():
            p.requires_grad = False
        for p in self.h.parameters():
            p.requires_grad = False

        self.policy_optimizer.zero_grad()
        loss_policy = self.__compute_loss_policy(uesc, new_act, new_log_prob, uesc_target)  # policy函数的loss和熵
        loss_policy.backward()

        for p in self.q1.parameters():
            p.requires_grad = True
        for p in self.q2.parameters():
            p.requires_grad = True
        for p in self.h.parameters():
            p.requires_grad = True

        if self.auto_alpha:
            self.alpha_optimizer.zero_grad()
            loss_alpha = self.__compute_loss_alpha(new_log_prob)
            loss_alpha.backward()

        self.q1_optimizer.step()
        self.q2_optimizer.step()
        self.h_optimizer.step()

        if iteration % self.delay_update == 0:
            self.policy_optimizer.step()  # 延迟更新策略网络

            if self.auto_alpha:
                self.alpha_optimizer.step()

            with torch.no_grad():
                polyak = 1 - self.tau
                for p, p_targ in zip(
                        self.q1.parameters(), self.q1_target.parameters()
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak)*p.data)
                for p, p_targ in zip(
                        self.q2.parameters(), self.q2_target.parameters()
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak)*p.data)
                for p, p_targ in zip(
                        self.policy.parameters(),
                        self.policy_target.parameters(),
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak)*p.data)
                for p, p_targ in zip(
                        self.h.parameters(),self.h_target.parameters()
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak)*p.data)
        logger.add(iteration, q_loss=loss_q, policy_loss=loss_policy, new_log_prob = -new_log_prob.mean(), alpha = self.__get_alpha())

    def use_h(self, state, env_input, env_map, train=True):
        else_state = state[:, :11]
        set_state = state[:, 11:]
        self.h.eval()
        logits = self.h(env_input, env_map)
        #uesc = torch.cat((else_state, logits), dim=1)
        uesc = logits
        return uesc

    def use_frenet_state(self, state):
        start = int(agent_par.get("frenet_state_start", 7))
        size = int(agent_par.get("frenet_state_dim", 24))
        if state.shape[-1] < start + size:
            raise RuntimeError(
                "Frenet state requires input dim >= {}, got {}".format(
                    start + size, state.shape[-1]
                )
            )
        return state[..., start:start + size]

    def encode_policy_state(self, state, env_input=None, env_map=None, target=False):
        """Select state representation independently from the RL algorithm."""
        if self.state_encoder == "frenet":
            return self.use_frenet_state(state)
        if env_input is None or env_map is None:
            raise ValueError("STT state encoding requires env_input and env_map")
        if target:
            return self.use_target_h(state, env_input, env_map)
        return self.use_h(state, env_input, env_map)

    def use_target_h(self, state, env_input, env_map, train=True): #这个部分只要使用了stt都没有对原始的state进行h的状态表征
        else_state = state[:, :11]
        set_state = state[:, 11:]
        self.h_target.eval()
        logits = self.h_target(env_input, env_map)
        #uesc = torch.cat((else_state, logits), dim=1)
        uesc = logits
        return uesc


    def take_action(self, uesc, train=True):
        # print(111)
        # for name, param in self.h_target.named_parameters():
        #     if str(name) == 'agent_agent.interaction_1.ffn.0.weight':
        #         print(f"参数名: {name}")
        #         print(f"参数形状: {param.shape}")
        #         print(f"参数值（部分）:\n{param.data}\n")
        # print(222)
        # for name, param in self.h.named_parameters():
        #     if str(name) == 'agent_agent.interaction_1.ffn.0.weight':
        #         print(f"参数名: {name}")
        #         print(f"参数形状: {param.shape}")
        #         print(f"参数值（部分）:\n{param.data}\n")
        # print(333)

        self.policy.eval()
        with torch.no_grad():
            logits = self.policy(uesc)
            action_distribution = TanhGaussDistribution(
                logits,
                act_low_lim=self.action_low_limit,
                act_high_lim=self.action_high_limit,
            )
            action, logp = action_distribution.sample(train=train)
        action = action.detach()[0].numpy()
        logp = logp.detach()[0].numpy()
        return action, logp

    def update_epsilon(self):
        if self.epsilon > Config.min_epsilon:
            self.epsilon = self.epsilon * Config.epsilon_decay
    
    def stop_epsilon(self):
        self.epsilon_tmp = self.epsilon        
        self.epsilon = 0        
    
    def restore_epsilon(self):
        self.epsilon = self.epsilon_tmp        
    
    # def save(self, step, logs_path):
    #     os.makedirs(logs_path, exist_ok=True)
    #     model_list =  glob.glob(os.path.join(logs_path, '*.pth'))
    #     if len(model_list) > Config.maximum_model - 1 :
    #         min_step = min([int(li.split('/')[-1][6:-4]) for li in model_list])
    #         os.remove(os.path.join(logs_path, 'model-{}.pth' .format(min_step)))
    #     logs_path = os.path.join(logs_path, 'model-{}.pth' .format(step))
    #     self.Q_network.save(logs_path, step=step, optimizer=self.optimizer)
    #     print('=> Save {}' .format(logs_path))

    # def save(self, q1_name, q2_name, policy_name, h_name, episode):
    #     if not os.path.exists(self.ModelPath):
    #         os.makedirs(self.ModelPath)
    #     q1_name = self.ModelPath + '{}_'.format(episode) + q1_name
    #     q2_name = self.ModelPath + '{}_'.format(episode) + q2_name
    #     policy_name = self.ModelPath + '{}_'.format(episode) + policy_name
    #     h_name = self.ModelPath + '{}_'.format(episode) + h_name
    #     torch.save(self.q1.state_dict(), q1_name)
    #     torch.save(self.q2.state_dict(), q2_name)
    #     torch.save(self.policy.state_dict(), policy_name)
    #     torch.save(self.h.state_dict(), h_name)

    def restore(self, policy_logs_path, q1_logs_path, q2_logs_path, h_logs_path):
        self.policy.load(policy_logs_path)
        self.policy_target.load(policy_logs_path)
        print('=> Restore {}' .format(policy_logs_path))
        self.q1.load(q1_logs_path)
        self.q1_target.load(q1_logs_path)
        print('=> Restore {}'.format(q1_logs_path))
        self.q2.load(q2_logs_path)
        self.q2_target.load(q2_logs_path)
        print('=> Restore {}'.format(q2_logs_path))
        self.h.load(h_logs_path)
        self.h_target.load(h_logs_path)
        print('=> Restore {}'.format(h_logs_path))

    def __compute_loss_q(self, uesc, action, reward, uesc_new, terminal, uesc_target, uesc_new_target):
        logits_2 = self.policy_target(uesc_new_target)
        act2_dist = TanhGaussDistribution(
            logits_2,
            act_low_lim=self.action_low_limit,
            act_high_lim=self.action_high_limit,
        )
        act2, log_prob_act2 = act2_dist.rsample()
        act2 = act2.to(self.device)
        log_prob_act2 = log_prob_act2.to(self.device)

        q1, q1_std, _ = self.__q_evaluate(uesc, action, self.q1)
        q2, q2_std, _ = self.__q_evaluate(uesc, action, self.q2)

        if self.mean_std1 == -1.0:
            self.mean_std1 = torch.mean(q1_std.detach())
        else:
            self.mean_std1 = (1 - self.tau_b)*self.mean_std1 + self.tau_b*torch.mean(q1_std.detach())

        if self.mean_std2 == -1.0:
            self.mean_std2 = torch.mean(q2_std.detach())
        else:
            self.mean_std2 = (1 - self.tau_b)*self.mean_std2 + self.tau_b*torch.mean(q2_std.detach())


        q1_next, _, q1_next_sample = self.__q_evaluate(
            uesc_new_target, act2, self.q1_target
        )

        q2_next, _, q2_next_sample = self.__q_evaluate(
            uesc_new_target, act2, self.q2_target
        )
        q_next = torch.min(q1_next, q2_next)  # 双值分配技巧
        q_next_sample = torch.where(q1_next < q2_next, q1_next_sample, q2_next_sample)

        target_q1, target_q1_bound = self.__compute_target_q(
            reward,
            terminal,
            q1.detach(),
            self.mean_std1.detach(),
            q_next.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )

        target_q2, target_q2_bound = self.__compute_target_q(
            reward,
            terminal,
            q2.detach(),
            self.mean_std2.detach(),
            q_next.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )

        q1_std_detach = torch.clamp(q1_std, min=0.).detach()
        q2_std_detach = torch.clamp(q2_std, min=0.).detach()
        bias = 0.1

        q1_loss = (torch.pow(self.mean_std1, 2) + bias)*torch.mean(
            -(target_q1 - q1).detach()/(torch.pow(q1_std_detach, 2) + bias)*q1
            - ((torch.pow(q1.detach() - target_q1_bound, 2) - q1_std_detach.pow(2))/(torch.pow(q1_std_detach, 3) + bias)
               )*q1_std
        )

        q2_loss = (torch.pow(self.mean_std2, 2) + bias)*torch.mean(
            -(target_q2 - q2).detach()/(torch.pow(q2_std_detach, 2) + bias)*q2
            - ((torch.pow(q2.detach() - target_q2_bound, 2) - q2_std_detach.pow(2))/(torch.pow(q2_std_detach, 3) + bias)
               )*q2_std
        )
        return q1_loss + q2_loss

    def __q_evaluate(self, obs, act, qnet):  # 状态和动作输入value函数得到价值分布 mean：均值 q_value：分布
        StochaQ = qnet(obs, act)
        mean, std = StochaQ[..., 0], StochaQ[..., -1]
        # std = log_std.exp()
        normal = Normal(torch.zeros_like(mean), torch.ones_like(std))
        z = normal.sample()
        z = torch.clamp(z, -3, 3)
        q_value = mean + torch.mul(z, std)
        return mean, std, q_value

    def __compute_target_q(self, r, done, q,q_std, q_next, q_next_sample, log_prob_a_next):

        target_q = r + (1 - done) * self.gamma * (
            q_next - self.__get_alpha() * log_prob_a_next
        ) # yq
        target_q_sample = r + (1 - done) * self.gamma * (
            q_next_sample - self.__get_alpha() * log_prob_a_next
        ) # yz
        td_bound = 3 * q_std # 分布剪裁（3sigma原则）
        difference = torch.clamp(target_q_sample - q, -td_bound, td_bound)
        target_q_bound = q + difference
        return target_q.detach(), target_q_bound.detach()

    def __get_alpha(self): # 温度参数alpha
        if self.auto_alpha:
            alpha = self.log_alpha.exp()
            return alpha.item()
        else:
            #print(self.alpha)
            return self.alpha

    def __compute_loss_policy(self, uesc, new_act, new_log_prob, uesc_target):
        q1, _, _ = self.__q_evaluate(uesc, new_act, self.q1)
        q2, _, _ = self.__q_evaluate(uesc, new_act, self.q2)
        loss_policy = (self.__get_alpha() * new_log_prob - torch.min(q1,q2)).mean() # policy函数的loss
        return loss_policy

    def __compute_loss_alpha(self, new_log_prob):
        loss_alpha = (
            -self.log_alpha
            * (new_log_prob.detach() + self.target_entropy).mean()
        )
        return loss_alpha
    


    
    # def save_tep(self, q1_name1, q2_name1, policy_name1, h_name1, alpha_name1, episode):
    #     if not os.path.exists(self.ModelPath_temp):
    #         os.makedirs(self.ModelPath_temp)
    #     q1_name = self.ModelPath_temp + '{}_'.format(episode) + q1_name1
    #     q2_name = self.ModelPath_temp + '{}_'.format(episode) + q2_name1
    #     policy_name = self.ModelPath_temp + '{}_'.format(episode) + policy_name1
    #     h_name = self.ModelPath_temp + '{}_'.format(episode) + h_name1
    #     alpha_name = self.ModelPath_temp + '{}_'.format(episode) + alpha_name1
    #     q1_target_name = self.ModelPath_temp + '{}_'.format(episode) + "target_" + q1_name1
    #     q2_target_name = self.ModelPath_temp + '{}_'.format(episode) + "target_" + q2_name1
    #     policy_target_name = self.ModelPath_temp + '{}_'.format(episode) + "target_" + policy_name1
    #     h_target_name = self.ModelPath_temp + '{}_'.format(episode) + "target_" + h_name1
    #     q1_optimizer_name = self.ModelPath_temp + '{}_'.format(episode) + "opt_" + q1_name1
    #     q2_optimizer_name = self.ModelPath_temp + '{}_'.format(episode) + "opt_" + q2_name1
    #     policy_optimizer_name = self.ModelPath_temp + '{}_'.format(episode) + "opt_" + policy_name1
    #     h_optimizer_name = self.ModelPath_temp + '{}_'.format(episode) + "opt_" + h_name1
    #     alpha_optimizer_name = self.ModelPath_temp + '{}_'.format(episode) + "opt_" + alpha_name1
    #     torch.save(self.q1.state_dict(), q1_name)
    #     torch.save(self.q2.state_dict(), q2_name)
    #     torch.save(self.policy.state_dict(), policy_name)
    #     torch.save(self.h.state_dict(), h_name)
    #     torch.save(self.log_alpha.data, alpha_name)

    #     torch.save(self.q1_target.state_dict(), q1_target_name)
    #     torch.save(self.q2_target.state_dict(), q2_target_name)
    #     torch.save(self.policy_target.state_dict(), policy_target_name)
    #     torch.save(self.h_target.state_dict(), h_target_name)

    #     torch.save(self.q1_optimizer.state_dict(), q1_optimizer_name)
    #     torch.save(self.q2_optimizer.state_dict(), q2_optimizer_name)
    #     torch.save(self.policy_optimizer.state_dict(), policy_optimizer_name)
    #     torch.save(self.h_optimizer.state_dict(), h_optimizer_name)
    #     torch.save(self.alpha_optimizer.state_dict(), alpha_optimizer_name)

    # def continue_train_model(self,folder_path):
    #     if os.path.exists(str(folder_path)):

    #         file_names = []  # 用于保存文件名
    #         file_paths = []  # 用于保存文件路径

    #         # 遍历文件夹中的每一个文件
    #         for root, dirs, files in os.walk(folder_path):
    #             for file in files:
    #                 file_name = file  # 文件名（包含后缀）
    #                 file_path = os.path.join(root, file)  # 文件绝对路径
    #                 file_names.append(file_name)
    #                 file_paths.append(file_path)
    #         for i in range(len(file_names)):
    #             if str(file_names[i].split('_')[1]) == 'policy':
    #                 self.policy.load(file_paths[i])          
    #                 print('successfully load policy')
    #             if str(file_names[i].split('_')[1]) == 'q1':
    #                 self.q1.load(file_paths[i])          
    #                 print('successfully load q1')
    #             if str(file_names[i].split('_')[1]) == 'q2':
    #                 self.q2.load(file_paths[i])          
    #                 print('successfully load q2')
    #             if str(file_names[i].split('_')[1]) == 'h':
    #                 self.h.load(file_paths[i])          
    #                 print('successfully load h')
    #             if str(file_names[i].split('_')[1]) == 'alpha': 
    #                 self.log_alpha.data.copy_(torch.load(file_paths[i]))      
    #                 print('successfully load alpha')

    #             if str(file_names[i].split('_')[1]) == 'target':
    #                 if str(file_names[i].split('_')[2]) == 'policy':
    #                     self.policy_target.load(file_paths[i])          
    #                     print('successfully load policy_target')
    #                 if str(file_names[i].split('_')[2]) == 'q1':
    #                     self.q1_target.load(file_paths[i])          
    #                     print('successfully load q1_target')
    #                 if str(file_names[i].split('_')[2]) == 'q2':
    #                     self.q2_target.load(file_paths[i])          
    #                     print('successfully load q2_target')
    #                 if str(file_names[i].split('_')[2]) == 'h':
    #                     self.h_target.load(file_paths[i])          
    #                     print('successfully load h_target')

    #         for i in range(len(file_names)):
    #             if str(file_names[i].split('_')[1]) == 'opt':
    #                 if str(file_names[i].split('_')[2]) == 'policy':
    #                     self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=Config.lr,weight_decay=1e-4)
    #                     self.policy_optimizer.load_state_dict(torch.load(str(file_paths[i])))
    #                     print('successfully load policy_optimizer')
    #                 if str(file_names[i].split('_')[2]) == 'q1':
    #                     self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=Config.lr,weight_decay=1e-4)
    #                     self.q1_optimizer.load_state_dict(torch.load(str(file_paths[i])))        
    #                     print('successfully load q1_optimizer')
    #                 if str(file_names[i].split('_')[2]) == 'q2':
    #                     self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=Config.lr,weight_decay=1e-4)
    #                     self.q2_optimizer.load_state_dict(torch.load(str(file_paths[i])))       
    #                     print('successfully load q2_optimizer')
    #                 if str(file_names[i].split('_')[2]) == 'h':
    #                     self.h_optimizer = optim.Adam(self.h.parameters(), lr=Config.lr,weight_decay=1e-4)
    #                     self.h_optimizer.load_state_dict(torch.load(str(file_paths[i])))     
    #                     print('successfully load h_optimizer')
    #                 if str(file_names[i].split('_')[2]) == 'alpha':
    #                     self.alpha_optimizer = optim.Adam([self.log_alpha], lr=0.0003,weight_decay=1e-4)
    #                     self.alpha_optimizer.load_state_dict(torch.load(str(file_paths[i])))   
    #                     print('successfully load alpha_optimizer')

    #         print("continue train")
    #     else:
    #         print('new train')

    def save_buffer(self, name):
        if not os.path.exists(self.ModelPath_temp):
            os.makedirs(self.ModelPath_temp)
        if agent_par['two_buffer']:
            torch.save(self.collision_memory, name + "_collision_buffer_" + str(0) + ".pth")
            torch.save(self.other_memory, name + "_other_buffer_" + str(0) + ".pth")
        else:
            torch.save(self.memory, name + "_buffer_" + str(0) + ".pth")


    
    def save_tep(self, name):
        if not os.path.exists(self.ModelPath_temp):
            os.makedirs(self.ModelPath_temp)
        
        torch.save(self.q1.state_dict(), name + "_q1_local_" + str(0) + ".pth")
        torch.save(self.q2.state_dict(), name + "_q2_local_" + str(0) + ".pth")
        torch.save(self.policy.state_dict(), name + "_policy_local_" + str(0) + ".pth")
        torch.save(self.h.state_dict(), name + "_h_local_" + str(0) + ".pth")

        torch.save(self.q1_target.state_dict(), name + "_q1_target_" + str(0) + ".pth")
        torch.save(self.q2_target.state_dict(), name + "_q2_target_" + str(0) + ".pth")
        torch.save(self.policy_target.state_dict(), name + "_policy_target_" + str(0) + ".pth")
        torch.save(self.h_target.state_dict(), name + "_h_target_" + str(0) + ".pth")

        torch.save(self.q1_optimizer.state_dict(), name + "_" + str(self.lr) + "_q1_opt_" + str(0) + ".pth")
        torch.save(self.q2_optimizer.state_dict(), name + "_" + str(self.lr) + "_q2_opt_" + str(0) + ".pth")
        torch.save(self.policy_optimizer.state_dict(), name + "_" + str(self.lr) + "_policy_opt_" + str(0) + ".pth")
        torch.save(self.h_optimizer.state_dict(), name + "_" + str(self.lr) + "_h_opt_" + str(0) + ".pth")

        torch.save(self.log_alpha.data, name + "_log_alpha_" + str(0) + ".pth")
        torch.save(self.alpha_optimizer.state_dict(), name + "_" + str(0.0003) + "_alpha_opt_" + str(0) + ".pth")

        # torch.save(self.qnetwork_local.state_dict(), name + "_local_" + str(0) + ".pth")
        # torch.save(self.qnetwork_target.state_dict(), name + "_target_" + str(0) + ".pth")
        # torch.save(self.optimizer.state_dict(), name + "_" + str(self.lr) + "_opt_" + str(0) + ".pth")
    
    def save_model(self, name):
        if not os.path.exists(self.ModelPath):
            os.makedirs(self.ModelPath)
        torch.save(self.q1.state_dict(), name + "_q1_local_" + str(0) + ".pth")
        torch.save(self.q2.state_dict(), name + "_q2_local_" + str(0) + ".pth")
        torch.save(self.policy.state_dict(), name + "_policy_local_" + str(0) + ".pth")
        torch.save(self.h.state_dict(), name + "_h_local_" + str(0) + ".pth")
        torch.save(self.log_alpha.data, name + "_log_alpha_" + str(0) + ".pth")

        # torch.save(self.qnetwork_local.state_dict(), name + "_local_" + str(0) + ".pth")

    def continue_train_buffer(self,folder_path):
        print('bbbbbbbb')
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
                if agent_par['two_buffer']:
                    if str(file_names[i].split('_')[-2]) == 'buffer' and str(file_names[i].split('_')[-3]) == 'collision':
                        self.collision_memory = torch.load(str(file_paths[i]))
                        print('successfully load collision_buffer')
                    if str(file_names[i].split('_')[-2]) == 'buffer' and str(file_names[i].split('_')[-3]) == 'other':
                        self.other_memory = torch.load(str(file_paths[i]))
                        print('successfully load other_buffer')
                else:
                    if str(file_names[i].split('_')[-2]) == 'buffer':
                        self.memory = torch.load(str(file_paths[i]))
                        print('successfully load buffer')
        else:
            print('new buffer')
    
    def continue_train_model(self,folder_path, best_h_rl = None):
        print(11111111)
        if best_h_rl is not None:
            h_rl = best_h_rl
        else:
            h_rl = self.h_rl
        
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
                if str(file_names[i].split('_')[-2]) == 'opt' and str(file_names[i].split('_')[-3]) != 'alpha':
                    self.lr = float(file_names[i].split('_')[-4])
                    print(self.lr)
                    break
            # for i in range(len(file_names)):
            #     if str(file_names[i].split('_')[-2]) == 'opt' and str(file_names[i].split('_')[-3]) == 'alpha':
            #         self.lr_alpha = float(file_names[i].split('_')[-4])
            #         print(self.lr_alpha)
            #         break
            for i in range(len(file_names)):
                if str(file_names[i].split('_')[-2]) == 'local' and  str(file_names[i].split('_')[-3]) == 'q1':          
                    self.q1.load_state_dict(torch.load(str(file_paths[i]),map_location=str(self.device)))
                    print('successfully load q1_local ')
                if str(file_names[i].split('_')[-2]) == 'local' and  str(file_names[i].split('_')[-3]) == 'q2':          
                    self.q2.load_state_dict(torch.load(str(file_paths[i]),map_location=str(self.device)))
                    print('successfully load q2_local ')
                if str(file_names[i].split('_')[-2]) == 'local' and  str(file_names[i].split('_')[-3]) == 'policy':          
                    self.policy.load_state_dict(torch.load(str(file_paths[i]),map_location=str(self.device)))
                    print('successfully load policy_local ')
                if str(file_names[i].split('_')[-2]) == 'local' and  str(file_names[i].split('_')[-3]) == 'h':          
                    self.h.load_state_dict(torch.load(str(file_paths[i]),map_location=str(self.device)))
                    print('successfully load h_local ')
                
                if str(file_names[i].split('_')[-2]) == 'target' and  str(file_names[i].split('_')[-3]) == 'q1':          
                    self.q1_target.load_state_dict(torch.load(str(file_paths[i]),map_location=str(self.device)))
                    print('successfully load q1_target ')
                if str(file_names[i].split('_')[-2]) == 'target' and  str(file_names[i].split('_')[-3]) == 'q2':          
                    self.q2_target.load_state_dict(torch.load(str(file_paths[i]),map_location=str(self.device)))
                    print('successfully load q2_target  ')
                if str(file_names[i].split('_')[-2]) == 'target' and  str(file_names[i].split('_')[-3]) == 'policy':          
                    self.policy_target.load_state_dict(torch.load(str(file_paths[i]),map_location=str(self.device)))
                    print('successfully load policy_target  ')
                if str(file_names[i].split('_')[-2]) == 'target' and  str(file_names[i].split('_')[-3]) == 'h':          
                    self.h_target.load_state_dict(torch.load(str(file_paths[i]),map_location=str(self.device)))
                    print('successfully load h_target  ')
                if str(file_names[i].split('_')[-2]) == 'alpha' and  str(file_names[i].split('_')[-3]) == 'log':  
                    # self.log_alpha.data.copy_(torch.load(file_paths[i],map_location=str(self.device)))       
                    # self.log_alpha(torch.load(str(file_paths[i]),map_location=str(self.device)))
                    self.log_alpha.data = torch.load(str(file_paths[i]))
                    print('successfully load log_alpha  ')
            
                
            for i in range(len(file_names)):
                if str(file_names[i].split('_')[-2]) == 'opt' and str(file_names[i].split('_')[-3]) == 'q1':
                    self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.lr,weight_decay=1e-4)
                    self.q1_optimizer.load_state_dict(torch.load(str(file_paths[i])))
                    print('successfully load q1_opt ')
                if str(file_names[i].split('_')[-2]) == 'opt' and str(file_names[i].split('_')[-3]) == 'q2':
                    self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.lr,weight_decay=1e-4)
                    self.q2_optimizer.load_state_dict(torch.load(str(file_paths[i])))
                    print('successfully load q2_opt ')
                if str(file_names[i].split('_')[-2]) == 'opt' and str(file_names[i].split('_')[-3]) == 'h':
                    self.h_optimizer = optim.Adam(self.h.parameters(), lr=h_rl,weight_decay=1e-4)
                    self.h_optimizer.load_state_dict(torch.load(str(file_paths[i])))
                    print('successfully load h_opt ')
                if str(file_names[i].split('_')[-2]) == 'opt' and str(file_names[i].split('_')[-3]) == 'policy':
                    self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr,weight_decay=1e-4)
                    self.policy_optimizer.load_state_dict(torch.load(str(file_paths[i])))
                    print('successfully load policy_opt ')
                if str(file_names[i].split('_')[-2]) == 'opt' and str(file_names[i].split('_')[-3]) == 'alpha':
                    self.alpha_optimizer = optim.Adam([self.log_alpha], lr=0.0003,weight_decay=1e-4)
                    self.alpha_optimizer.load_state_dict(torch.load(str(file_paths[i])))
                    print('successfully load alpha_opt ')

            print("continue train")
        else:
            print('new train')
    
    def continue_test_model(self,folder_path,e):
        print(2222222222)
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
                if str(file_names[i].split('_')[-2]) == 'local' and  str(file_names[i].split('_')[-3]) == 'q1':# and str(file_names[i].split('_')[-4]) == str(e):          
                    self.q1.load_state_dict(torch.load(str(file_paths[i]),map_location=str(self.device)))
                    print('successfully load q1_local ')
                if str(file_names[i].split('_')[-2]) == 'local' and  str(file_names[i].split('_')[-3]) == 'q2':# and str(file_names[i].split('_')[-4]) == str(e):          
                    self.q2.load_state_dict(torch.load(str(file_paths[i]),map_location=str(self.device)))
                    print('successfully load q2_local ')
                if str(file_names[i].split('_')[-2]) == 'local' and  str(file_names[i].split('_')[-3]) == 'policy':# and str(file_names[i].split('_')[-4]) == str(e):          
                    self.policy.load_state_dict(torch.load(str(file_paths[i]),map_location=str(self.device)))
                    print('successfully load policy_local ')
                if str(file_names[i].split('_')[-2]) == 'local' and  str(file_names[i].split('_')[-3]) == 'h':# and str(file_names[i].split('_')[-4]) == str(e):          
                    self.h.load_state_dict(torch.load(str(file_paths[i]),map_location=str(self.device)))
                    print('successfully load h_local ')
        else:
            print('not successfully load local')


