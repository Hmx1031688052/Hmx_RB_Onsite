import torch
from torch.nn.functional import mse_loss
from torch.autograd import Variable
import torch.optim as optim
import random
import glob
import os
from config import Config
from model import QNet, PolicyNet, HNet
import numpy as np
from utilss import ReplayBuffer, PrioritizedReplayBuffer
from config import Config
from torch.optim import lr_scheduler
from torch.distributions import Normal
from copy import deepcopy
from utilsa.act_distribution_cls import TanhGaussDistribution, GaussDistribution
from epre_dsac.parameters import agent_par

class Agent:
    def __init__(self, ModelPath, ModelPath_temp, floder_path):
        self.device = torch.device(agent_par.get("rl_device", "cpu"))
        self.action_number = Config.action_dim
        self.action_low_limit = Config.action_low_limit
        self.action_high_limit = Config.action_high_limit
        self.epsilon = Config.initial_epsilon
        self. set_state_size = 42
        self.state_size = 53
        self.q_state_size = 32+11
        self.build_network()
        self.memory = ReplayBuffer(Config.reply_buffer_size)
        self.ModelPath = ModelPath
        self.ModelPath_temp = ModelPath_temp
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
        self.floder_path = floder_path
        

    def build_network(self):
        self.q1 = QNet(self.q_state_size, self.action_number).to(self.device)
        self.q2 = QNet(self.q_state_size, self.action_number).to(self.device)
        self.h = HNet(self.set_state_size).to(self.device)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        self.h_target = deepcopy(self.h)
        self.policy = PolicyNet(self.q_state_size, self.action_number).to(self.device)
        self.policy_target = deepcopy(self.policy)
        self.log_alpha = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))

        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=Config.lr,weight_decay=1e-4)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=Config.lr,weight_decay=1e-4)
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
    
    def update_Q_network(self, state, action, reward, state_new, terminal, logp, iteration, logger):
        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).float()
        state_new = torch.from_numpy(state_new).float()
        terminal = torch.from_numpy(terminal).float()
        reward = torch.from_numpy(reward).float()
        state = Variable(state).to(self.device)
        action = Variable(action).to(self.device)
        state_new = Variable(state_new).to(self.device)
        terminal = Variable(terminal).to(self.device)
        reward = Variable(reward).to(self.device)
        uesc = self.use_h(state)
        uesc_new = self.use_h(state_new)
        uesc_target = self.use_target_h(state)
        uesc_new_target = self.use_target_h(state_new)
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
        logger.add(iteration, q_loss=loss_q, policy_loss=loss_policy)


    def use_h(self, state, train=True):
        else_state = state[:, :11]
        set_state = state[:, 11:]
        
        self.h.eval()
        logits = self.h(set_state)
        uesc = torch.cat((else_state, logits), dim=1)
        return uesc

    def use_target_h(self, state, train=True):
        else_state = state[:, :11]
        set_state = state[:, 11:]
        self.h_target.eval()
        logits = self.h_target(set_state)
        uesc = torch.cat((else_state, logits), dim=1)
        return uesc

    def take_action(self, uesc, train=True):
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
    

    def save(self, q1_name, q2_name, policy_name, h_name, episode):
        if not os.path.exists(self.ModelPath):
            os.makedirs(self.ModelPath)
        q1_name = self.ModelPath + '{}_'.format(episode) + q1_name
        q2_name = self.ModelPath + '{}_'.format(episode) + q2_name
        policy_name = self.ModelPath + '{}_'.format(episode) + policy_name
        h_name = self.ModelPath + '{}_'.format(episode) + h_name
        torch.save(self.q1.state_dict(), q1_name)
        torch.save(self.q2.state_dict(), q2_name)
        torch.save(self.policy.state_dict(), policy_name)
        torch.save(self.h.state_dict(), h_name)

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
    
    def save_tep(self, q1_name1, q2_name1, policy_name1, h_name1, alpha_name1, episode):
        if not os.path.exists(self.ModelPath_temp):
            os.makedirs(self.ModelPath_temp)
        q1_name = self.ModelPath_temp + '{}_'.format(episode) + q1_name1
        q2_name = self.ModelPath_temp + '{}_'.format(episode) + q2_name1
        policy_name = self.ModelPath_temp + '{}_'.format(episode) + policy_name1
        h_name = self.ModelPath_temp + '{}_'.format(episode) + h_name1
        alpha_name = self.ModelPath_temp + '{}_'.format(episode) + alpha_name1
        q1_target_name = self.ModelPath_temp + '{}_'.format(episode) + "target_" + q1_name1
        q2_target_name = self.ModelPath_temp + '{}_'.format(episode) + "target_" + q2_name1
        policy_target_name = self.ModelPath_temp + '{}_'.format(episode) + "target_" + policy_name1
        h_target_name = self.ModelPath_temp + '{}_'.format(episode) + "target_" + h_name1
        q1_optimizer_name = self.ModelPath_temp + '{}_'.format(episode) + "opt_" + q1_name1
        q2_optimizer_name = self.ModelPath_temp + '{}_'.format(episode) + "opt_" + q2_name1
        policy_optimizer_name = self.ModelPath_temp + '{}_'.format(episode) + "opt_" + policy_name1
        h_optimizer_name = self.ModelPath_temp + '{}_'.format(episode) + "opt_" + h_name1
        alpha_optimizer_name = self.ModelPath_temp + '{}_'.format(episode) + "opt_" + alpha_name1
        torch.save(self.q1.state_dict(), q1_name)
        torch.save(self.q2.state_dict(), q2_name)
        torch.save(self.policy.state_dict(), policy_name)
        torch.save(self.h.state_dict(), h_name)
        torch.save(self.log_alpha.data, alpha_name)

        torch.save(self.q1_target.state_dict(), q1_target_name)
        torch.save(self.q2_target.state_dict(), q2_target_name)
        torch.save(self.policy_target.state_dict(), policy_target_name)
        torch.save(self.h_target.state_dict(), h_target_name)

        torch.save(self.q1_optimizer.state_dict(), q1_optimizer_name)
        torch.save(self.q2_optimizer.state_dict(), q2_optimizer_name)
        torch.save(self.policy_optimizer.state_dict(), policy_optimizer_name)
        torch.save(self.h_optimizer.state_dict(), h_optimizer_name)
        torch.save(self.alpha_optimizer.state_dict(), alpha_optimizer_name)

    def continue_train_model(self,folder_path):
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
                if str(file_names[i].split('_')[1]) == 'policy':
                    self.policy.load(file_paths[i])          
                    print('successfully load policy')
                if str(file_names[i].split('_')[1]) == 'q1':
                    self.q1.load(file_paths[i])          
                    print('successfully load q1')
                if str(file_names[i].split('_')[1]) == 'q2':
                    self.q2.load(file_paths[i])          
                    print('successfully load q2')
                if str(file_names[i].split('_')[1]) == 'h':
                    self.h.load(file_paths[i])          
                    print('successfully load h')
                if str(file_names[i].split('_')[1]) == 'alpha': 
                    self.log_alpha.data.copy_(torch.load(file_paths[i]))      
                    print('successfully load alpha')

                if str(file_names[i].split('_')[1]) == 'target':
                    if str(file_names[i].split('_')[2]) == 'policy':
                        self.policy_target.load(file_paths[i])          
                        print('successfully load policy_target')
                    if str(file_names[i].split('_')[2]) == 'q1':
                        self.q1_target.load(file_paths[i])          
                        print('successfully load q1_target')
                    if str(file_names[i].split('_')[2]) == 'q2':
                        self.q2_target.load(file_paths[i])          
                        print('successfully load q2_target')
                    if str(file_names[i].split('_')[2]) == 'h':
                        self.h_target.load(file_paths[i])          
                        print('successfully load h_target')

            for i in range(len(file_names)):
                if str(file_names[i].split('_')[1]) == 'opt':
                    if str(file_names[i].split('_')[2]) == 'policy':
                        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=Config.lr,weight_decay=1e-4)
                        self.policy_optimizer.load_state_dict(torch.load(str(file_paths[i])))
                        print('successfully load policy_optimizer')
                    if str(file_names[i].split('_')[2]) == 'q1':
                        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=Config.lr,weight_decay=1e-4)
                        self.q1_optimizer.load_state_dict(torch.load(str(file_paths[i])))        
                        print('successfully load q1_optimizer')
                    if str(file_names[i].split('_')[2]) == 'q2':
                        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=Config.lr,weight_decay=1e-4)
                        self.q2_optimizer.load_state_dict(torch.load(str(file_paths[i])))       
                        print('successfully load q2_optimizer')
                    if str(file_names[i].split('_')[2]) == 'h':
                        self.h_optimizer = optim.Adam(self.h.parameters(), lr=Config.lr,weight_decay=1e-4)
                        self.h_optimizer.load_state_dict(torch.load(str(file_paths[i])))     
                        print('successfully load h_optimizer')
                    if str(file_names[i].split('_')[2]) == 'alpha':
                        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=0.0003,weight_decay=1e-4)
                        self.alpha_optimizer.load_state_dict(torch.load(str(file_paths[i])))   
                        print('successfully load alpha_optimizer')

            print("continue train")
        else:
            print('new train')
    

