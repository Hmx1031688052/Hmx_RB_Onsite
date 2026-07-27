import torch
import torch.nn as nn
from config import Config
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
import math
from torch.autograd import Variable
from epre_dsac.parameters import agent_par
import matplotlib.pyplot as plt


# def init_weights(net):
#     for m in net.modules():
#         if isinstance(m, nn.Linear):
#             weight_shape = list(m.weight.data.size())
#             fan_in = weight_shape[1]
#             fan_out = weight_shape[0]
#             w_bound = np.sqrt(6. / (fan_in + fan_out))
#             m.weight.data.uniform_(-w_bound, w_bound)
#             m.bias.data.fill_(0)
#         elif isinstance(m, nn.BatchNorm1d):
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()
torch.manual_seed(248794110)



base_history_feature = 8 + (1 if agent_par['use_other_direction'] else 0)
path_history_feature = 3
expected_history_feature = base_history_feature + path_history_feature
history_feature = int(agent_par.get("history_feature", expected_history_feature))
if history_feature != expected_history_feature:
    raise ValueError(
        "history_feature must be {} for current frenet/global-path state, got {}. "
        "Please update epre_dsac/parameters.py and restart all worker processes.".format(
            expected_history_feature,
            history_feature,
        )
    )
USE_DIPP_EGO = False
obs_size =128*7
use_agent_map = False
use_agent_map_ego = False
use_mlp_down = False
use_cnn_down = False
low_cuda = False
use_wwr_encode = True
use_cnn_transformer = False
DROPOUT = 0.1
use_cnn_wwr = False


def assert_history_features(inputs, expected_features, encoder_name):
    actual_features = int(inputs.shape[-1])
    if actual_features != expected_features:
        raise RuntimeError(
            "{} expected env_input feature dim {}, got {}. "
            "Current state definition is 8 vehicle history features + 3 frenet/path features.".format(
                encoder_name,
                expected_features,
                actual_features,
            )
        )


class DIPP_AgentEncoder(nn.Module):
    def __init__(self):
        super(DIPP_AgentEncoder, self).__init__()
        self.motion = nn.LSTM(history_feature, 128, 2, batch_first=True)

    def forward(self, inputs):
        assert_history_features(inputs, self.motion.input_size, "DIPP_AgentEncoder")
        if self.motion.weight_ih_l0.is_cuda:
            # print(111)
            self.motion.flatten_parameters()
        traj, _ = self.motion(inputs)
        output = traj[:, -1]

        return output


class CNNMapEncoder(nn.Module):#处理未来轨迹信息，最后一个维度变成128
    def __init__(self):
        super(CNNMapEncoder, self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv1d(1, 128, 4, 4),
            nn.ReLU())
        self.con2 = nn.Sequential(
            nn.Conv1d(128, 128, 51, 51),
            nn.ReLU())
        self.pool = nn.MaxPool1d(3)

    def forward(self, inputs):#inputs:[1,3,51,4], logits:[1,128] 编码单个车辆的未来轨迹
        inputs = inputs.reshape(inputs.shape[0], -1).unsqueeze(1)
        x = self.con1(inputs)
        x = self.con2(x)
        logits = self.pool(x).squeeze(-1)

        return logits

class CNNAgentEncoder(nn.Module):#编码某一时刻的车辆交互信息
    def __init__(self):
        super(CNNAgentEncoder, self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv1d(1, 128, history_feature, history_feature),
            nn.ReLU())
        self.con2 = nn.Sequential(
            nn.Conv1d(128, 128, 1, 1),
            nn.ReLU())
        self.pool = nn.MaxPool1d(7)

    def forward(self, inputs):#inputs:[1,7,7], logits:[1,128] 编码某一时刻所有车辆的交互信息
        assert_history_features(inputs, history_feature, "CNNAgentEncoder")
        inputs = inputs.reshape(inputs.shape[0], -1).unsqueeze(1)
        x = self.con1(inputs)
        x = self.con2(x)
        logits = self.pool(x).squeeze(-1)
        return logits

class WWRCNNAgentEncoder(nn.Module):#编码车辆历史信息[1,11,6] 输入车辆历史信息，过了一个完整的transformer的编码器
    def __init__(self):
        super(WWRCNNAgentEncoder, self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv1d(1, 128, 1, 1),
            nn.ReLU())
        self.pool = nn.MaxPool1d(7)
        self.encode = PositionalEncoding(d_model=128, max_len=11)
        self.history = SelfTransformer()

    def forward(self, inputs):
        mask = torch.eq(inputs[:,:, 0], 0)
        mask[:, -1] = False
        # mask = None 
        
        print(inputs.shape)
        x = self.con1(inputs)
        print(x)
        x = self.pool(x)
        print(x)
        time = self.history(self.encode(x), mask=mask)
        output = time[:, -1]

        return output

class CNNAgent2Map(nn.Module):
    def __init__(self):
        super(CNNAgent2Map, self).__init__()
        self.position_encode = PositionalEncoding(d_model=128, max_len=51)
        self.lane = CrossTransformer()
        self.map = CrossTransformer()

    def forward(self, actor, waypoints, mask):
        query = actor.unsqueeze(1) #[1,1,128]
        waypoints = waypoints.unsqueeze(1)
        # print(11111)
        # print(waypoints.shape)
        # print(self.position_encode(waypoints[:, 0]).shape)
        # print(self.lane(query, self.position_encode(waypoints[:, 0]), mask[:, 0]).shape)

        # print(waypoints.shape)
        # print(query.shape)

        # lane_attention = torch.cat([self.lane(query, self.position_encode(waypoints[:, i]), mask[:, i]) 
        #                             for i in range(waypoints.shape[1])], dim=1) #[1,3,128]
        map_attention = self.map(query, self.position_encode(waypoints), mask) #[1,1,128]
        output = map_attention.squeeze(1)

        return output




class PositionalEncoding(nn.Module): #位置编码层
    def __init__(self, d_model: int, dropout: float = DROPOUT, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe

        return self.dropout(x)

class CrossTransformer(nn.Module): 
    def __init__(self):
        super(CrossTransformer, self).__init__()
        if low_cuda:
            self.cross_attention = nn.MultiheadAttention(128, 8, DROPOUT) #128维度，8个头，0.1丢弃率
        else:
            self.cross_attention = nn.MultiheadAttention(128, 8, DROPOUT, batch_first=True) #128维度，8个头，0.1丢弃率
        self.ffn = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 512), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(512, 128), nn.LayerNorm(128))#nn.LayerNorm(128)归一化处理，nn.Dropout(0.1)防止过拟合
        #前馈神经网路self.ffn:单隐线性层，隐藏层是输入维度的4倍
    def forward(self, query, key, mask=None):
        value = key
        if mask is not None:
            mask[:, 0] = False
        if low_cuda:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0,1)
            # if mask is not None:
            #     mask = mask.transpose(0, 1)

        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)#query[1,1,128],key[1,51,128]/[1,3,128]
        output = self.ffn(attention_output)#[1,1,128]
        if low_cuda:
            output = output.transpose(0, 1)

        return output

class SelfTransformer(nn.Module): #encode编码层
    def __init__(self):
        super(SelfTransformer, self).__init__()
        if low_cuda:
            self.self_attention = nn.MultiheadAttention(128, 8, DROPOUT)
        else:
            self.self_attention = nn.MultiheadAttention(128, 8, DROPOUT, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 512), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(512, 128), nn.LayerNorm(128))

    def forward(self, input, mask=None):
        if low_cuda:
            input = input.transpose(0, 1) 
            
            # if mask is not None:
            #     mask = mask.transpose(0, 1)

        attention_output, _ = self.self_attention(input, input, input, key_padding_mask=mask)
        output = self.ffn(attention_output)
        if low_cuda:
            output = output.transpose(0, 1)

        return output

class AgentEncoder(nn.Module):#编码车辆历史信息[1,11,6] 输入车辆历史信息，过了一个完整的transformer的编码器
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.position = nn.Sequential(nn.Linear(history_feature, 64), nn.ReLU(), nn.Linear(64, 128))
        self.encode = PositionalEncoding(d_model=128, max_len=11)
        self.history = SelfTransformer()

    def forward(self, inputs):
        assert_history_features(inputs, self.position[0].in_features, "AgentEncoder")
        mask = torch.eq(inputs[:,:, 0], 0)
        mask[:, -1] = False
        # mask = None 
        time = self.history(self.encode(self.position(inputs)), mask=mask)
        output = time[:, -1]

        return output
    

class MapEncoder(nn.Module):#处理未来轨迹信息，最后一个维度变成128
    def __init__(self):
        super(MapEncoder, self).__init__()
        self.waypoint = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 128))

    def forward(self, inputs):
        output = self.waypoint(inputs)
    
        return output

class Agent2Agent(nn.Module):
    def __init__(self):
        super(Agent2Agent, self).__init__()
        self.interaction_1 = SelfTransformer()
        self.interaction_2 = SelfTransformer()

    def forward(self, inputs, mask=None):
        output = self.interaction_1(inputs, mask=mask)
        output = self.interaction_2(inputs+output, mask=mask)

        return output

class Agent2Map(nn.Module):
    def __init__(self):
        super(Agent2Map, self).__init__()
        self.position_encode = PositionalEncoding(d_model=128, max_len=51)
        self.lane = CrossTransformer()
        self.map = CrossTransformer()

    def forward(self, actor, waypoints, mask):
        query = actor.unsqueeze(1) #[1,1,128]
        # print(11111)
        # print(waypoints.shape)
        # print(self.position_encode(waypoints[:, 0]).shape)
        # print(self.lane(query, self.position_encode(waypoints[:, 0]), mask[:, 0]).shape)
        lane_attention = torch.cat([self.lane(query, self.position_encode(waypoints[:, i]), mask[:, i]) 
                                    for i in range(waypoints.shape[1])], dim=1) #[1,3,128]
        map_attention = self.map(query, lane_attention, mask[:, :, 10]) #[1,1,128]
        output = map_attention.squeeze(1)

        return output
    
class Decoder(nn.Module):
    def __init__(self, use_interaction):
        super(Decoder, self).__init__()
        self.use_interaction = use_interaction
        if use_interaction:
            self.cell = nn.GRUCell(input_size=128, hidden_size=384)
            self.plan_input = nn.Linear(3, 128)
            self.state_input = nn.Linear(3, 128)
        else:
            self.cell = nn.GRUCell(input_size=3, hidden_size=384)
        self.decode = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(384, 64), nn.ELU(), nn.Linear(64, 3))

    def forward(self, init_hidden, plan, gate, init_state):
        output = []
        hidden = init_hidden
        state = init_state

        for t in range(5):
            if self.use_interaction:
                plan_input = self.plan_input(plan[:, t, :3]) 
                state_input = self.state_input(state[:, :3])
                input = state_input + plan_input * gate
            else:
                input = state[:, :3]

            hidden = self.cell(input, hidden)
            state = self.decode(hidden) + state[:, :3]
            output.append(state)

        output = torch.stack(output, dim=1)

        return output


class QNet(nn.Module):

    def __init__(self, state_num, action_num=2):
        super().__init__()
        # self.seed = torch.manual_seed(914)
        self.state_num = int(state_num)
        self.action_num = action_num
        self.fc1 = nn.Linear(self.state_num + self.action_num, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 2)

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.gelu(x)
        logits = self.fc4(x)
        value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
        value_log_std = torch.nn.functional.softplus(value_std)  # avoid 0

        return torch.cat((value_mean, value_log_std), dim=-1)
    
    # def save(self, path, step, optimizer):
    #     torch.save({
    #         'step': step,
    #         'state_dict': self.state_dict(),
    #         'optimizer': optimizer.state_dict()
    #     }, path)
    #
    def load(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(torch.load(checkpoint_path))
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
    #
    # def evaluate(self, state, action, device=torch.device("cpu"), min=False):
    #     mean, log_std = self.forward(state, action)
    #     std = log_std.exp()
    #     normal = Normal(torch.zeros(mean.shape), torch.ones(std.shape))
    #
    #     if min == False:
    #         z = normal.sample().to(device)
    #         z = torch.clamp(z, -2, 2)
    #     elif min == True:
    #         z = -torch.abs(normal.sample()).to(device)
    #
    #     q_value = mean + torch.mul(z, std)
    #     return mean, std, q_value


class PolicyNet(nn.Module):

    def __init__(self, state_num, action_num):
        super().__init__()
        # self.seed = torch.manual_seed(914)
        self.state_num = int(state_num)
        self.action_num = action_num
        self.fc1 = nn.Linear(self.state_num, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, self.action_num * 2)
        self.min_log_std = -20
        self.max_log_std = 0.5

    def forward(self, observation):
        x = observation
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.gelu(x)
        logits = self.fc4(x)
        action_mean, action_log_std = torch.chunk(
            logits, chunks=2, dim=-1
        )  # output the mean
        action_std = torch.clamp(
            action_log_std, self.min_log_std, self.max_log_std
        ).exp()
        action_mean = action_mean
        return torch.cat((action_mean, action_std), dim=-1)

    def reset(self):
        # 重新初始化fc4层
        self.fc4.reset_parameters()


    # def save(self, path, step, optimizer):
    #     torch.save({
    #         'step':step,
    #         'state_dict':self.state_dict(),
    #         'optimizer':optimizer.state_dict()
    #     }, path)

    def load(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(torch.load(checkpoint_path))
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])


class SafetyCritic(nn.Module):
    """Unbounded scalar critic used for FDPI safety and recovery values."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=-1)).squeeze(-1)


# class HNet(nn.Module):
#
#     def __init__(self, set_state_num):
#         super().__init__()
#         self.set_state_num = set_state_num
#         self.fc1 = nn.Linear(self.set_state_num, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.fc4 = nn.Linear(256, 32)
#
#     def forward(self, set_observation):
#         x = set_observation
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         x = F.gelu(x)
#         x = self.fc3(x)
#         x = F.gelu(x)
#         logits = self.fc4(x)
#         return logits
#
#
#     # def save(self, path, step, optimizer):
#     #     torch.save({
#     #         'step':step,
#     #         'state_dict':self.state_dict(),
#     #         'optimizer':optimizer.state_dict()
#     #     }, path)
#
#     def load(self, checkpoint_path, optimizer=None):
#         checkpoint = torch.load(checkpoint_path)
#         self.load_state_dict(torch.load(checkpoint_path))
#         if optimizer is not None:
#             optimizer.load_state_dict(checkpoint['optimizer'])

# class HNet(nn.Module):

#     def __init__(self, set_state_num):
#         super().__init__()
#         self.set_state_num = set_state_num
#         self.con1 = nn.Sequential(
#             nn.Conv1d(1, 32, 6, 6),
#             nn.ReLU())
#         self.con2 = nn.Sequential(
#             nn.Conv1d(32, 32, 1, 1),
#             nn.ReLU())
#         self.pool = nn.MaxPool1d(6)
#         self.fc1 = nn.Linear(self.set_state_num, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.fc4 = nn.Linear(256, 32)

#     def forward(self, set_observation):
#         x = set_observation.unsqueeze(1)
#         x_other = self.con1(x)
#         x_other = self.con2(x_other)
#         logits = self.pool(x_other).squeeze(-1)
#         return logits


#     # def save(self, path, step, optimizer):
#     #     torch.save({
#     #         'step':step,
#     #         'state_dict':self.state_dict(),
#     #         'optimizer':optimizer.state_dict()
#     #     }, path)

#     def load(self, checkpoint_path, optimizer=None):
#         checkpoint = torch.load(checkpoint_path)
#         self.load_state_dict(torch.load(checkpoint_path))
#         if optimizer is not None:
#             optimizer.load_state_dict(checkpoint['optimizer'])


class HNet(nn.Module):

    def __init__(self, set_state_num):
        super().__init__()
        # agent layer
        self.ego_net = AgentEncoder()
        self.neighbor_net = AgentEncoder()

        # map layer
        self.map_net = MapEncoder()


        self.cnn_map_net = CNNMapEncoder()
        self.cnn_agent_net = CNNAgentEncoder()
        self.cnn_agent_map = CNNAgent2Map()


        self.history = SelfTransformer()
        self.agent_position_encode = PositionalEncoding(d_model=128, max_len=11)
        self.map_position_encode = PositionalEncoding(d_model=128, max_len=7)
        self.agent_map_encoder = CrossTransformer()
        

        # attention layers
        self.agent_map = Agent2Map()
        self.agent_agent = Agent2Agent()

        self.gate = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

        # decoder layer
        self.decoder = Decoder(False)

        if USE_DIPP_EGO:
            self.DIPP_ego_net = DIPP_AgentEncoder()
            self.DIPP_neighbor_net = DIPP_AgentEncoder()

        self.fc1 = nn.Linear(128,32)
        self.fc2 = nn.Linear(224, 512)
        self.fc3 = nn.Linear(512, 128)

        if use_mlp_down:
            self.encoder = nn.Sequential(
                nn.Linear(128*7, 1024),  # 输入层到第一个隐藏层
                nn.ReLU(),  # 激活函数
                nn.Linear(1024, 1024),  # 第二个隐藏层
                nn.ReLU(),
                nn.Linear(1024, 128)
            )
        elif use_cnn_down:
            self.con1 = nn.Sequential(
                nn.Conv1d(1, 128, 8, 8),
                nn.ReLU())
            self.con2 = nn.Sequential(
                nn.Conv1d(128, 128, 6, 6),
                nn.ReLU())
            self.con3 = nn.Sequential(
                nn.Conv1d(128, 128, 2, 2),
                nn.ReLU())
            self.pool = nn.MaxPool1d(6)
        self.i = 0


    def forward(self, env_input, env_map):
        # self.i += 1
        # if self.i > 50:
        #     print(env_input.shape, env_map.shape)
        #     # for i in env_input.cpu():
        #     #     for j in i:
        #     #         plt.plot(j[0], j[1], 'ko', markersize = 0.2)
        #     # plt.show()
        #     for i in env_map.cpu():
        #         for j in i:
        #             for k in j:
        #                 plt.plot(k[0], k[1], 'ko', markersize = 0.2)
        #     plt.xlim(-100, 100)  # 设置x轴范围
        #     plt.ylim(-100, 100)  # 设置y轴范围
        #     plt.show()

        if env_map.shape[0] == 7:
                env_map = env_map.unsqueeze(0)
        if env_input.shape[0] == 7:
                env_input = env_input.unsqueeze(0)
        if use_cnn_transformer:
            
            cnn_map_encode = [self.cnn_map_net(env_map[:,i])for i in range(env_map.shape[1])] #编码每个车辆的未来轨迹信息
            cnn_map_encode = torch.stack(cnn_map_encode,dim=0)
            cnn_map_encode = Variable(cnn_map_encode)
            cnn_map_encode = cnn_map_encode.permute(1, 0, 2) #[batch_size, 7, 128]

            cnn_agent_encode = [self.cnn_agent_net(env_input[:,:,0])for i in range(env_input.shape[2])] #编码同一时间的车辆交互  #[11,batch_size, 128]
            cnn_agent_encode = torch.stack(cnn_agent_encode,dim=0)
            cnn_agent_encode = Variable(cnn_agent_encode)
            cnn_agent_encode = cnn_agent_encode.permute(1, 0, 2) #[batch_size, 11, 128]

            agent_time_encode = self.history(self.agent_position_encode(cnn_agent_encode), mask = None)[:,-1] #时间编码，不同时间的车辆交互信息#[batch_size, 128]
            agent_map_encode = self.agent_map_encoder(agent_time_encode.unsqueeze(1), self.map_position_encode(cnn_map_encode), mask = None)[:,-1] #车辆交互信息与地图编码[batch_size, 128]
            logits = agent_map_encode

        elif use_cnn_wwr:  #map用cnn编码
            ego = env_input[:,0,:,:]
            neighbors_state = env_input[:,1:,:,:]
            if USE_DIPP_EGO:
                ego_encode = [self.DIPP_ego_net(ego)]  
                neighbors = [self.DIPP_neighbor_net(neighbors_state[:,i])for i in range(neighbors_state.shape[1])]
                encoded_actors = torch.stack(ego_encode + neighbors, dim=1)
            else:
                neighbors = [self.neighbor_net(neighbors_state[:,i])for i in range(neighbors_state.shape[1])]
                ego_encode = [self.ego_net(ego)] #[1,128]
                encoded_actors = torch.stack(ego_encode + neighbors, dim=1)

            cnn_map_encode = [self.cnn_map_net(env_map[:,i])for i in range(env_map.shape[1])] #编码每个车辆的未来轨迹信息
            cnn_map_encode = torch.stack(cnn_map_encode,dim=0)
            cnn_map_encode = Variable(cnn_map_encode)
            cnn_map_encode = cnn_map_encode.permute(1, 0, 2) #[batch_size, 7, 128]

            actor_map_list = []

            for i in range(neighbors_state.shape[1]+1):
                map_mask = None
                agent_map = self.cnn_agent_map(encoded_actors[:,i], cnn_map_encode[:, i], map_mask) #编码交通参与者和地图[1,128]
                actor_map_list.append(agent_map)
            actor_map = torch.stack(actor_map_list,dim=1)
            actor_mask = torch.eq(torch.cat([ego.unsqueeze(1), neighbors_state], dim=1), 0)[:, :, -1, 0]
            actor_mask[:, 0] = False
            agent_agent = self.agent_agent(actor_map, actor_mask)
            logits = agent_agent.contiguous().reshape(agent_agent.shape[0], -1)


        else:
            
            ego = env_input[:,0,:,:]
            neighbors_state = env_input[:,1:,:,:]

            #####################################################DIPP中LSTM网络进行自车和旁车历史行为编码
            if USE_DIPP_EGO:
                ego_encode = [self.DIPP_ego_net(ego)]  
                neighbors = [self.DIPP_neighbor_net(neighbors_state[:,i])for i in range(neighbors_state.shape[1])]
                encoded_actors = torch.stack(ego_encode + neighbors, dim=1)
            else:
                neighbors = [self.neighbor_net(neighbors_state[:,i])for i in range(neighbors_state.shape[1])]
                ego_encode = [self.ego_net(ego)] #[1,128]
                encoded_actors = torch.stack(ego_encode + neighbors, dim=1)

            ego_map = env_map[:, 0]
            neighbor_map = env_map[:, 1:]
            encoded_neighbor_map = self.map_net(neighbor_map)
            if use_agent_map_ego or use_wwr_encode:
                encoded_ego_map = self.map_net(ego_map)

            if use_wwr_encode:
                actor_map_list = []
                neighbor_map_list = []
                ego_map_mask = torch.eq(ego_map[:, :, :, -1], 0)
                ego_agent_map = self.agent_map(ego_encode[0], encoded_ego_map, ego_map_mask)
                actor_map_list.append(ego_agent_map)

                for i in range(neighbors_state.shape[1]):
                    map_mask = torch.eq(neighbor_map[:, i, :, :, -1], 0)
                    agent_map = self.agent_map(neighbors[i], encoded_neighbor_map[:, i], map_mask) #编码交通参与者和地图[1,128]
                    neighbor_map_list.append(agent_map)
                actor_map = torch.stack(actor_map_list+neighbor_map_list,dim=1)#ego+所有旁车的各自与其路点进行编码[1,7,128]
                actor_mask = torch.eq(torch.cat([ego.unsqueeze(1), neighbors_state], dim=1), 0)[:, :, -1, 0]
                actor_mask[:, 0] = False
                
                agent_agent = self.agent_agent(actor_map, actor_mask)
                logits = agent_agent.contiguous().reshape(agent_agent.shape[0], -1)

                
            else:
                actor_mask = torch.eq(torch.cat([ego.unsqueeze(1), neighbors_state], dim=1), 0)[:, :, -1, 0]
                actor_mask[:, 0] = False
                agent_agent = self.agent_agent(encoded_actors, actor_mask) #[1,7,128]

                per_agent_tensor_list = []
                agent_map_list = []
                agent_map_ego_list = []


                if use_agent_map_ego:
                    ego_map_mask = torch.eq(ego_map[:, :, :, -1], 0)
                    ego_agent_map = self.agent_map(agent_agent[:, 0], encoded_ego_map, ego_map_mask)
                    agent_map_ego_list.append(ego_agent_map)
                
                for i in range(neighbors_state.shape[1]):
                    map_mask = torch.eq(neighbor_map[:, i, :, :, -1], 0)
                    agent_map = self.agent_map(agent_agent[:, i+1], encoded_neighbor_map[:, i], map_mask) #编码交通参与者和地图[1,128]
                    per_agent_tensor_list.append(torch.cat([agent_map, neighbors[i], agent_agent[:, i+1]], dim=-1)) #隐藏编码信息 [1,3*128]
                    if use_agent_map:
                        agent_map_list.append(agent_map)
                    elif use_agent_map_ego:
                        agent_map_ego_list.append(agent_map)


                if  use_agent_map:
                    agent_map_tensor = torch.stack(agent_map_list,dim=0)
                    agent_map_tensor = Variable(agent_map_tensor)
                    agent_map_tensor = agent_map_tensor.unsqueeze(0)
                    agent_map_tensor = agent_map_tensor.permute(2, 1, 0, 3)  # 调整维度顺序
                    agent_map_tensor = agent_map_tensor.contiguous().view(agent_map_tensor.shape[0], 6, 1, 128)
                    agent_map_tensor = agent_map_tensor.view(agent_map_tensor.shape[0], -1)
                    logits = agent_map_tensor
                elif use_agent_map_ego:
                    agent_map_tensor = torch.stack(agent_map_ego_list,dim=0)
                    agent_map_tensor = Variable(agent_map_tensor)
                    agent_map_tensor = agent_map_tensor.unsqueeze(0)
                    agent_map_tensor = agent_map_tensor.permute(2, 1, 0, 3)  # 调整维度顺序
                    agent_map_tensor = agent_map_tensor.contiguous().view(agent_map_tensor.shape[0], 7, 1, 128)
                    agent_map_tensor = agent_map_tensor.view(agent_map_tensor.shape[0], -1)
                    logits = agent_map_tensor
                else:
                    per_agent_tensor = torch.stack(per_agent_tensor_list,dim=0)
                    per_agent_tensor = Variable(per_agent_tensor)
                    per_agent_tensor = per_agent_tensor.unsqueeze(0)
                    per_agent_tensor = per_agent_tensor.permute(2, 1, 0, 3)  # 调整维度顺序
                    per_agent_tensor = per_agent_tensor.contiguous().view(per_agent_tensor.shape[0], 6, 1, 384)
                    per_agent_tensor = per_agent_tensor.view(per_agent_tensor.shape[0], -1)
                    logits = per_agent_tensor
                

        if use_mlp_down:
            logits = self.encoder(logits)
        elif use_cnn_down:
            x = logits.unsqueeze(1)
            x_other = self.con1(x)
            x_other = self.con2(x_other)
            x_other = self.con3(x_other)
            logits = self.pool(x_other).squeeze(-1)
        


        # per_agent_prediction_list = []
        # for i in range(neighbors_state.shape[1]):
        #     gate = self.gate(torch.cat([ego_encode[0], neighbors[i]], dim=-1))
        #     predict_traj = self.decoder(per_agent_tensor_list[i], None, gate, neighbors_state[:, i, -1])
        #     per_agent_prediction_list.append(predict_traj)

        # prediction = torch.stack(per_agent_prediction_list, dim=1)
        # prediction = prediction.view(prediction.shape[0], 6*5*3)

        # agent_encode = torch.stack([self.fc1(agent_agent[:,i])for i in range(agent_agent.shape[1])], dim=1)
        # agent_encode = agent_encode.reshape(agent_encode.shape[0],7*32)
        # logits = self.fc2(agent_encode)
        # logits = self.fc3(logits)
        return logits


    # def save(self, path, step, optimizer):
    #     torch.save({
    #         'step':step,
    #         'state_dict':self.state_dict(),
    #         'optimizer':optimizer.state_dict()
    #     }, path)

    def load(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(torch.load(checkpoint_path))
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])


    
