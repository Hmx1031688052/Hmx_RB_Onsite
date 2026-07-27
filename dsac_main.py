import os 
import random
import numpy as np
import argparse
from env import Env
from config import Config
from reply_buffer import Reply_Buffer
from utils import *
from collections import deque
from utilss import logger
import time
import torch
from torch.autograd import Variable
import multiprocessing as mp
import queue
from Worker_agent import Worker_paral
from epre_dsac.parameters import agent_par
from utils.opendrive2discretenet import parse_opendrive
import matplotlib.pyplot as plt


np.random.seed(914)
random.seed(914)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['PYTHONHASHSEED'] = '1'
class Main():
    def __init__(self,use_epre_dsac = False):
        algorithm_mode = str(agent_par.get("algorithm_mode", "stt_dsac")).lower()
        if algorithm_mode not in (
            "original_dsac", "stt_dsac", "dsac_fdpi", "dsac_fdpi_stt"
        ):
            raise ValueError("unsupported algorithm_mode: {}".format(algorithm_mode))
        # Keep the old constructor argument compatible, but make the explicit
        # algorithm selection authoritative for new runs.
        use_epre_dsac = algorithm_mode != "original_dsac"
        localtime = time.strftime("%Y-%m-%d_%H%M" , time.localtime())  # 获取本地时间
        q1_name = 'q1_{}_{}_{}.pth'.format(Config.reply_buffer_size, Config.total_episode, Config.lr)
        q2_name = 'q2_{}_{}_{}.pth'.format(Config.reply_buffer_size, Config.total_episode, Config.lr)
        policy_name = 'policy_{}_{}_{}.pth'.format(Config.reply_buffer_size, Config.total_episode, Config.lr)
        h_name = 'h_{}_{}_{}.pth'.format(Config.reply_buffer_size, Config.total_episode, Config.lr)
        ModelPath = os.path.dirname(os.path.realpath(
            __file__)) + '/logs/' + str(localtime) + '/model/'
        logpath = os.path.dirname(os.path.realpath(
            __file__)) + '/logs/' + str(localtime) + '/log/'
       
        self.train = True
        self.continue_train = True
        self.state = None
        self.step = 0
        self.global_step = 0
        self.action_request_id = 0
        self.last_action = 0
        self.state_size = 0
        self.map_name = None
        self.episode = 0
        self.env = Env(self.train, use_epre_dsac = use_epre_dsac)
        self.start = 0
        self.map = None
        self.use_epre_dsac = use_epre_dsac
        self.relax_time = 0
        self.good_step = False
        self.good_reward_list = []
        self.last_time = None
        self.last_step_time = 0.02
        self.last_acc = 0
        self.acc_limit = 0.15
        self.time_last = None
        self.map_dir = os.path.join('samples_epre_wutfsd', 'maps')
        
        if self.train:
            self.input_queue = mp.Queue()
            self.output_queue = mp.Queue()
            worker = Worker_paral(self.input_queue, self.output_queue, train=self.train, use_epre_dsac = self.use_epre_dsac)
            worker.daemon = True
            worker.start()
            self.worker = worker

        else:
            self.input_queue = mp.Queue()
            self.output_queue = mp.Queue()
            worker = Worker_paral(self.input_queue, self.output_queue, train=self.train, use_epre_dsac = self.use_epre_dsac)
            worker.daemon = True
            worker.start()
            self.worker = worker
        # if self.continue_train and self.train:
        #     self.load_continue_net()


    def get_map_jiexi_old(self):
        return self.get_map_jiexi()

    def get_action(self, ego_x, ego_y, ego_v, ego_a, ego_yaw, ego_length, ego_width, obstacles, r, cost, done, collision_done, time_out_done, ganzhi):
        time3 = time.time()
        if self.step == 0:
            self.last_time = time.time()
        # print([ego_x, ego_y, ego_v, ego_a, ego_yaw, ego_length, ego_width,  r, done])
        # print(obstacles)
        if self.start == 0:
            return [0.0, 0.0], [0.0, 0.0]
        other_data = {'1':{},'2':{},'3':{},'4':{},'5':{},'6':{},'7':{},'8':{},'9':{},'10':{},'11':{},'12':{},'13':{},'14':{},'15':{},'16':{},'17':{},'18':{},'19':{},'20':{},'21':{},'22':{},'23':{},'24':{},'25':{},'26':{},'27':{},'28':{},'29':{},'30':{},'31':{},'32':{},'33':{},'34':{},'35':{},'36':{},'37':{},'38':{},'39':{},'40':{}}
        i = 1
        obstacles1 = obstacles
        for other in obstacles:
            if str(other.id) != str(-1):
                i_id = str(other.id)
                other_data[i_id]['x'] = other.x
                other_data[i_id]['y'] = other.y
                other_data[i_id]['v'] = other.speed
                other_data[i_id]['a'] = other.acc
                other_data[i_id]['yaw'] = other.theta
                other_data[i_id]['length'] = other.length
                other_data[i_id]['width'] = other.width
                other_data[i_id]['id'] = other.id
                
                if str(other.roleType) == 'RoleType.PEDESTRIAN':
                    other_data[i_id]['type'] = 1
                else:
                    other_data[i_id]['type'] = 0
                i +=1
            
        obstacles = other_data
        reward = r
        # print(state[:7])
        # print(self.last_action)
        # print(1,time.time() - time3)
        
        if self.use_epre_dsac:
            self.relax_time += 1
            self.good_step = False
            
            if  len(obstacles1)>0 or done or collision_done or time_out_done or ganzhi==1:
                # if self.time_last is not None:
                #     print(time.time() - self.time_last)
                # self.time_last = time.time()
                self.good_step = True
                self.env.episode_step += 1
                self.relax_time = 0

            time1 = time.time()
            state, env_input, env_map= self.env._get_features(ego_x, ego_y, ego_v, ego_a, ego_yaw, ego_length, ego_width, obstacles, i)
            # print(111,time.time() - time1)
            if state is None:
                return [0.0, 0.0], [0.0, 0.0]
            
            # print('env',time.time() - time1)
            # print(2, time.time() - time3)

            # if self.env.change_lane_success and not collision_done and not time_out_done:
            #     reward += 50

            self.good_reward_list.append(reward)
            expected_request_id = None
            if self.good_step:
                state = np.reshape(state, [1, self.env.state_dim]).astype(float)
                good_reward = max(self.good_reward_list, key=abs)
                self.action_request_id += 1
                expected_request_id = self.action_request_id
                self.input_queue.put([
                    'act', expected_request_id, [state, env_input, env_map],
                    good_reward, cost, done, collision_done, time_out_done,
                    self.env.is_intersection,
                ])
                self.good_reward_list=[]
            # print(3,time.time() - time3)

        else:
            state = self.env._get_features(ego_x, ego_y, ego_v, ego_a, ego_yaw, ego_length, ego_width, obstacles, state[10])
            if state is None:
                return [0.0, 0.0], [0.0, 0.0]
            state = np.reshape(state, [1, self.env.state_dim]).astype(float)

            self.action_request_id += 1
            expected_request_id = self.action_request_id
            self.input_queue.put([
                'act', expected_request_id, state, reward, cost, done,
                collision_done, time_out_done, False,
            ])
        try:

            # if self.env.is_intersection:
            #     action = 5
            # else:
            #     import sys
            #     import select
                
            #     # 设置0.1秒的超时等待用户输入
            #     ready, _, _ = select.select([sys.stdin], [], [], 0.0005)
                
            #     if ready:  # 如果检测到输入
            #         user_input = sys.stdin.readline().strip()
            #         if user_input and int(user_input) <= 3:  # 如果有实际输入内容
            #             action = int(user_input)  # 将输入转换为整数
            #             print(f"用户选择了动作: {action}")
            #         else:
            #             raise ValueError("无输入内容")
            #     else:
            #         raise TimeoutError("超时未输入")
            #     print('action', action)
            


            # print(4,time.time() - time3)
            if expected_request_id is None:
                raise queue.Empty
            deadline = time.time() + float(os.environ.get("E2E_ACT_TIMEOUT", "1.0"))
            while True:
                k = self.output_queue.get(timeout=max(0.0, deadline - time.time()))
                response_id = k[1] if len(k) > 1 else None
                if response_id == expected_request_id:
                    break
                print(
                    f"[main][WARN] ignore stale action response "
                    f"response_id={response_id} expected={expected_request_id}",
                    flush=True,
                )
            
            # print(5,time.time() - time3)
            
            action = k[2]
        
            assert k[0] == 'act success!'
            # if self.env.map_type == 'AI_town':
            #     action = 0
            # action = -1
            time2 = time.time()


            control = self.env.cal_control(action, self.step)
            # print(222,time.time() - time2)
            # print('control',time.time() - time2)

            self.last_action = action
            self.global_step += 1
            self.step+=1
            self.env.guikong_step += 1
            
            # print('get_action',time.time()-time3)

            # if control[0] > 3:
            #     control[0] = 2.99
            # elif control[0] < -3:
            #     control[0] = -2.99

            # acc_error = control[0] - self.last_acc
            # if acc_error > self.acc_limit:
            #     control[0] = self.last_acc + self.acc_limit
            # elif acc_error < - self.acc_limit:
            #     control[0] = self.last_acc - self.acc_limit
                
            # self.last_acc = control[0]

            return control, action
        except:
            action = self.last_action
            time2 = time.time()
            control = self.env.cal_control(action, self.step)
            # print('control2',time.time() - time2)
            self.global_step += 1
            self.step+=1
            self.env.guikong_step += 1

            # print('get_action2',time.time()-time3)

            # if control[0] > 3:
            #     control[0] = 2.99
            # elif control[0] < -3:
            #     control[0] = -2.99

            # acc_error = control[0] - self.last_acc
            # if acc_error > self.acc_limit:
            #     control[0] = self.last_acc + self.acc_limit
            # elif acc_error < - self.acc_limit:
            #     control[0] = self.last_acc - self.acc_limit
                
            # self.last_acc = control[0]

            return control, action
        


    def get_map_jiexi(self):
        # Kept for old callers; maps are loaded per scenario in change_map().
        return None

    def _load_map(self, map_name):
        map_path = os.path.join(self.map_dir, map_name)
        if not os.path.exists(map_path):
            raise FileNotFoundError(f'OpenDRIVE map not found: {map_path}')
        print('load map', map_name)
        return parse_opendrive(map_path)

    def change_map(self, brief_data, weather):
        target_state = brief_data['testees'][0]['target_state']
        x_goal = target_state['x']
        y_goal = target_state['y']
        start_state = brief_data['testees'][0]['init_state']
        x_start = start_state['x']
        y_start = start_state['y']
        zjl_odv_file = brief_data['zjl_odv_file']
        self.map_name = brief_data.get('map_name')
        self.map = os.path.splitext(zjl_odv_file)[0]
        print('mmmm', zjl_odv_file)
        map_jiexi = self._load_map(zjl_odv_file)
        self.step = 0
        self.total_reward = 0
        self.env.reset(x_start, y_start, x_goal, y_goal, map_jiexi, zjl_odv_file, weather)
        self.start = 1

    def finish(self, collision, done_type, real):
        if self.step != 0 and self.last_time!=0:
            self.last_step_time = (time.time() - self.last_time)/self.step
            print(self.last_step_time, 'ave_global_time')
        self.last_time = 0
        
        # print(self.step, self.last_step_time*self.step, 'global_step')
        self.step = 0
        self.last_acc = 0
        self.start = 0
        self.good_reward_list = []
        data = [collision, done_type, real]
        self.input_queue.put(['end', data])
        output_list = []
        print('进入finish函数,等待act_end success!')
        finish_timeout = float(os.environ.get("E2E_FINISH_TIMEOUT", "35.0"))
        finish_wait_start = time.time()
        last_finish_wait_log = finish_wait_start
        worker_alive = self.worker.is_alive() if hasattr(self, "worker") else None
        print(
            f"[finish-debug][main] sent end done_type={done_type} real={real} "
            f"worker_alive={worker_alive} timeout={finish_timeout:.1f}s",
            flush=True,
        )
        if self.env.show_guiji:
            for j in self.env.guiji_list:
                plt.plot(j[0], j[1], 'ko', markersize = 0.2)
            for i in range(len(self.env.road_info_dict_pingjie)):
                road_1 =self.env.road_info_dict_pingjie[str(i)]['center_vertices']
                road_1 = np.array(road_1)
                # 提取x坐标和y坐标
                x = road_1[:,0]
                y = road_1[:,1]
                # 绘制点
                bind = 'ro' #红
                line = 'r-'
                if i ==1:
                    bind = 'go'#绿
                    line = 'g-'
                elif i == 2:
                    bind = 'bo'#蓝
                    line = 'b-'  
                elif i == 3:
                    bind = 'co'#青
                    line = 'c-'  
                elif i == 4:
                    bind = 'yo'#黄
                    line = 'y-'  
                elif i == 5:
                    bind = 'mo'#洋红
                    line = 'm-'  
                elif i == 6:
                    bind = 'grey'#灰
                    line = 'grey'  
                plt.plot(x, y, bind, markersize = 0.2)  

                # 绘制连线
                plt.plot(x, y, line, linewidth=0.2) 

            if self.env.tongji:
                for j in self.env.bijingdian:
                    plt.plot(j[0], j[1], 'ko', zorder = 10,markersize = 3)

            plt.plot(self.env.x_start, self.env.y_start, 'wo', zorder = 10)
            plt.plot(self.env.x_goal, self.env.y_goal, 'ko',zorder = 10)
            plt.gca().set_aspect('equal')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Lane Center Points')
            plt.show()
        
        # while True:
        #     try:
        #         k = self.output_queue.get(timeout=120)
        #         output_list.append(k)
        #         if k[0] == 'act_end success!':
        #             break
        #     except:
        #         print('找不到finish!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #         output_list.append(['act_end success!'])

        while True:
            try:
                k = self.output_queue.get(timeout=1.0)
                output_list.append(k)
                print(f"[finish-debug][main] got output {k[0]}", flush=True)
                if k[0] == 'act_end success!':
                    break
                if k[0] == 'act_end failed!':
                    print(k[1], flush=True)
                    break
                #这个进程这里拿不到动作。
            except KeyboardInterrupt:
                raise
            except Exception:
                now = time.time()
                if now - last_finish_wait_log >= 1.0:
                    worker_alive = self.worker.is_alive() if hasattr(self, "worker") else None
                    print(
                        f"[finish-debug][main] still waiting act_end "
                        f"elapsed={now - finish_wait_start:.1f}s worker_alive={worker_alive}",
                        flush=True,
                    )
                    last_finish_wait_log = now
                if now - finish_wait_start >= finish_timeout:
                    print('找不到finish!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    output_list.append(['act_end timeout!'])
                    break
        print(output_list)
        assert output_list[-1][0] in ('act_end success!', 'act_end timeout!', 'act_end failed!')

    

    def load_net(self, policy, q1, q2, h):
        net = [policy, q1, q2, h]
        self.input_queue.put(['load_net', net])
        assert self.output_queue.get()[0] == 'load_net success!'
    
    def load_continue_net(self):
        self.input_queue.put(['load_continue_net'])
        assert self.output_queue.get()[0] == 'load_continue_net success!'
