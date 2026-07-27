from dataclasses import dataclass, field
import numpy as np
from typing import List, Tuple, Dict
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import matplotlib.animation as animation

import math
import time
import re ,ast
import csv
import os
import pandas as pd

import logging
logging.basicConfig(level=logging.ERROR)

C_RISK_LABEL = 'white'

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def v_split(v, phi):#当输入为世界坐标系下的和速度是
    return v , 0 #车体坐标系下的速度
    # return v * math.cos(phi), v * math.sin(phi) #世界坐标系下的速度


def dynamic_model(states, actions, delta_t):#单轨车辆动力学模型
    k_f=-128915.5  # front wheel cornering stiffness [N/rad]
    k_r=-85943.6  # rear wheel cornering stiffness [N/rad]
    l_f=1.06  # distance from CG to front axle [m]
    l_r=1.85  # distance from CG to rear axle [m]
    m=1412.0  # mass [kg]
    I_z=1536.7  # Polar moment of inertia at CG [kg*m^2]

    x, y, phi, u, v, w = states#当前状态
    steer, a_x = actions#控制输入

    next_state = [
        x + delta_t * (u * np.cos(phi) - v * np.sin(phi)),
        y + delta_t * (u * np.sin(phi) + v * np.cos(phi)),
        phi + delta_t * w,
        np.clip(u + delta_t * a_x, 0., None),
        (
            m * v * u
            + delta_t * (l_f * k_f - l_r * k_r) * w
            - delta_t * k_f * steer * u
            - delta_t * m * np.square(u) * w
        )
        / (m * u - delta_t * (k_f + k_r)),
        (
            I_z * w * u
            + delta_t * (l_f * k_f - l_r * k_r) * v
            - delta_t * l_f * k_f * steer * u
        )
        / (I_z * u - delta_t * (np.square(l_f) * k_f + np.square(l_r) * k_r)),
    ]
    next_state[2] = angle_normalize(next_state[2])
    return np.array(next_state, dtype=np.float32)


def rotation(l, w, phi):
    """phi: rad"""
    diff_x = l * np.cos(phi) - w * np.sin(phi)
    diff_y = l * np.sin(phi) + w * np.cos(phi)
    return (diff_x, diff_y)


@dataclass
class Veh:
    # attributes
    id: int = None
    name: str = None
    type: str = None
    length: float = None
    width: float = None
    fix_damage: float = None
    rel_posi: str = None #相对位置
    

    # states
    x: float = None  # [m]
    y: float = None  # [m]
    phi: float = None  # heading angle, to east = 0 [rad]
    u: float = None  # longitudinal/scalar velocity [m/s]
    v: float = 0.  # lateral velocity [m/s]
    w: float = 0.  # angular velocity [rad/s]
    s: float = 0.
    p: float = 0.
    
    # actions
    a: float = 0.  # acceleration  [m/s2]
    st: float = 0.  # steering angle  [rad]

    # prediction info
    pred_traj: np.ndarray = None
    pred_shape: List[Polygon] = field(default_factory=list)

    

    # Consistency check
    pred_traj_required_length: int = None

    # risk and collision
    risk: float = 0.
    risk2obj: Dict[int, float] = field(default_factory=dict)
    collision: bool = False
    collision2obj: Dict[int, bool] = field(default_factory=dict)
    pred_collision: bool = False
    pred_collision2obj: Dict[int, bool] = field(default_factory=dict)
    danger_flag : bool = False

    # debug info
    risk_curve: np.ndarray = None
    distance2ego: list = field(default_factory=list)
    damage: np.ndarray = None

    def _check_and_set_after_init(self):
        assert self.pred_traj_required_length != None, "prediction trajectory required length is not set"
        assert self.type in ['car', 'customized','pedestrian','bicycle','roadedge'], f"Invalid vehicle type: {self.type}"
        if self.type == 'car':
            if self.fix_damage == None: 
                self.fix_damage = 1.
                # logging.info(f"[Veh] Set default fix_damage={self.fix_damage} for vehicle name={self.name}")
            if self.length == None: 
                self.length = 4.5
                logging.info(f"[Veh] Set default length={self.length} for vehicle name={self.name}")
            if self.width == None:
                self.width = 1.8
                logging.info(f"[Veh] Set default width={self.width} for vehicle name={self.name}")
        else:
            assert self.length != None, f"Length must be specified for customized vehicle name={self.name}"
            assert self.width != None, f"Width must be specified for customized vehicle name={self.name}"
            assert self.fix_damage != None, f"Fix_damage must be specified for customized vehicle name={self.name}"
        if not isinstance(self.pred_traj, type(None)):  # if input prediction, then its length should be the same as required
            assert self.pred_traj.shape == (self.pred_traj_required_length, 4), f"the input predicted trajectory size should be ({self.pred_traj_required_length}, 4), now is {self.pred_traj.shape}"
        else:  # or, the basic states should be set for prediction
            assert self.x != None, f"x must be specified for vehicle name={self.name}"
            assert self.y != None, f"y must be specified for vehicle name={self.name}"
            assert self.phi != None, f"phi must be specified for vehicle name={self.name}"
            assert self.u != None, f"v must be specified for vehicle name={self.name}"
    
    def _trajectory_prediction(self, delta_t, steps):
        if not isinstance(self.pred_traj, type(None)):
            return
        self.pred_traj = np.array([self.x, self.y, self.phi, self.u])
        state = [self.x, self.y, self.phi, self.u, self.v, self.w]#初始状态
        for i in range(steps):
            state = dynamic_model(state, [self.st, self.a], delta_t)#使用单轨模型计算状态

            self.pred_traj = np.vstack([self.pred_traj, state[:4]])#存储轨迹

    def _shape_prediction(self):
        if isinstance(self.pred_traj, type(None)):
            raise ValueError("Predicted trajectory is not available")
        l, w = self.length/2, self.width/2
        rotation_mtx_x = np.vstack([np.cos(self.pred_traj[:,2]), np.sin(self.pred_traj[:,2])]).T
        rotation_mtx_y = rotation_mtx_x[:,[1,0]]
        points = np.hstack([
            (self.pred_traj[:,0] + (np.array([-l, -w * -1]) * rotation_mtx_x).sum(1)).reshape(-1,1),  # right bottom x
            (self.pred_traj[:,1] + (np.array([-l, -w * 1]) * rotation_mtx_y).sum(1)).reshape(-1,1),  # right bottom y
            (self.pred_traj[:,0] + (np.array([l, -w * -1]) * rotation_mtx_x).sum(1)).reshape(-1,1),  # right top x
            (self.pred_traj[:,1] + (np.array([l, -w * 1]) * rotation_mtx_y).sum(1)).reshape(-1,1),  # right top y
            (self.pred_traj[:,0] + (np.array([l, w * -1]) * rotation_mtx_x).sum(1)).reshape(-1,1),  # left top x
            (self.pred_traj[:,1] + (np.array([l, w * 1]) * rotation_mtx_y).sum(1)).reshape(-1,1),  # left top y
            (self.pred_traj[:,0] + (np.array([-l, w * -1]) * rotation_mtx_x).sum(1)).reshape(-1,1),  # left bottom x
            (self.pred_traj[:,1] + (np.array([-l, w * 1]) * rotation_mtx_y).sum(1)).reshape(-1,1),  # left bottom y
        ])
        
        for i in range(self.pred_traj.shape[0]):
            self.pred_shape.append(Polygon(points[i].reshape(4, 2)))
    
    def _prediction(self, delta_t, steps):
        self._trajectory_prediction(delta_t, steps)
        self._shape_prediction()

    def _getegotrajectory(self,ego_traj):
        self.pred_traj = ego_traj
        self._shape_prediction()




class SafetyResponder:#仿真减速度为实际减速的0.1倍
    def __init__(self, max_acc=3.0, min_brake=-1.0, max_brake=-5.0, ego=None, obj=None,danger = False):
        self.max_acc = max_acc
        self.min_brake = min_brake
        self.max_brake = max_brake
        self.ego = ego if ego is not None else Veh()
        self.obj = obj if obj is not None else Veh()
        self.ro = 0.5 #反应时间
        # self.lat_threshold = 1.2#根据横向距离判断是否需进行纵向安全判断
        
        self.lat_threshold = 2#根据横向距离判断是否需进行纵向安全判断


        if self.ego.width is not None:
            self.threshold = 0.5 * self.ego.width
        else:
            self.threshold = 0.5 * 1.8  # 设定一个默认值 2.0，如果没有设置 width
        
        self.rel_lat= 0
        self.rel_lon = 0

        self.lon_state = False
        self.lon_response = 0
        self.lon_safedis = None
        self.lon_currdis = None


        self.lat_state = False
        self.lat_safedis = None
        self.lat_currdis = None
        self.lat_response = 0

        self.danger = danger

    

#     vector<double> stationCalculate() {
#     std::vector<double> s_road;
#     s_road.resize(x_road.size());
#     s_road[0] = 0;
#     for (int i = 1; i < x_road.size(); i++){
#         s_road[i] = s_road[i - 1] + sqrt((x_road[i] - x_road[i - 1]) * (x_road[i] - x_road[i - 1]) + (y_road[i] - y_road[i - 1]) * (y_road[i] - y_road[i - 1]));
#     }

#     return s_road;
# }
        # nearestnum = nearestindex_veh(obstx,obsty);
        # abslateral = sqrt((obstx - x_road[nearestnum]) * (obstx - x_road[nearestnum]) + (obsty - y_road[nearestnum]) * (obsty - y_road[nearestnum]));
        # refvector[0] = obstx - x_road[nearestnum]; 
        # refvector[1] = obsty - y_road[nearestnum];
        # phivector[0] = x_road[nearestnum+1] - x_road[nearestnum]; 
        # phivector[1] = y_road[nearestnum+1] - y_road[nearestnum];
        # anglevector = refvector[0] * phivector[1] - refvector[1] * phivector[0];
        # if (anglevector == 0) { 
        #     q_lateral = 0.0; 
        # } else if (anglevector > 0) { 
        #     q_lateral = 1.0 * abslateral; 
        # } else { 
        #     q_lateral = -1.0 * abslateral; 
        # }
        # q_lateral = -q_lateral;

    def get_relative_position_sl(self):
       # 构建 s-l 坐标系
    
        dx = self.ego.pred_traj[1:, 0] - self.ego.pred_traj[:-1, 0]
        dy = self.ego.pred_traj[1:, 1] - self.ego.pred_traj[:-1, 1]
        ds = np.sqrt(dx ** 2 + dy ** 2)
        s_road = np.zeros(len(self.ego.pred_traj))
        s_road[1:] = np.cumsum(ds)

        # 计算最近点索引
        nearest_dist = float('inf')
        nearest_index = -1
        for i in range(len(self.ego.pred_traj)):
            dist = (self.obj.x - self.ego.pred_traj[i, 0]) ** 2 + (self.obj.y - self.ego.pred_traj[i, 1]) ** 2
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_index = i

        if nearest_index + 1 >= len(self.ego.pred_traj):
            nearest_index = len(self.ego.pred_traj) - 2  # 避免越界

        # 计算横向位置与纵向位置
        abslateral = math.sqrt((self.obj.x - self.ego.pred_traj[nearest_index, 0]) ** 2 + 
                            (self.obj.y - self.ego.pred_traj[nearest_index, 1]) ** 2)
        
        refvector = [self.obj.x - self.ego.pred_traj[nearest_index, 0], 
                    self.obj.y - self.ego.pred_traj[nearest_index, 1]]
        phivector = [self.ego.pred_traj[nearest_index + 1, 0] - self.ego.pred_traj[nearest_index, 0], 
                    self.ego.pred_traj[nearest_index + 1, 1] - self.ego.pred_traj[nearest_index, 1]]
        
        anglevector = refvector[0] * phivector[1] - refvector[1] * phivector[0]
        
        if anglevector == 0:
            q_lateral = 0.0
        elif anglevector > 0:
            q_lateral = 1.0 * abslateral
        else:
            q_lateral = -1.0 * abslateral

        # 设置 obj 的 p 和 s 坐标
        self.obj.p = -q_lateral  # 左边是 +，所以取反
        self.obj.s = s_road[nearest_index]





        # abslateral = math.sqrt((obstx - x_road[nearest_num]) ** 2 + (obsty - y_road[nearest_num]) ** 2)
    
        # refvector = [obstx - x_road[nearest_num], obsty - y_road[nearest_num]]
        # phivector = [x_road[nearest_num + 1] - x_road[nearest_num], y_road[nearest_num + 1] - y_road[nearest_num]]
        
        # anglevector = refvector[0] * phivector[1] - refvector[1] * phivector[0]
        
        # if anglevector == 0:
        #     q_lateral = 0.0
        # elif anglevector > 0:
        #     q_lateral = 1.0 * abslateral
        # else:
        #     q_lateral = -1.0 * abslateral
        
        # q_lateral = -q_lateral







        
    def get_relative_position(self):
        """
        判断目标车相对于自车的位置（8个方向）
        :param x_ego, y_ego: 自车位置
        :param phi_ego: 自车朝向角（单位：弧度）
        :param x_obj, y_obj: 目标车位置
        :param threshold: 距离容差（默认0.5米以内视为同一位置）
        :return: 字符串，表示目标车相对于自车的位置
        """
        x_obj = self.obj.pred_traj[0,0]
        y_obj = self.obj.pred_traj[0,1]

        x_ego = self.ego.pred_traj[0,0]
        y_ego = self.ego.pred_traj[0,1]
        phi_ego = self.ego.pred_traj[0,2]




        # 世界坐标下的相对位置向量
        dx = x_obj - x_ego
        dy = y_obj - y_ego

        

        # 坐标变换：旋转 -phi_ego，将其变换到自车坐标系
        x_rel = np.cos(phi_ego) * dx + np.sin(phi_ego) * dy
        y_rel = -np.sin(phi_ego) * dx + np.cos(phi_ego) * dy

        self.rel_lon = x_rel #纵向相对距离 前+
        self.rel_lat = y_rel #横向相对距离  左+
        

        # 定义角度阈值：π/8 = 22.5°
        angle = np.arctan2(y_rel, x_rel)

        if np.hypot(x_rel, y_rel) < self.threshold:
            return 'same position'

        # 八个象限判断
        if -np.pi/8 <= angle < np.pi/8:
            self.obj.rel_posi = "f" #前
        elif np.pi/8 <= angle < 3*np.pi/8:
            self.obj.rel_posi = "fl" #左前
        elif 3*np.pi/8 <= angle < 5*np.pi/8:
            self.obj.rel_posi = "l" #左
        elif 5*np.pi/8 <= angle < 7*np.pi/8:
            self.obj.rel_posi = "bl" #左后
        elif angle >= 7*np.pi/8 or angle < -7*np.pi/8:
            self.obj.rel_posi = "b" #后
        elif -7*np.pi/8 <= angle < -5*np.pi/8:
            self.obj.rel_posi = "br" #右后
        elif -5*np.pi/8 <= angle < -3*np.pi/8:
            self.obj.rel_posi = "r" #右
        elif -3*np.pi/8 <= angle < -np.pi/8:
            self.obj.rel_posi = "fr" #右前
        else:
            self.obj.rel_posi = "unknow" #未知

   

    def longitudinal_safety_threshold(self):
        """
        rssk计算纵向安全距离
        """
        use_sl = True
        # print('危险车辆：',self.obj.type,self.obj.phi,'危险系数： ',self.ego.risk,self.ego.phi)

   


        # print("紧急状态 ：",self.danger)

        if use_sl:
            #sl相对位置判断
            # print('当前s： ',self.obj.s,'当前p： ',self.obj.p)

            if abs(math.sin(self.ego.phi) - math.sin(self.obj.phi))> math.sin(np.pi/6):#交叉情况下扩展横向检测范围
                self.lat_threshold = 3.5
            
            # print( abs(math.sin(self.ego.phi) - math.sin(self.obj.phi)) > math.sin(np.pi/6),self.lat_threshold)

            if self.obj.s > 0.01 and self.obj.type in ['car','pedestrian','bicycle'] or (self.danger and self.obj.s > 0 and self.obj.s < 10): #判断前后             
                if abs(self.obj.p) < self.lat_threshold or (self.danger and abs(self.obj.p) < 10):  #判断车道  根据航向设置阈值
                    self.lon_currdis = self.obj.s - 0.5* self.ego.length-0.5*self.obj.length  #sl纵向相对距离中心距离
                    # self.lon_currdis = self.rel_lon - 0.5*self.ego.length  #xy两者几何中心距离

                    if self.lon_currdis> -  self.ego.length and self.lon_currdis<0:
                        self.lon_currdis = 0

                #速度投影 1.将旁车车体坐标系速度转换到全局坐标系下 2.使用全局坐标系速度计算在自车纵向方向上的速度分量
                    vx_w = self.obj.u * math.cos(self.obj.phi) - self.obj.v * math.sin(self.obj.phi)
                    vy_w = self.obj.u * math.sin(self.obj.phi) + self.obj.v * math.cos(self.obj.phi)


                    vf=vx_w*math.cos(self.ego.phi)+vy_w*math.sin(self.ego.phi)#(旁车在自车纵向方向上的投影速度)
                    vr = math.sqrt(self.ego.u**2+self.ego.v**2)#自车纵向速度（不考虑侧滑时纵向速度等于和速度）
                    # print("vf",vf,'vr',vr)
                    self.lon_safedis = vr*self.ro + 0.5 * self.max_acc * self.ro**2 + (vr + self.ro*self.max_acc)**2/(2*abs(self.min_brake)) - vf**2 / (2*abs(self.max_brake)) #rss lonsafedis
                    # self.lon_safedis += 0.5*self.ego.length + 0.5 *self.obj.length
                    if self.lon_safedis < 0:
                        self.lon_safedis = 0
                else:
                    # print("障碍物不在同一车道")
                    self.lon_state = True
            else:
                # print("障碍物在自车后方")
                self.lon_state = True

            #sl相对位置判断 5_19
            # print('当前s： ',self.obj.s,'当前p： ',self.obj.p)

            # if abs(math.sin(self.ego.phi) - math.sin(self.obj.phi))> math.sin(np.pi/6):#交叉情况下扩展横向检测范围
            #     self.lat_threshold = 3.5
            # print( abs(math.sin(self.ego.phi) - math.sin(self.obj.phi)) > math.sin(np.pi/6),self.lat_threshold)

            # if self.obj.s > 0 and self.obj.type in ['car','pedestrian','bicycle'] or (self.ego.risk > 0.25 and self.obj.s>0): #判断前后             
            #     if abs(self.obj.p) < self.lat_threshold or (self.ego.risk > 0.25 and self.obj.s>0):  #判断车道  根据航向设置阈值
            #         self.lon_currdis = self.obj.s - 0.5*self.ego.length-0.5*self.obj.length  #sl纵向相对距离中心距离
            #         # self.lon_currdis = self.rel_lon - 0.5*self.ego.length  #xy两者几何中心距离

            #         if self.lon_currdis> - 0.5 * self.ego.length and self.lon_currdis<0:
            #             self.lon_currdis = 0

            #     #速度投影 1.将旁车车体坐标系速度转换到全局坐标系下 2.使用全局坐标系速度计算在自车纵向方向上的速度分量
            #         vx_w = self.obj.u * math.cos(self.obj.phi) - self.obj.v * math.sin(self.obj.phi)
            #         vy_w = self.obj.u * math.sin(self.obj.phi) + self.obj.v * math.cos(self.obj.phi)


            #         vf=vx_w*math.cos(self.ego.phi)+vy_w*math.sin(self.ego.phi)#(旁车在自车纵向方向上的投影速度)
            #         vr = math.sqrt(self.ego.u**2+self.ego.v**2)#自车纵向速度（不考虑侧滑时纵向速度等于和速度）
            #         # print("vf",vf,'vr',vr)
            #         self.lon_safedis = vr*self.ro + 0.5 * self.max_acc * self.ro**2 + (vr + self.ro*self.max_acc)**2/(2*abs(self.min_brake)) - vf**2 / (2*abs(self.max_brake)) #rss lonsafedis
            #         # self.lon_safedis += 0.5*self.ego.length + 0.5 *self.obj.length
            #         if self.lon_safedis < 0:
            #             self.lon_safedis = 0
            #     else:
            #         print("障碍物不在同一车道")
            #         self.lon_state = True
            # else:
            #     print("障碍物在自车后方")
            #     self.lon_state = True
        else:
            #xy相对位置判断
            if self.obj.rel_posi in ["f",'fl,','fr'] and self.obj.type in ['car','pedestrian','bicycle'] or self.ego.risk > 0.5: #自车前方障碍物 （通过角度与自车航向判断旁车位置）             
                # self.lon_currdis = self.obj.distance2ego[0]#两者外轮廓最近距离
                if abs(self.rel_lat) < self.lat_threshold  or self.ego.risk > 0.5 :  #同车道
                    self.lon_currdis = self.rel_lon - 0.5*self.ego.length  #两者几何中心距离
                    if self.lon_currdis> - 0.5 * self.ego.length and self.lon_currdis<0:
                        self.lon_currdis = 0

                #速度投影 1.将旁车车体坐标系速度转换到全局坐标系下 2.使用全局坐标系速度计算在自车纵向方向上的速度分量
                    vx_w = self.obj.u * math.cos(self.obj.phi) - self.obj.v * math.sin(self.obj.phi)
                    vy_w = self.obj.u * math.sin(self.obj.phi) + self.obj.v * math.cos(self.obj.phi)


                    vf=vx_w*math.cos(self.ego.phi)+vy_w*math.sin(self.ego.phi)#(旁车在自车纵向方向上的投影速度)
                    vr = math.sqrt(self.ego.u**2+self.ego.v**2)#自车纵向速度（不考虑侧滑时纵向速度等于和速度）
                    # print("vf",vf,'vr',vr)
                    # print("ego phi",self.ego.phi)
                    self.lon_safedis = vr*self.ro + 0.5 * self.max_acc * self.ro**2 + (vr + self.ro*self.max_acc)**2/(2*abs(self.min_brake)) - vf**2 / (2*abs(self.max_brake)) #rss lonsafedis
                    if self.lon_safedis < 0:
                        self.lon_safedis = 0
                else:
                    # print("障碍物不在同一车道")
                    self.lon_state = True

            else:
                # print("障碍物在自车后方")
                self.lon_state = True

        


        

        


    def compute_response(self):
        """
        判断是否需要制动，并计算响应制动强度
        返回值: 制动强度（负值），0 表示无需响应
        """

        if self.obj.name == None:
            self.lon_state = True
            self.lon_response = 0
        else:

            self.get_relative_position() #计算相对位置关系
            self.get_relative_position_sl()
            
            #纵向评估
            self.longitudinal_safety_threshold()
            self.lastrisk = self.ego.risk
            if  not self.lon_state:
                if self.lon_currdis > self.lon_safedis + self.ro * self.ego.u:
                    self.lon_state = True
                    self.lon_response = 0
                elif self.lon_currdis > self.lon_safedis:
                    self.lon_response = self.min_brake
                    if self.danger:
                        self.lon_response = self.max_brake

                else:
                    self.lon_response =  self.max_brake

            
            
            # print("相对横向距离： ",self.rel_lat,"相对纵向距离： ",self.rel_lon)
            # print("sl横向： ",self.obj.p,"sl纵向： ",self.obj.s)

            # print('当前纵向距离：',self.lon_currdis,'纵向安全距离： ',self.lon_safedis)
            # print("纵向相应为： ",self.lon_response)
            # print()
            # print()


     

        
    def display_result(self):


        print("=== Safety Analysis Result ===")
        print("Lat_rel: ",self.rel_lat,"Lon_rel: ",self.rel_lon)


        print("Lon Safe distance is ",self.lon_safedis)
        print("current dis is ",self.lon_currdis)
        print("lon response is ", self.lon_response)

       
        if not self.lon_state:
            print("Lon Safe distance is ",self.lon_safedis)
            print("current dis is ",self.lon_currdis)
            print("lon response is ", self.lon_response)










@dataclass
class PODAR:
    # style: str = 'normal'
    # style: str = 'aggr'
    style: str = 'cons'


    horizon: float = 4.  #总预测时长
    delta_t: float = 0.1 #时间步长
    pred_time_series: np.ndarray = None
    pred_step: int = None
    ego: Veh = field(default_factory=Veh)
    obj: List[Veh] = field(default_factory=list)
    lon_res: float = None
    lat_res: float = None
    riskchangethreshord: float = 0.05
    highrisk : float  = 0.25
    lowrisk: float = 0.15
    danger : bool  = False
    

    def __post_init__(self):
        # print('pred_horizon time is ',self.horizon,self.delta_t)
        # if self.horizon != 4. or self.delta_t != 0.1:
        #     logging.info(f"[PDOAR] Use customrized prediction method: horizon={self.horizon} and delta_t={self.delta_t}")
        # else:
        #     logging.info(f"[PDOAR] Use default prediction method: horizon={self.horizon} and delta_t={self.delta_t}")
        self.pred_time_series = np.linspace(0, self.horizon, int(self.horizon / self.delta_t) + 1)
        self.pred_step = int(self.horizon / self.delta_t)

    def set_ego(self, **kwargs):#自车创建
        if not 'name' in kwargs.keys():
            logging.warning("it would be better to asign a name to ego vehicle")
        kwargs.update({'pred_traj_required_length': self.pred_step + 1})
        _o = Veh(**kwargs)
        _o._check_and_set_after_init()
        self.ego = _o
    
    def add_obj(self, **kwargs):#目标车辆添加
        assert self.ego.type != None, 'Please add a ego vehicle first'
        if not 'name' in kwargs.keys():
            logging.warning("[PDOAR] It would be better to asign a name to object vehicle")
        kwargs.update({'pred_traj_required_length': self.pred_step + 1})
        _o = Veh(**kwargs)
        _o._check_and_set_after_init()
        _o.id = len(self.obj)
        self.obj.append(_o)

    def estimate_risk(self,ego_traj=None):#风险估计
        if ego_traj is None or np.isnan(ego_traj[:, 0]).any():
            self.ego._prediction(self.delta_t, self.pred_step)#基于动力学模型自车轨迹预测
        else:
            self.ego._getegotrajectory(ego_traj)#基于轨迹规划的自车轨迹
        # print("动力学模型traj",self.ego.pred_traj)

        if len(self.obj) == 0:
            logging.info("[PDOAR] No object vehicle is added")
            return 0            

        for obj in self.obj:
            
            if len(obj.distance2ego) == 0:
                obj._prediction(self.delta_t, self.pred_step)#障碍物轨迹预测
                assert self.ego.pred_traj.shape[0] == obj.pred_traj.shape[0], "Prediction steps are not consistent"
                for i in range(self.pred_step + 1):
                    obj.distance2ego.append(self.ego.pred_shape[i].distance(obj.pred_shape[i]))#主车在第i步的形状，与旁车在第i步的形状之间的距离，计算两个轮廓之间最近点点距离
                obj.distance2ego = np.array(obj.distance2ego)
                obj.distance2ego[obj.distance2ego < 0] = 0

                

            moving_dir = -np.sign(np.diff(np.around(obj.distance2ego, 3)))
            moving_dir = np.concatenate((moving_dir, [moving_dir[-1]]))
            moving_dir[moving_dir == 0] = 1
            moving_dir[moving_dir < 0] = 0


            delta_v = np.abs(self.ego.pred_traj[:,3] - obj.pred_traj[:,3]) * moving_dir
            
            abs_v = (self.ego.pred_traj[:,3] + obj.pred_traj[:,3]) * moving_dir

            #风格参数设置
            if self.style == 'normal':
                # normal driving configuration, R2=0.942, Newton a=-5m/s2
                alpha, beta, gamma, A, B = 0.005, 1.1, 0.6, 1.0, 2.3
            elif self.style == 'aggr':
                # aggressive driving configuration, R2=0.922, Newton a=-6m/s2
                alpha, beta, gamma, A, B = 0.004, 1.1, 0.6, 1.2, 2.4
            elif self.style == 'cons':
                # # conservative driving configuration, R2=0.940, Newton a=-4m/s2
                # alpha, beta, gamma, A, B = 0.006, 1.1, 0.6, 0.8, 2.2
                alpha, beta, gamma, A, B = 0.006, 1.1, 0.6, 0.8, 1.8

            
            #损伤风险计算
            v_ = (gamma * delta_v + (1-gamma) * abs_v) ** 2 * alpha 
            obj.damage = (self.ego.fix_damage + obj.fix_damage) * np.log(v_ + beta)

            omega_t = np.exp(-1 * self.pred_time_series * A) #时间衰减函数
            omega_d = np.exp(-1 * obj.distance2ego * B) #距离衰减函数

            obj.risk_curve = omega_t * omega_d * obj.damage

            obj.risk = np.max(obj.risk_curve)

            #碰撞判断
            if obj.distance2ego[0] == 0: # collision
                obj.collision = True
            elif 0 in obj.distance2ego: # pred collision
                obj.pred_collision = True

            self.ego.risk2obj[str(obj.id) + ":" + str(obj.name)] = obj.risk
            self.ego.collision2obj[str(obj.id) + ":" + str(obj.name)] = obj.collision
            self.ego.pred_collision2obj[str(obj.id) + ":" + str(obj.name)] = obj.pred_collision

       
        self.ego.risk = np.max(list(self.ego.risk2obj.values()))
        self.ego.collision = 1 if np.max(list(self.ego.collision2obj.values())) >0 else 0
        self.ego.pred_collision = 1 if np.max(list(self.ego.pred_collision2obj.values())) >0 else 0
        return self.ego.risk
    
    def get_risk_in_stru(self): #风险结果输出
        _ = []
        for ov in self.obj:
            _.append(
                [self.ego.u, np.sqrt(ov.x ** 2 + ov.y ** 2), ov.type, ov.phi, ov.phi / np.pi * 180,
                 ov.x, ov.y, ov.u, ov.u * 3.6, ov.risk, ov.risk_curve, ov.pred_collision])
            # columns=['ego_speed', 'r', 'type', 'phi', 'phi_de', 'x', 'y', 'ov_speed', 'ov_speed_km', 'risk', 'risk_curve', 'pred_collision']
        return _
    
    def get_max_risk_obj_info(self):
        """
        输出风险最高的目标车辆及自车当前状态，并返回这两个对象
        """
        if not self.obj:
            # print("[PODAR] No object vehicles present.")
            return None, None  # 如果没有目标车，返回 None

        # 找到风险最高的目标车
        max_risk_obj = max(self.obj, key=lambda o: o.risk)
        
        # 返回自车和最大风险目标车对象
        return self.ego, max_risk_obj
    
    def parse_frame_to_podar(self,frame=None,ego_traj=None,lastres= None):
        """
            用于将 DataFrame 中的 ego 与旁车信息，初始化到 podar 对象中。

            参数:
                frame: pandas.DataFrame，行索引为对象名称（如 'ego', 'car1', ...）
                podar: 具有 set_ego(name, type, x, y, u, v, phi, length, width, a, st)
                    与 add_obj(name, type, x, y, u, v, phi, length, width, a, fix_damage) 方法
        """
        # 1. 初始化 Ego
        ego = frame['ego']
        phi_ego = float(ego['yaw'])
        v_ego_total = float(ego['v'])
        u_ego, v_ego = v_split(v_ego_total, phi_ego)
        self.set_ego(
            name='ego',
            type='car',
            x=float(ego['x']),
            y=float(ego['y']),
            u=u_ego,
            v=v_ego,
            phi=phi_ego,
            length=float(ego['length']),
            width=float(ego['width']+0.4),
            a=float(ego['a']),
            st=0.0  # 如果有转向角，可替换此处
        )

        # 2. 初始化其他车辆
        damage_map = {
                'car': 1,
                'bicycle': 1.5,
                'pedestrian': 2,
                'obstacle': 0
        }

        for name, row in frame.drop(index=['ego']).iterrows():
            # 判断对象类型
            if name.startswith('car'):
                obj_type = 'car'
            elif name.startswith('bicycle'):
                obj_type = 'bicycle'
            elif name.startswith('pedestrian'):
                obj_type = 'pedestrian'
            else:
                # 未知类型，跳过或默认为普通障碍物
                obj_type = 'obstacle'

            phi = float(row['yaw'])
            v_total = float(row['v'])
            u, v = v_split(v_total, phi)
            # 调用 podar 的 add_obj，fix_damage 对车辆有效，可根据类型调整
            self.add_obj(
                name=name,
                type=obj_type,
                x=float(row['x']),
                y=float(row['y']),
                u=u,
                v=v,
                phi=phi,
                length=float(row['length']),
                width=float(row['width']),
                a=self.limit(float(row['a']),-5,5),
                # a=(float(row['a'])),
                fix_damage=damage_map.get(obj_type, 0)
            )
        
        
        risk = self.estimate_risk(ego_traj)

        

        if lastres is not None:
            # print("危险状态标志： ",lastres.risk - risk < self.riskchangethreshord,risk > self.lowrisk,lastres.danger_flag)
            if risk > self.highrisk:
                self.danger = True
            elif lastres.risk - risk < self.riskchangethreshord and risk > self.lowrisk  and lastres.danger_flag:
                self.danger = True
            else:
                self.danger = False
                

        # print("紧急状态： ",self.danger)

        ego,max_risk_obj = self.get_max_risk_obj_info()


        saferes = SafetyResponder(ego=ego,obj=max_risk_obj,danger= self.danger)
        saferes.compute_response()
        self.lon_res = saferes.lon_response
        self.lat_res = saferes.lat_response
        # saferes.display_result()

    def parse_frame_to_podar_fragment(self, frame=None, ego_traj=None, lastres=None, ego_frame = None):
        """
            用于将 DataFrame 中的 ego 与旁车信息，初始化到 podar 对象中。

            参数:
                frame: pandas.DataFrame，行索引为对象名称（如 'ego', 'car1', ...）
                podar: 具有 set_ego(name, type, x, y, u, v, phi, length, width, a, st)
                    与 add_obj(name, type, x, y, u, v, phi, length, width, a, fix_damage) 方法
        """
        # 1. 初始化 Ego
        ego = ego_frame['ego']
        phi_ego = float(ego['yaw'])
        v_ego_total = float(ego['v'])
        u_ego, v_ego = v_split(v_ego_total, phi_ego)
        self.set_ego(
            name='ego',
            type='car',
            x=float(ego['x']),
            y=float(ego['y']),
            u=u_ego,
            v=v_ego,
            phi=phi_ego,
            length=float(ego['length']),
            width=float(ego['width'] + 0.4),  # 这里宽度增加了 0.4
            a=float(ego['a']),
            st=0.0  # 如果有转向角，可替换此处
        )

        # 2. 初始化其他车辆
        damage_map = {
            'car': 1,
            'bicycle': 1.5,
            'pedestrian': 2,
            'obstacle': 0
        }

        # 车辆类型判断：通过长款来判断
        def determine_vehicle_type(length, width):
            if length > 4.0:  # 假设车长大于5米是车
                return 'car'
            elif length <= 4.0 and length > 1.5:  # 如果车长在1.5米到5米之间，可以是自行车
                return 'bicycle'
            elif length <= 1.5:  # 如果车长小于1.5米，判断为行人
                return 'pedestrian'
            else:
                return 'obstacle'  # 其他情况，默认为障碍物

        # 3. 处理每一辆车的具体信息
        for name, row in frame.items():
            # 根据车辆的长宽来判断类型
            obj_type = determine_vehicle_type(float(row['length']), float(row['width']))

            phi = float(row['yaw'])
            v_total = float(row['v'])
            u, v = v_split(v_total, phi)
            
            # 调用 podar 的 add_obj，fix_damage 对车辆有效，可根据类型调整
            self.add_obj(
                name=str(name),  # 将索引值转为字符串
                type=obj_type,
                x=float(row['x']),
                y=float(row['y']),
                u=u,
                v=v,
                phi=phi,
                length=float(row['length']),
                width=float(row['width']),
                a=self.limit(float(row['a']), -5, 5),
                fix_damage=damage_map.get(obj_type, 0)
            )

        # 4. 计算风险
        risk = self.estimate_risk(ego_traj)

        # 5. 判断危险状态
        if lastres is not None:
            # print("危险状态标志： ", lastres.risk - risk < self.riskchangethreshord, risk > self.lowrisk, lastres.danger_flag)
            if risk > self.highrisk:
                self.danger = True
            elif lastres.risk - risk < self.riskchangethreshord and risk > self.lowrisk and lastres.danger_flag:
                self.danger = True
            else:
                self.danger = False

        # 6. 获取最大风险对象
        ego, max_risk_obj = self.get_max_risk_obj_info()
        saferes = SafetyResponder(ego=ego, obj=max_risk_obj, danger=self.danger)
        saferes.compute_response()

        # 7. 保存车辆响应
        self.lon_res = saferes.lon_response
        self.lat_res = saferes.lat_response
        # saferes.display_result()


    def limit(self,para,low,up):
        if para >= up:
            return up
        elif para <= low:
            return low
        else:
            return para





 

def _draw_rotate_rec(veh, ec, fc: str='white'):
        diff_x, diff_y = rotation(-veh.length / 2, -veh.width / 2, veh.phi)
        rec = patches.Rectangle((veh.x + diff_x, veh.y + diff_y), veh.length, veh.width, 
            angle=veh.phi/np.pi*180, ec=ec, fc=fc)
        return rec


def render_moment(frame: PODAR, ax):
    # define colors
    art_collection = []
    cmap = plt.cm.jet
    mycmap = cmap.from_list('Custom cmap', 
        [[0 / 255, 255 / 255, 0 / 255], [255 / 255, 255 / 255, 0 / 255], [255 / 255, 0 / 255, 0 / 255]], cmap.N) #绿色（低风险）→ 黄色（中等）→ 红色（高风险）
    c_norm = colors.Normalize(vmin=0.00001, vmax=0.105, clip=True)

    x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
    
    #主车绘制
    rec_handle = _draw_rotate_rec(frame.ego, 'blue', mycmap(c_norm(frame.ego.risk)))#主车轮廓颜色
    rec_handle.set_linewidth(2.5)  # 加粗轮廓线
    art_collection += [rec_handle]
    ax.add_patch(rec_handle)  
    x_min, x_max, y_min, y_max = np.min([x_min, frame.ego.x]), np.max([x_max, frame.ego.x]), np.min([y_min, frame.ego.y]), np.max([y_max, frame.ego.y])
    # art_collection += ax.text(frame.ego.x, frame.ego.y, 'ego, R={:.2f}, v={:.1f}'.format(0, frame.ego.risk, frame.ego.u), c="red", fontsize=12).findobj() #添加文字标签
    # ax.scatter(frame.ego.x, frame.ego.y, c='black', s=5)
    art_collection += ax.plot(frame.ego.pred_traj[:,0], frame.ego.pred_traj[:,1], linestyle='--', c='darkorange') #预测轨迹线

    #旁车绘制
    i = 0
    for _veh in frame.obj:
        nameid = str(_veh.id) + ":" + str(_veh.name)
        if frame.ego.risk2obj[nameid] > 0.01:
            # art_collection += ax.text(_veh.x, _veh.y, 'R={:.2f}'.format(frame.ego.risk2obj[nameid]), c=C_RISK_LABEL, fontsize=12).findobj()
            art_collection += ax.text(_veh.x, _veh.y, 'R={:.2f}'.format(frame.ego.risk2obj[nameid]), c="black", fontsize=8).findobj()

            # art_collection += ax.text(_veh.x, _veh.y, '{}, R={:.2f}'.format(nameid, frame.ego.risk2obj[nameid]), c="blue", fontsize=12).findobj()
            ec_ = 'blue'
        else:
            # art_collection += ax.text(_veh.x, _veh.y, '{}'.format(nameid), c="blue", fontsize=12).findobj()
            ec_ = 'black'
        rec_handle1 = _draw_rotate_rec(_veh, ec_, mycmap(c_norm(frame.ego.risk2obj[nameid])))
        art_collection += [rec_handle1]
        ax.add_patch(rec_handle1)    
        # ax.scatter(_veh.x, _veh.y, c='ec_', s=5)
        art_collection += ax.plot(_veh.pred_traj[:,0], _veh.pred_traj[:,1], linestyle='--', c='darkgray', linewidth=1)
        x_min, x_max, y_min, y_max = np.min([x_min, _veh.x]), np.max([x_max, _veh.x]), np.min([y_min, _veh.y]), np.max([y_max, _veh.y])
        i += 1

    # plt.xlim(frame.ego.x - 60, frame.ego.x + 60)
    # plt.ylim(frame.ego.y - 60, frame.ego.y + 60)
    # plt.xlim(741700, 741800)
    # plt.ylim(y_min - 10, y_max + 10)
    # print('x_max, x_min, y_max, y_min =', x_max, x_min, y_max, y_min)
    # print('actual xylim=', plt.gca().get_xlim(), plt.gca().get_ylim())
    plt.axis('equal')

    return art_collection


def draw_InD_cases(id, data_path, img_path, fold_path, trackid=None):
    from PIL import Image
    import logging
    import matplotlib.pyplot as plt
    import pandas as pd
    from tqdm import tqdm
    plt.rcParams['font.family'] = ['Times New Roman']

    data_path = data_path
    im = Image.open(img_path)

    data = pd.read_csv(data_path)

    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot()

    def single_draw(data, ego_id):    
        podar = PODAR()
        ego_info = data[data.trackId==ego_id].iloc[0, :]
        podar.set_ego(type='customized',name=int(ego_info.trackId), x=ego_info.xCenter, y=ego_info.yCenter, u=ego_info.lonVelocity, phi=ego_info.heading / 180 * np.pi, width=ego_info.width, length=ego_info.length, fix_damage=1.)
        for _, info in data.iterrows():
            if info.trackId == ego_id:
                ...
            else:
                podar.add_obj(type='customized', name=int(info.trackId), x=info.xCenter, y=info.yCenter, u=info.lonVelocity, phi=info.heading / 180 * np.pi, width=info.width, length=info.length, fix_damage=1.)
            # print(info.xCenter, info.yCenter, info.heading)
        podar.estimate_risk()
        
        art_collection = render_moment(podar, ax)

        return art_collection

    ego_id = id
    frames = data[data["trackId"] == ego_id]["frame"].unique()

    coll = []
    i = 0
    for frame_id in tqdm(frames, desc="frame_id", total=frames.shape[0]): #[40 + frames[0]]: #tqdm(frames, desc="frame_id", total=frames.shape[0]):  #[120 + frames[0]]:
        if i % 1 != 0:
            i += 1
            continue
        data_frame = data[data["frame"] == frame_id]
        art_collection = single_draw(data_frame, ego_id)
        plt.title("frame: {}, ego_id: {}".format(data_frame.frame.values[0], ego_id))
        if trackid == 0:
            art_collection += plt.imshow(im, extent=(0, im.width/(im.width/196), -im.height/(im.width/196), 0)).findobj()
        else:
            art_collection += plt.imshow(im, extent=(0, im.width/(im.width/114), -im.height/(im.width/114), 0)).findobj()
        
        # plt.savefig(fold_path + "\id_{}\{}.png".format(ego_id, frame_id), dpi=300)

        i += 1
        coll.append(art_collection)

    ani = animation.ArtistAnimation(fig=fig, artists=coll, interval=100)
    # print('saving')
    ani.save(fold_path + "\id_{}.mp4".format(ego_id), writer='ffmpeg', dpi=500)
    # ani.save(fold_path + "\id_{}.gif".format(ego_id), writer='pillow', dpi=100)
    # print('Done')


def run_InD_cases():    
    data_path = r"Ind_Dataset\17_tracks.csv"
    img_path = r"Ind_Dataset\17_background.png"
    fold_path = r"gifs_videos"
    trackid = 17
    id= 44
    draw_InD_cases(id, data_path, img_path, fold_path, trackid)

    data_path = r"Ind_Dataset\00_tracks.csv"
    img_path = r"Ind_Dataset\00_background.png"
    fold_path = r"gifs_videos"
    trackid = 0
    id = 354
    draw_InD_cases(id, data_path, img_path, fold_path, trackid)

    id = 226
    draw_InD_cases(id, data_path, img_path, fold_path, trackid)


class TrajectoryPlanner:
    def __init__(self, num_change=21, num_straight=21):
        """
        num_change: 换道段采样点数
        num_straight: 直行段或纯直行采样点数
        """
        self.num_change   = num_change
        self.num_straight = num_straight
        

    @staticmethod
    def derivative_coeffs(coeffs):
        return np.array([i * coeffs[i] for i in range(1, len(coeffs))], dtype=float)

    @staticmethod
    def compute_quintic_coeffs(p0, v0, a0, pT, vT, aT, T):
        M = np.array([
            [1,     0,      0,         0,        0,         0],
            [0,     1,      0,         0,        0,         0],
            [0,     0,      2,         0,        0,         0],
            [1,     T,    T**2,      T**3,     T**4,      T**5],
            [0,     1,    2*T,     3*T**2,   4*T**3,    5*T**4],
            [0,     0,      2,     6*T,     12*T**2,   20*T**3],
        ], dtype=float)
        b = np.array([p0, v0, a0, pT, vT, aT], dtype=float)
        return np.linalg.solve(M, b)

    @staticmethod
    def compute_cubic_coeffs(p0, v0, pT, vT, T):
        M = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, T, T**2, T**3],
            [0, 1, 2*T, 3*T**2]
        ], dtype=float)
        b = np.array([p0, v0, pT, vT], dtype=float)
        return np.linalg.solve(M, b)

    @staticmethod
    def sample_quintic_trajectory(coeffs, T, num_samples):
        t = np.linspace(0, T, num_samples)
        powers = np.vstack([t**i for i in range(len(coeffs))]).T
        pos = powers.dot(coeffs)
        dcoeffs = TrajectoryPlanner.derivative_coeffs(coeffs)
        dpowers = np.vstack([t**i for i in range(len(dcoeffs))]).T
        vel = dpowers.dot(dcoeffs)
        return pos, vel

    @staticmethod
    def sample_cubic_trajectory(coeffs, T, num_samples):
        t = np.linspace(0, T, num_samples)
        pos = coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + coeffs[3]*t**3
        dcoeffs = TrajectoryPlanner.derivative_coeffs(coeffs)
        vel = dcoeffs[0] + dcoeffs[1]*t + dcoeffs[2]*t**2
        return pos, vel

    
    def _plan_lane_change(self, start, mid, end):
        # start, mid, end 都是 (x, y, phi)
        p0_xy, phi0 = np.array(start[:2]), start[2]
        pT_xy, phiT = np.array(mid[:2]),   mid[2]
        pE_xy, phiE = np.array(end[:2]),   end[2]

        # print("p0_xy : ",p0_xy)
        # print("phi_0 : ",phi0,"phi_T: ",phiT,"phi_E ",phiE)


 


        v0_xy = np.array([start[3]*math.cos(phi0), start[3]*math.sin(phi0)])
        vT_xy = np.array([mid[3]*math.cos(phiT), mid[3]*math.sin(phiT)])
        vE_xy = np.array([end[3]*math.cos(phiE), end[3]*math.sin(phiE)])




        a0_xy = np.zeros(2)
        aT_xy = np.zeros(2)
        
        # 时间划分
        T_total  = np.linalg.norm(pE_xy - p0_xy) / np.mean(mid[3]+end[3])
        
        T_change = np.linalg.norm(pT_xy - p0_xy) / np.mean(start[3]+mid[3])
        
        T_remain = T_total - T_change
    
      
        # print("规划时间： ",T_total,T_change,T_remain)

        if T_remain <= 0:
            # print("规划退化三次多项式")
            return self._plan_straight(start,end)
        
        
            

        # 第一段：五次多项式
        cx1 = self.compute_quintic_coeffs(p0_xy[0], v0_xy[0], a0_xy[0],
                                          pT_xy[0], vT_xy[0], aT_xy[0], T_change)
        cy1 = self.compute_quintic_coeffs(p0_xy[1], v0_xy[1], a0_xy[1],
                                          pT_xy[1], vT_xy[1], aT_xy[1], T_change)
        xs1, vx1 = self.sample_quintic_trajectory(cx1, T_change, self.num_change)
        ys1, vy1 = self.sample_quintic_trajectory(cy1, T_change, self.num_change)

        # 第二段：三次延伸
        cx2 = self.compute_cubic_coeffs(pT_xy[0], vT_xy[0], pE_xy[0], vE_xy[0], T_remain)
        cy2 = self.compute_cubic_coeffs(pT_xy[1], vT_xy[1], pE_xy[1], vE_xy[1], T_remain)
        xs2, vx2 = self.sample_cubic_trajectory(cx2, T_remain, self.num_straight)
        ys2, vy2 = self.sample_cubic_trajectory(cy2, T_remain, self.num_straight)

        # 拼接（去掉第二段首点）
        xs = np.concatenate([xs1, xs2[1:]])
        ys = np.concatenate([ys1, ys2[1:]])
        vx = np.concatenate([vx1, vx2[1:]])
        vy = np.concatenate([vy1, vy2[1:]])

        phis = np.arctan2(vy, vx)
        vs   = np.hypot(vx, vy)
        return np.stack([xs, ys, phis, vs], axis=1)

    def _plan_straight(self, start, end):
        p0_xy, phi0 = np.array(start[:2]), start[2]
        pE_xy, phiE = np.array(end[:2]),   end[2]
        

        v0_xy = np.array([start[3]*math.cos(phi0), start[3]*math.sin(phi0)])
        vE_xy = np.array([end[3]*math.cos(phiE), end[3]*math.sin(phiE)])

        T_total = np.linalg.norm(pE_xy - p0_xy) / np.mean(start[3]+end[3]) 

        # x, y 分别拟合三次多项式
        cx = self.compute_cubic_coeffs(p0_xy[0], v0_xy[0], pE_xy[0], vE_xy[0], T_total)
        cy = self.compute_cubic_coeffs(p0_xy[1], v0_xy[1], pE_xy[1], vE_xy[1], T_total)
        xs, vx = self.sample_cubic_trajectory(cx, T_total, self.num_straight+self.num_change-1)
        ys, vy = self.sample_cubic_trajectory(cy, T_total, self.num_straight+self.num_change-1)

        phis = np.arctan2(vy, vx)
        vs   = np.hypot(vx, vy)
        return np.stack([xs, ys, phis, vs], axis=1)

    
    def plan_by_command(self, command, **kwargs):
        """
        command:
          - 'lane_change': 需要 start, mid, end 三个三元组
          - 'straight'   : 需要 start, end 两个三元组
        """
        if command == 'lane_change':
            return self._plan_lane_change(
                kwargs['start'], kwargs['mid'], kwargs['end']
            )
        elif command == 'straight':
            return self._plan_straight(
                kwargs['start'], kwargs['end']
            )
        else:
            raise ValueError(f"Unknown command: {command}")


