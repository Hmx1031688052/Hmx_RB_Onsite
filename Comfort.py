import numpy as np
import math

class ComfortValidator:
    def __init__(self, wheelbase=2.8, dt=0.1):
        """
        初始化舒适度控制器
        
        参数:
            wheelbase (float): 车辆轴距 (m)
            dt (float): 控制间隔 (s)
        """
        self.wheelbase = wheelbase
        self.dt = dt
        
        # 上一帧的状态（用于计算加加速度）
        self.a_x_prev = 0.0
        self.a_y_prev = 0.0
        
        # 舒适度阈值
        self.a_x_max = 3.0 - 0.001     # 最大纵向加速度 (m/s²)
        self.a_x_min = -3.0  + 0.001  # 最小纵向加速度 (m/s²)
        self.jerk_x_max = 6.0  - 0.001  # 最大纵向加加速度 (m/s³)
        self.jerk_x_min = -6.0 + 0.001  # 最小纵向加加速度 (m/s³)
        
        self.a_y_max = (0.5  - 0.0001 )*2   # 最大横向加速度 (m/s²)
        self.a_y_min =( -0.5  + 0.0001)*2    # 最小横向加速度 (m/s²)
        self.omega_max = (0.5  - 0.0001 )*2  # 最大横摆角速度 (rad/s)
        self.omega_min = (-0.5  + 0.0001 )*2 # 最小横摆角速度 (rad/s)
    
    def compute_lateral_acceleration(self, v, delta):
        """计算横向加速度和横摆角速度（基于单车模型）"""
        omega = v * math.tan(delta) / self.wheelbase  # 横摆角速度 (rad/s)
        a_y = v * omega  # 横向加速度 (m/s²)
        return a_y, omega
    
    def compute_jerk(self, a_x, a_y):
        """计算加加速度 (jerk)"""
        jerk_x = (a_x - self.a_x_prev) / self.dt  # 纵向加加速度 (m/s³)
        jerk_y = (a_y - self.a_y_prev) / self.dt  # 横向加加速度 (m/s³)
        return jerk_x, jerk_y
    
    def adjust_acceleration(self, a_x, jerk_x, a_y, omega):
        """
        修正加速度：
        1. 如果超过阈值，则设为最大允许值
        2. 如果无法满足舒适度要求，则返回0
        """
        # 先检查横向舒适度是否满足
        lateral_ok = (self.a_y_min <= a_y <= self.a_y_max) and \
                    (self.omega_min <= omega <= self.omega_max)
        
        if not lateral_ok:
            return 0.0  # 无法满足横向舒适度，加速度归零
        
        # 限制加加速度（jerk）
        if jerk_x > self.jerk_x_max:
            a_x = self.a_x_prev + self.jerk_x_max * self.dt
        elif jerk_x < self.jerk_x_min:
            a_x = self.a_x_prev + self.jerk_x_min * self.dt
        
        # 限制加速度绝对值
        if a_x > self.a_x_max:
            return self.a_x_max
        elif a_x < self.a_x_min:
            return self.a_x_min
        else:
            return a_x
    
    def control(self, a_x, delta, v):
        """
        主控制函数：仅修正加速度，不修改转向角
        
        参数:
            a_x (float): 当前帧的纵向加速度 (m/s²)
            delta (float): 当前帧的前轮转角 (rad)（不会被修改）
            v (float): 当前车速 (m/s)
            
        返回:
            tuple: 修正后的 (a_x, delta)
        """
        # 计算横向加速度和横摆角速度
        a_y, omega = self.compute_lateral_acceleration(v, delta)
        
        # 计算加加速度
        jerk_x, jerk_y = self.compute_jerk(a_x, a_y)
        
        # 修正加速度（可能返回0）
        a_x_adj = self.adjust_acceleration(a_x, jerk_x, a_y, omega)
        
        # 更新上一帧状态
        self.a_x_prev = a_x_adj
        self.a_y_prev = a_y
        
        # 返回修正后的加速度和原始转向角
        return a_x_adj, delta