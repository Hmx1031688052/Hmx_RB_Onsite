import os
import sys
import math
import copy
import numpy as np
import pandas as pd
import math
from numpy import *
import random
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from copy import deepcopy
from utils.observation import Observation
from epre_dsac.parameters import agent_par
from Comfort import ComfortValidator

# from OnSiteReplay.ReplayParser import ReplayParser
from guikong import kongzhi, Poly_planner_onsite, Obstacle
import warnings
from collections import deque
from epre_dsac.smarts_math import wrap_value, position_to_ego_frame
import time
import json
from global_plan_visualizer import save_global_plan_visualization
from PODAR import Veh,PODAR,SafetyResponder,TrajectoryPlanner,v_split,dynamic_model,render_moment,angle_normalize
try:
    from speed_limits import scene_speed_limit_for_map
except ImportError:
    from .speed_limits import scene_speed_limit_for_map
warnings.filterwarnings("ignore")

class Env():
    def __init__(self,train, x_goal=0, y_goal=0, state_dim=53, action_dim=2, use_epre_dsac = False):
        self.action_dim = action_dim
        self.use_epre_dsac = use_epre_dsac
        self.state_dim = int(agent_par.get("state_dim", state_dim)) if self.use_epre_dsac else state_dim
        self.sensor_range = 200
        self.x_goal = x_goal
        self.y_goal = y_goal
        self.v_max = 21
        self.scene_speed_limit = self.v_max
        self.scenario_dt = 0.02
        self.show_map = False
        self.big_turn = False

        self.show_guiji = False
        self.guiji_list = []
        self.ego_x = None
        self.ego_y = None
        self.ego_lane = 0
        self.last_ego_lane = -1
        self.in_goal_lane = False
        self.zhenlv = 1
        self.train = train
        self.frame_list = deque(maxlen = 30)
        self.dis_goal = 999
        if self.use_epre_dsac:
            self.use_dsac = True
            self.num_neighbors = 6   
            self.history_steps = 11
            self.num_lanes = 3
            self.num_waypoints = 51
            self.features = int(agent_par.get("history_feature", 8))
        self.episode_step = 0
        self.guikong_step = 0
        self.last_episode_step = 0
        self.use_predict_map = agent_par['predict_map']
        self.x_goal_min = 99999
        self.x_goal_max = 99999
        self.y_goal_min = 99999
        self.y_goal_max = 99999
        self.change_lane_sleep = 0
        self.epre_state = False
        self.front_list = deque(maxlen = 30)
        self.front_pedestrain_list = deque(maxlen = 10)
        self.last_front_pedestrain = {}
        self.follow = False
        self.follow_lane = -1
        self.change_lane_success = False
        self.bad_weather = False
        self.pre_jiaodu = 0
        self.npc_info_dict = {}
        self.is_intersection = False
        self.keep_str = False
        self.keep_time = 0
        self.map_file = None

        self.safe = 0
        self.light = None
        self.road_info = None  #场景地图信息
        self.lastrisk = None
        self.highrisk = 0.25
        self.lowrisk = 0.15
        self.danger_flag = False
        self.gg = False
        self.max_acc = True
        self.start = False
        self.tongji = False
        self.bijingdian = []
        self.tongji_path = None
        self.global_path = None
        self.global_path_tree = None
        self.global_path_source = None
        self.global_path_stamp = 0.0
        self.ego_path_s = 0.0
        self.ego_path_d = 0.0
        self.ego_path_lateral_error = 0.0
        self.global_path_future_points = int(agent_par.get("global_path_future_points", 5))
        self.frenet_pose_dim = int(agent_par.get("frenet_pose_dim", 3))
        self.frenet_state_dim = int(agent_par.get(
            "frenet_state_dim",
            9 + self.global_path_future_points * self.frenet_pose_dim,
        ))
        self.frenet_state = np.zeros(self.frenet_state_dim, dtype=float)
        self.frenet_state_fields = {}
        self.visualize_global_plan = os.environ.get("E2E_VIS_GLOBAL_PLAN", "0") == "1"
        self.global_plan_full_map = os.environ.get("E2E_VIS_GLOBAL_PLAN_FULL_MAP", "0") == "1"
        self.global_plan_vis_dir = os.environ.get("E2E_VIS_GLOBAL_PLAN_DIR")
        self.global_plan_vis_index = 0
        

    def _save_global_plan_visualization(self):
        if not self.visualize_global_plan:
            return

        try:
            output_path = save_global_plan_visualization(
                lane_info=self.lane_info,
                route_dict=self.road_info_dict_pingjie,
                map_file=self.map_file,
                start_xy=(self.x_start, self.y_start),
                goal_xy=(self.x_goal, self.y_goal),
                goal_lane=self.goal_lane,
                waypoints=self.bijingdian,
                output_dir=self.global_plan_vis_dir,
                episode_index=self.global_plan_vis_index,
                full_map=self.global_plan_full_map,
            )
            self.global_plan_vis_index += 1
            print("[global-plan-vis] saved {}".format(output_path))
        except Exception as exc:
            print("[global-plan-vis] failed: {}".format(exc))


    def clear_global_path(self):
        self.global_path = None
        self.global_path_tree = None
        self.global_path_source = None
        self.global_path_stamp = 0.0
        self.ego_path_s = 0.0
        self.ego_path_d = 0.0
        self.ego_path_lateral_error = 0.0
        if hasattr(self, "frenet_state_dim"):
            self.frenet_state = np.zeros(self.frenet_state_dim, dtype=float)
        self.frenet_state_fields = {}


    def _apply_scene_speed_limit(self, map_file=None):
        speed_limit = float(scene_speed_limit_for_map(map_file if map_file is not None else getattr(self, "map_file", "")))
        if not math.isfinite(speed_limit) or speed_limit <= 0.0:
            speed_limit = 10.0
        self.scene_speed_limit = speed_limit
        self.v_max = speed_limit
        if hasattr(self, "target_v"):
            try:
                self.target_v = min(float(self.target_v), speed_limit)
            except Exception:
                self.target_v = speed_limit
        else:
            self.target_v = speed_limit
        return speed_limit


    def set_global_path(self, path):
        if path is None:
            self.clear_global_path()
            return

        try:
            x = np.asarray(path.get("x", []), dtype=float).reshape(-1)
            y = np.asarray(path.get("y", []), dtype=float).reshape(-1)
        except Exception:
            return
        count = min(len(x), len(y))
        if count < 2:
            return
        x = x[:count]
        y = y[:count]
        valid = np.isfinite(x) & np.isfinite(y)
        indices = np.where(valid)[0]
        if len(indices) < 2:
            return

        x = x[indices]
        y = y[indices]
        yaw = self._path_array(path.get("yaw"), indices)
        kappa = self._path_array(path.get("kappa"), indices, default=0.0)
        station = self._path_array(path.get("s"), indices)

        if yaw is None:
            yaw = self._compute_path_yaw(x, y)
        if station is None or (len(station) > 1 and station[-1] <= station[0]):
            station = self._compute_path_station(x, y)
        if kappa is None:
            kappa = np.zeros(len(x), dtype=float)

        self.global_path = {
            "x": x,
            "y": y,
            "yaw": yaw,
            "kappa": kappa,
            "s": station,
            "frame_id": path.get("frame_id", ""),
        }
        self.global_path_tree = KDTree(np.column_stack([x, y]))
        self.global_path_source = path.get("source")
        self.global_path_stamp = float(path.get("stamp", time.time()))


    def _path_array(self, values, indices, default=None):
        if values is None:
            return None
        try:
            values = np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return None
        if len(values) <= int(np.max(indices)):
            return None
        result = values[indices]
        finite = np.isfinite(result)
        if np.all(finite):
            return result
        if default is None:
            return None
        result[~finite] = default
        return result


    def _compute_path_station(self, x, y):
        station = np.zeros(len(x), dtype=float)
        for index in range(1, len(x)):
            station[index] = station[index - 1] + math.hypot(x[index] - x[index - 1], y[index] - y[index - 1])
        return station


    def _compute_path_yaw(self, x, y):
        yaw = np.zeros(len(x), dtype=float)
        for index in range(len(x)):
            if index < len(x) - 1:
                dx = x[index + 1] - x[index]
                dy = y[index + 1] - y[index]
            else:
                dx = x[index] - x[index - 1]
                dy = y[index] - y[index - 1]
            yaw[index] = math.atan2(dy, dx)
        return yaw


    def _active_reference_path(self):
        if self.global_path is not None:
            return self.global_path, self.global_path_tree
        if not hasattr(self, "road_info_dict_pingjie") or not hasattr(self, "ego_lane"):
            return None, None
        lane_key = str(getattr(self, "ego_lane", 0))
        if lane_key not in self.road_info_dict_pingjie:
            return None, None
        lane = self.road_info_dict_pingjie[lane_key]
        center_vertices = np.asarray(lane.get("center_vertices", []), dtype=float)
        if len(center_vertices) < 2:
            return None, None
        yaw = lane.get("phi_road")
        if yaw is None:
            yaw = self._compute_path_yaw(center_vertices[:, 0], center_vertices[:, 1])
        station = lane.get("station")
        if station is None:
            station = self._compute_path_station(center_vertices[:, 0], center_vertices[:, 1])
        path = {
            "x": center_vertices[:, 0],
            "y": center_vertices[:, 1],
            "yaw": np.asarray(yaw, dtype=float),
            "kappa": np.asarray(lane.get("curvature", np.zeros(len(center_vertices))), dtype=float),
            "s": np.asarray(station, dtype=float),
            "frame_id": "lane",
        }
        return path, KDTree(center_vertices)


    def _project_to_reference_path(self, x, y):
        path, tree = self._active_reference_path()
        if path is None or tree is None:
            return {
                "s": 0.0,
                "d": 0.0,
                "lateral_error": 0.0,
                "index": 0,
                "path_length": max(1.0, float(getattr(self, "dis_goal", 1.0))),
            }

        point = np.array([float(x), float(y)], dtype=float)
        _, nearest = tree.query(point)
        nearest = int(nearest)
        xs = path["x"]
        ys = path["y"]
        stations = path["s"]
        best = None
        for seg_start in (nearest - 1, nearest):
            if seg_start < 0 or seg_start >= len(xs) - 1:
                continue
            p0 = np.array([xs[seg_start], ys[seg_start]], dtype=float)
            p1 = np.array([xs[seg_start + 1], ys[seg_start + 1]], dtype=float)
            vec = p1 - p0
            seg_len2 = float(np.dot(vec, vec))
            if seg_len2 < 1e-8:
                continue
            t = float(np.dot(point - p0, vec) / seg_len2)
            t = max(0.0, min(1.0, t))
            proj = p0 + t * vec
            dist2 = float(np.dot(point - proj, point - proj))
            if best is None or dist2 < best["dist2"]:
                seg_len = math.sqrt(seg_len2)
                cross = vec[0] * (point[1] - p0[1]) - vec[1] * (point[0] - p0[0])
                signed_d = cross / max(seg_len, 1e-6)
                s0 = float(stations[min(seg_start, len(stations) - 1)])
                best = {
                    "s": s0 + t * seg_len,
                    "d": signed_d,
                    "lateral_error": signed_d,
                    "index": seg_start,
                    "dist2": dist2,
                }
        if best is None:
            best = {
                "s": float(stations[min(nearest, len(stations) - 1)]),
                "d": 0.0,
                "lateral_error": 0.0,
                "index": nearest,
                "dist2": 0.0,
            }
        best["path_length"] = max(1.0, float(stations[-1]) if len(stations) else float(getattr(self, "dis_goal", 1.0)))
        return best


    def _normalized_path_features(self, x, y):
        projection = self._project_to_reference_path(x, y)
        path_length = max(1.0, projection["path_length"])
        return [
            projection["s"] / path_length,
            self.limit(projection["d"] / 10.0, -5.0, 5.0),
            self.limit(projection["lateral_error"] / 10.0, -5.0, 5.0),
        ]


    def _global_path_env_waypoints(self, count=None):
        if self.global_path is None or self.global_path_tree is None:
            return None
        count = self.num_waypoints if count is None else int(count)
        projection = self._project_to_reference_path(self.ego_x, self.ego_y)
        start_index = max(0, min(int(projection["index"]), len(self.global_path["x"]) - 1))
        waypoints = []
        for offset in range(count):
            index = min(start_index + offset, len(self.global_path["x"]) - 1)
            x = float(self.global_path["x"][index])
            y = float(self.global_path["y"][index])
            yaw = float(self.global_path["yaw"][index])
            vehicle_pos = self.transform(np.append([x, y], [0]))[:2]
            head = self.adjust_angle(yaw)
            waypoints.append([vehicle_pos[0], vehicle_pos[1], self.adjust_heading(head), 1])
        return waypoints


    def _global_path_future_state(self, count=None):
        count = self.global_path_future_points if count is None else int(count)
        features = []
        if self.global_path is None or self.global_path_tree is None:
            return [0.0] * (count * 3)
        projection = self._project_to_reference_path(self.ego_x, self.ego_y)
        start_index = max(0, min(int(projection["index"]) + 1, len(self.global_path["x"]) - 1))
        for offset in range(count):
            index = min(start_index + offset, len(self.global_path["x"]) - 1)
            x = float(self.global_path["x"][index])
            y = float(self.global_path["y"][index])
            yaw = float(self.global_path["yaw"][index])
            vehicle_pos = self.transform(np.append([x, y], [0]))[:2]
            head = self.adjust_angle(yaw)
            features.extend([
                vehicle_pos[0] / self.sensor_range,
                vehicle_pos[1] / self.sensor_range,
                self.adjust_heading(head) / math.pi,
            ])
        return features


    def _reference_future_state(self, count=None):
        count = self.global_path_future_points if count is None else int(count)
        features = []
        path, _ = self._active_reference_path()
        if path is None:
            return [0.0] * (count * self.frenet_pose_dim)

        projection = self._project_to_reference_path(self.ego_x, self.ego_y)
        start_index = max(0, min(int(projection["index"]) + 1, len(path["x"]) - 1))
        for offset in range(count):
            index = min(start_index + offset, len(path["x"]) - 1)
            x = float(path["x"][index])
            y = float(path["y"][index])
            yaw = float(path["yaw"][index])
            vehicle_pos = self.transform(np.append([x, y], [0]))[:2]
            head = self.adjust_angle(yaw)
            features.extend([
                vehicle_pos[0] / self.sensor_range,
                vehicle_pos[1] / self.sensor_range,
                self.adjust_heading(head) / math.pi,
            ])
        return features[:count * self.frenet_pose_dim]


    def _select_frenet_obstacle(self, obstacles, ego_s):
        if not obstacles:
            return None

        best_front = None
        best_any = None
        for key, value in obstacles.items():
            if not isinstance(value, dict) or value == {}:
                continue
            if "x" not in value or "y" not in value:
                continue
            projection = self._project_to_reference_path(value["x"], value["y"])
            delta_s = projection["s"] - ego_s
            candidate = {
                "id": key,
                "data": value,
                "projection": projection,
                "delta_s": delta_s,
            }
            if delta_s >= 0.0 and (best_front is None or delta_s < best_front["delta_s"]):
                best_front = candidate
            if best_any is None or abs(delta_s) < abs(best_any["delta_s"]):
                best_any = candidate

        return best_front if best_front is not None else best_any


    def _relative_obstacle_velocity(self, obstacle, ego_v):
        ego_yaw = float(getattr(self, "ego_yaw", 0.0))
        ego_v = float(ego_v or 0.0)
        ego_vx = ego_v * math.cos(ego_yaw)
        ego_vy = ego_v * math.sin(ego_yaw)

        obs_vx = obstacle.get("vx")
        obs_vy = obstacle.get("vy")
        if obs_vx is None or obs_vy is None:
            obs_v = float(obstacle.get("v", 0.0) or 0.0)
            obs_yaw = float(obstacle.get("yaw", ego_yaw) or 0.0)
            obs_vx = obs_v * math.cos(obs_yaw)
            obs_vy = obs_v * math.sin(obs_yaw)
        else:
            obs_vx = float(obs_vx or 0.0)
            obs_vy = float(obs_vy or 0.0)

        rel_vx_world = obs_vx - ego_vx
        rel_vy_world = obs_vy - ego_vy
        rel_vx_ego = rel_vx_world * math.cos(ego_yaw) + rel_vy_world * math.sin(ego_yaw)
        rel_vy_ego = -rel_vx_world * math.sin(ego_yaw) + rel_vy_world * math.cos(ego_yaw)
        return rel_vx_ego, rel_vy_ego


    def _build_frenet_state(self, ego_x, ego_y, ego_v, obstacles):
        ego_projection = self._project_to_reference_path(ego_x, ego_y)
        self.ego_path_s = ego_projection["s"]
        self.ego_path_d = ego_projection["d"]
        self.ego_path_lateral_error = ego_projection["lateral_error"]
        path_length = max(1.0, ego_projection["path_length"])

        obstacle = self._select_frenet_obstacle(obstacles, self.ego_path_s)
        obs_s = 0.0
        obs_d = 0.0
        obs_rel_vx = 0.0
        obs_rel_vy = 0.0
        if obstacle is not None:
            obs_projection = obstacle["projection"]
            obs_s = obs_projection["s"]
            obs_d = obs_projection["d"]
            obs_rel_vx, obs_rel_vy = self._relative_obstacle_velocity(obstacle["data"], ego_v)

        speed_limit = float(getattr(self, "scene_speed_limit", getattr(self, "v_max", 0.0)))
        if not math.isfinite(speed_limit):
            speed_limit = float(getattr(self, "v_max", 0.0))

        base_state = [
            self.ego_path_s / path_length,
            self.limit(self.ego_path_d / 10.0, -5.0, 5.0),
            ego_v / 50.0,
            speed_limit / 50.0,
            obs_s / path_length,
            self.limit(obs_d / 10.0, -5.0, 5.0),
            self.limit(obs_rel_vx / 50.0, -5.0, 5.0),
            self.limit(obs_rel_vy / 50.0, -5.0, 5.0),
            self.limit(abs(self.ego_path_lateral_error) / 10.0, 0.0, 5.0),
        ]
        pose_state = self._reference_future_state(self.global_path_future_points)
        frenet_state = base_state + pose_state
        if len(frenet_state) < self.frenet_state_dim:
            frenet_state += [0.0] * (self.frenet_state_dim - len(frenet_state))
        frenet_state = frenet_state[:self.frenet_state_dim]

        self.frenet_state = np.array(frenet_state, dtype=float)
        pose_fields = {}
        pose_start = len(base_state)
        for index in range(self.global_path_future_points):
            start = pose_start + index * self.frenet_pose_dim
            end = start + self.frenet_pose_dim
            pose_fields["pose{}".format(index + 1)] = frenet_state[start:end]

        self.frenet_state_fields = {
            "ego_s": frenet_state[0],
            "ego_d": frenet_state[1],
            "ego_v": frenet_state[2],
            "speed_limit": frenet_state[3],
            "obs_s": frenet_state[4],
            "obs_d": frenet_state[5],
            "obs_rel_vx": frenet_state[6],
            "obs_rel_vy": frenet_state[7],
            "lat_loss": frenet_state[8],
        }
        self.frenet_state_fields.update(pose_fields)
        return self.frenet_state.tolist()


    def _get_features(self, ego_x, ego_y, ego_v, ego_a, ego_yaw, ego_length, ego_width, obstacles, i_id):
    

        index_temp_ego = np.array(list(self.car_info_pre_ego.keys())) 
        index_temp_obs = np.array(list(self.car_info_pre_obs.keys())) 
        if self.show_guiji:
            self.guiji_list.append([ego_x, ego_y])
        self.change_lane_success = False

        ego_lane2 = None
        
        i = 0
        
        if self.ego_x is not None:
            self.ego_pre_x = self.ego_x
        else:
            self.ego_pre_x = ego_x
        if self.ego_y is not None:
            self.ego_pre_y = self.ego_y
        else:
            self.ego_pre_y = ego_y

        self.ego_x = ego_x
        self.ego_y = ego_y
        
        self.ego_v = ego_v
        self.ego_a = ego_a
        self.ego_yaw = ego_yaw
        self.ego_length = ego_length
        self.ego_width = ego_width
        v_max_temp = 0

        if agent_par['two_agent'] and self.map_type == 'AI_town':
            if (ego_x > 784500 and ego_x < 784560 and ego_y > 3352801 and ego_y < 3352862) or (ego_x > 784846 and ego_x < 784874 and ego_y > 3352905 and ego_y < 3352943):
                self.is_intersection = True
            else:
                self.is_intersection = False
        else:
            self.is_intersection = False
        
        ego_station = np.array([ego_x, ego_y])
        frame_dict = {}  # 用于存储帧数据的字典
        ego_list = {'ego': {'x':ego_x,'y':ego_y,'v':ego_v,'a':ego_a,'yaw':ego_yaw,'length':ego_length,'width':ego_width}}
        for key,value in ego_list.items():
            car_station = np.array([value['x'], value['y']])
            car_station_houlun = np.array([
                value['x'] - value['length'] / 1.7 / 2 * math.cos(ego_yaw),
                value['y'] - value['length'] / 1.7 / 2 * math.sin(ego_yaw)
            ])
            
            # 初始化一个字典来存储当前键的信息
            sub_dict = {
                'x': value['x'],
                'y': value['y'],
                'v': value['v'],
                'a': value['a'],
                'yaw': ego_yaw,
                'length': value['length'],
                'width': value['width'],
                'lane': 0,
                'lane_id':'-1.-1.-1.-1',
                'rel_pos_ind': 0,
                'rel_des': abs(np.linalg.norm(ego_station - car_station)),
                'ind_node_x':0,
                'ind_node_y': 0, 
                'dis_ind': 0,
                'zuoyou': 0,
                'rel_pos_ind_houlun': 0,
                'yaw_v':0, 
            }

            if key in index_temp_ego:
                error = sub_dict['yaw'] - self.car_info_pre_ego['ego']['yaw']
                if error > math.pi:
                    error = -2*math.pi + error
                if error < -math.pi:
                    error = 2*math.pi + error
                sub_dict['yaw_v'] = error / self.scenario_dt

            m = 9999999
            if self.ego_x > self.x_max or self.ego_x < self.x_min or self.ego_y > self.y_max or self.ego_y < self.y_min:
                if self.use_epre_dsac:
                    # print(555555555555555555555555555555555555555555555555555)
                    return None, None, None
                else:
                    return None


            lane_dict = {}
            for k1,v1 in self.road_info_dict_pingjie.items():
                refer_tree = KDTree(v1['center_vertices'])
                distance, ind = refer_tree.query(ego_station)
                distance_houlun, ind_houlun = refer_tree.query(car_station_houlun)

                lane = {}
                lane['id'] = int(k1)
                lane['rel_pos_ind'] = ind
                lane['rel_pos_ind_houlun'] = ind_houlun
                lane['ind_node_x'] = v1['center_vertices'][ind,0]
                lane['ind_node_y'] = v1['center_vertices'][ind,1]
                lane['dis_ind'] = distance
                if ind < len(v1['center_vertices']) - 2:
                    xiangliang_car = ego_station - v1['center_vertices'][ind,:]
                    xiangliang_line = v1['center_vertices'][ind+1,:] - v1['center_vertices'][ind,:]
                    
                    car_direction_vector = np.array([math.cos(self.ego_yaw), math.sin(self.ego_yaw)]) 
                    direction = 1 if np.dot(xiangliang_line, car_direction_vector) > 0 else -1

                    if direction == 1:
                        if xiangliang_car[0]*xiangliang_line[1] - xiangliang_car[1]*xiangliang_line[0] > 0:
                            lane['zuoyou'] = 1 #车在道路中心线右侧
                        elif xiangliang_car[0]*xiangliang_line[1] - xiangliang_car[1]*xiangliang_line[0] < 0:
                            lane['zuoyou'] = 2 #车在道路中心线左侧
                        else:
                            lane['zuoyou'] = 0 #车在道路中心线
                    else:
                        lane['zuoyou'] = -1
                else:
                    lane['zuoyou'] = -1 #车在道路外前方

                lane_dict[k1] = lane

            # self.ego_lane_list = []
            # self.left_lane_list = []
            # self.right_lane_list = []


            sorted_lanes = sorted(lane_dict.items(), key=lambda x: x[1]['dis_ind'])
            sorted_lane_ids = [lane_id for lane_id, _ in sorted_lanes]

            if self.last_ego_lane != -1:
                # print('lllll',lane_dict[str(self.last_ego_lane)]['dis_ind'], self.last_ego_lane)
                if lane_dict[str(self.last_ego_lane)]['dis_ind'] < 1.6:
                    ego_lane = self.last_ego_lane
                    filtered_lanes = {k: v for k, v in lane_dict.items() if v['zuoyou'] != -1}
                    if filtered_lanes != {}:
                        filtered_sorted_lanes = sorted(filtered_lanes.items(), key=lambda x: x[1]['dis_ind'])
                        filtered_sorted_lane_ids = [lane_id for lane_id, _ in filtered_sorted_lanes]
                        ego_lane2 = int(filtered_sorted_lane_ids[0])
                    else:
                        # 如果没有同向车道，则选择所有车道中最近的
                        ego_lane2 = int(sorted_lane_ids[0])

                else:
                    filtered_lanes = {k: v for k, v in lane_dict.items() if v['zuoyou'] != -1}
                    if filtered_lanes != {}:
                        filtered_sorted_lanes = sorted(filtered_lanes.items(), key=lambda x: x[1]['dis_ind'])
                        filtered_sorted_lane_ids = [lane_id for lane_id, _ in filtered_sorted_lanes]
                        ego_lane2 = ego_lane = int(filtered_sorted_lane_ids[0])
                    else:
                        # 如果没有同向车道，则选择所有车道中最近的
                        ego_lane2 = ego_lane = int(sorted_lane_ids[0])
            else:
                filtered_lanes = {k: v for k, v in lane_dict.items() if v['zuoyou'] != -1}
                if filtered_lanes != {}:
                    filtered_sorted_lanes = sorted(filtered_lanes.items(), key=lambda x: x[1]['dis_ind'])
                    filtered_sorted_lane_ids = [lane_id for lane_id, _ in filtered_sorted_lanes]
                    ego_lane2 = ego_lane = int(filtered_sorted_lane_ids[0])
                else:
                    # 如果没有同向车道，则选择所有车道中最近的
                    ego_lane2 = ego_lane = int(sorted_lane_ids[0])

            # ego_lane2 = int(sorted_lane_ids[0])
            


            # self.ego_lane_list.append(ego_lane)
            # print(lane_dict[str(ego_lane)]['id'], lane_dict[str(sorted_lane_ids[1])]['id'], abs(lane_dict[str(ego_lane)]['dis_ind'] - lane_dict[str(sorted_lane_ids[1])]['dis_ind']), math.sqrt((lane_dict[str(ego_lane)]['ind_node_x'] - lane_dict[str(sorted_lane_ids[1])]['ind_node_x'])**2 + (lane_dict[str(ego_lane)]['ind_node_y'] - lane_dict[str(sorted_lane_ids[1])]['ind_node_y'])**2))
            # if self.map_type == 'AI_town' and len(sorted_lane_ids) >= 2 and abs(lane_dict[str(ego_lane)]['dis_ind'] - lane_dict[str(sorted_lane_ids[1])]['dis_ind']) < 1 \
            #     and math.sqrt((lane_dict[str(ego_lane)]['ind_node_x'] - lane_dict[str(sorted_lane_ids[1])]['ind_node_x'])**2 + (lane_dict[str(ego_lane)]['ind_node_y'] - lane_dict[str(sorted_lane_ids[1])]['ind_node_y'])**2) < 2.7:
            #     self.ego_lane_list.append(int(sorted_lane_ids[1]))
            #     ego_lane2 = int(sorted_lane_ids[1])

            # 4. 右侧第一条车道（zuoyou=1，且不是自车所在车道）

            left_lane_candidates = [
                lane_id for lane_id in sorted_lane_ids
                if lane_dict[lane_id]['zuoyou'] == 1 and lane_dict[lane_id]['id'] != ego_lane and lane_dict[lane_id]['id'] != ego_lane2
            ]

            left_lane = int(left_lane_candidates[0]) if left_lane_candidates else None
            # if left_lane_candidates:
            #     self.left_lane_list.append(left_lane)

            # if len(left_lane_candidates) >= 2:
            #     print('left',lane_dict[str(left_lane)]['id'],  lane_dict[str(left_lane_candidates[1])]['id'], abs(lane_dict[str(left_lane)]['dis_ind'] - lane_dict[str(left_lane_candidates[1])]['dis_ind']), math.sqrt((lane_dict[str(left_lane)]['ind_node_x'] - lane_dict[str(left_lane_candidates[1])]['ind_node_x'])**2 + (lane_dict[str(left_lane)]['ind_node_y'] - lane_dict[str(left_lane_candidates[1])]['ind_node_y'])**2))
            # if self.map_type == 'AI_town' and len(left_lane_candidates) >= 2 and abs(lane_dict[str(left_lane)]['dis_ind'] - lane_dict[str(left_lane_candidates[1])]['dis_ind']) < 1 \
            #     and math.sqrt((lane_dict[str(left_lane)]['ind_node_x'] - lane_dict[str(left_lane_candidates[1])]['ind_node_x'])**2 + (lane_dict[str(left_lane)]['ind_node_y'] - lane_dict[str(left_lane_candidates[1])]['ind_node_y'])**2) < 2.7:
            #     self.left_lane_list.append(int(left_lane_candidates[1]))

            right_lane_candidates = [
                lane_id for lane_id in sorted_lane_ids
                if lane_dict[lane_id]['zuoyou'] == 2 and lane_dict[lane_id]['id'] != ego_lane and lane_dict[lane_id]['id'] != ego_lane2
            ]
            right_lane =int(right_lane_candidates[0]) if right_lane_candidates else None
            # if right_lane_candidates:
            #     self.right_lane_list.append(right_lane)

            # if self.map_type == 'AI_town' and len(right_lane_candidates) >= 2 and abs(lane_dict[str(right_lane)]['dis_ind'] - lane_dict[str(right_lane_candidates[1])]['dis_ind']) < 1 \
            #     and math.sqrt((lane_dict[str(right_lane)]['ind_node_x'] - lane_dict[str(right_lane_candidates[1])]['ind_node_x'])**2 + (lane_dict[str(right_lane)]['ind_node_y'] - lane_dict[str(right_lane_candidates[1])]['ind_node_y'])**2) < 2.7:
            #     self.right_lane_list.append(int(right_lane_candidates[1]))
            
            sub_dict['lane'] = int(ego_lane)
            sub_dict['rel_pos_ind'] = lane_dict[str(ego_lane)]['rel_pos_ind']
            sub_dict['rel_pos_ind_houlun'] = lane_dict[str(ego_lane)]['rel_pos_ind_houlun']
            sub_dict['ind_node_x'] = lane_dict[str(ego_lane)]['ind_node_x']
            sub_dict['ind_node_y'] = lane_dict[str(ego_lane)]['ind_node_y']
            sub_dict['dis_ind'] = lane_dict[str(ego_lane)]['dis_ind']
            sub_dict['zuoyou'] = lane_dict[str(ego_lane)]['zuoyou']

            self.right_lane_id = right_lane
            self.left_lane_id = left_lane
            if self.left_lane_id is not None:
                self.rel_pos_ind_left = lane_dict[str(left_lane)]['rel_pos_ind']
            else:
                self.rel_pos_ind_left = None
            if self.right_lane_id is not None:
                self.rel_pos_ind_right = lane_dict[str(right_lane)]['rel_pos_ind']
            else:
                self.rel_pos_ind_right = None

            self.lane_dict = lane_dict

            if self.use_epre_dsac and (self.last_episode_step < self.episode_step or self.guikong_step==0):
                self.epre_state = True
                car_id = key
                car_dict = {
                    'step' : self.episode_step,
                    'x' : sub_dict['x'],
                    'y' : sub_dict['y'],
                    'v' : sub_dict['v'],
                    'angle' : sub_dict['yaw'],
                    'a' : sub_dict['a'],
                    'lane':sub_dict['lane'],
                    'zuoyou':sub_dict['zuoyou'],
                    'rel_pos_ind':sub_dict['rel_pos_ind'],
                }
                if car_id not in self.vehicle_dict:
                    self.vehicle_dict[car_id] = {}
                self.vehicle_dict[car_id][self.episode_step] = (car_dict)
            else:
                self.epre_state = False
            frame_dict[key] = sub_dict
            if key == 'ego':
                self.ego_info_dict = frame_dict

        ego_lane = int(self.ego_info_dict['ego']['lane'])
        self.ego_lane = ego_lane
        self.last_ego_lane = self.ego_lane
        ego_lane_width = self.road_info_dict_pingjie[str(ego_lane)]['width'][self.ego_info_dict['ego']['rel_pos_ind']]
        refer_tree = KDTree(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'])
        if self.left_lane_id is not None:
            left_refer_tree = KDTree(self.road_info_dict_pingjie[str(self.left_lane_id)]['center_vertices'])
        if self.right_lane_id is not None:
            right_refer_tree = KDTree(self.road_info_dict_pingjie[str(self.right_lane_id)]['center_vertices'])
        frame_dict_obj = {}  # 用于存储帧数据的字典
        for key, value in obstacles.items():
            if value != {}:
                key = value['id']
                car_station = np.array([value['x'],value['y']])
                obs_yaw = value.get('yaw', value.get('theta', ego_yaw))

                sub_dict = {
                    'x': value['x'],
                    'y': value['y'],
                    'v': value['v'],
                    'a': value['a'],
                    'yaw': obs_yaw,
                    'vx': value.get('vx'),
                    'vy': value.get('vy'),
                    'length': value['length'],
                    'width': value['width'],
                    'lane': -1,
                    'lane_id':'-1.-1.-1.-1',
                    'rel_pos_ind': 0,
                    'rel_des': abs(np.linalg.norm(ego_station - car_station)),
                    'ind_node_x':0,
                    'ind_node_y': 0, 
                    'dis_ind': 0,
                    'zuoyou': 0,
                    'rel_pos_ind_houlun': 0,
                    'yaw_v':0, 
                    'type':value['type'],
                }
                if self.show_map:
                    plt.plot(sub_dict['x'], sub_dict['y'], 'ro', zorder = 10, markersize = 0.8)
                    plt.pause(0.1)
                    

                if key in index_temp_obs:
                    error = sub_dict['yaw'] - self.car_info_pre_obs[key]['yaw']
                    if error > math.pi:
                        error = -2*math.pi + error
                    if error < -math.pi:
                        error = 2*math.pi + error
                    sub_dict['yaw_v'] = error / self.scenario_dt

                if sub_dict['v'] > v_max_temp:
                    v_max_temp = sub_dict['v']

                distance, ind = refer_tree.query(car_station)
                have_lane = False
                if distance <= self.road_info_dict_pingjie[str(ego_lane)]['width'][ind]/2 + 1e-4:
                    sub_dict['lane'] = ego_lane
                    have_lane = True
                else:
                    if self.left_lane_id is not None and not have_lane:
                        distance_left, ind_left = left_refer_tree.query(car_station)
                        if distance_left <= self.road_info_dict_pingjie[str(self.left_lane_id)]['width'][ind_left]/2 + 1e-4:
                            sub_dict['lane'] = self.left_lane_id
                            have_lane = True
                    if self.right_lane_id is not None and not have_lane:
                        distance_right, ind_right = right_refer_tree.query(car_station)
                        if distance_right <= self.road_info_dict_pingjie[str(self.right_lane_id)]['width'][ind_right]/2 + 1e-4:
                            sub_dict['lane'] = self.right_lane_id
                            have_lane = True

                

                sub_dict['rel_pos_ind'] = ind
                sub_dict['ind_node_x'] = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind,0]
                sub_dict['ind_node_y'] = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind,1]
                sub_dict['dis_ind'] = distance
                if ind < len(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices']) - 2:
                    xiangliang_car = car_station - self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind,:]
                    xiangliang_line = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind+1,:] - self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind,:]
                    if xiangliang_car[0]*xiangliang_line[1] - xiangliang_car[1]*xiangliang_line[0] > 0:
                        sub_dict['zuoyou'] = 1 #车在道路中心线右侧
                    elif xiangliang_car[0]*xiangliang_line[1] - xiangliang_car[1]*xiangliang_line[0] < 0:
                        sub_dict['zuoyou'] = 2 #车在道路中心线左侧
                    else:
                        sub_dict['zuoyou'] = 0 #车在道路中心线
                else:
                    sub_dict['zuoyou'] = 0 #车在道路外前方

                if self.use_epre_dsac and (self.last_episode_step < self.episode_step or self.guikong_step==0):
                    car_id = key
                    car_dict = {
                        'step' : self.episode_step,
                        'x' : sub_dict['x'],
                        'y' : sub_dict['y'],
                        'v' : sub_dict['v'],
                        'angle' : sub_dict['yaw'],
                        'a' : sub_dict['a'],
                        'lane':sub_dict['lane'],
                        'zuoyou':sub_dict['zuoyou'],
                        'rel_pos_ind':sub_dict['rel_pos_ind'],
                        'rel_des':sub_dict['rel_des']
                        
                    }
                    if car_id not in self.vehicle_dict:
                        self.vehicle_dict[car_id] = {}
                    self.vehicle_dict[car_id][self.episode_step] = (car_dict)
                    
                    
                frame_dict_obj[key] = sub_dict
                
        self.car_info_pre_ego = self.ego_info_dict
        self.car_info_pre_obs = frame_dict_obj
        
        if self.episode_step > self.last_episode_step:
            self.last_episode_step += 1
        # print(self.episode_step, self.last_episode_step)

                
        #速度限制，跟随道路曲率
        v_max_des = 0
        ind = self.ego_info_dict['ego']['rel_pos_ind']
        total = 0
        for i in range(0,5):
            ind2 = ind + i
            ind2 = self.limit(ind2,0,len(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'])-2)
            total += abs(self.road_info_dict_pingjie[str(ego_lane)]['curvature'][ind2])
        self.k_lane3 = total / 5
        total = 0
        for i in range(0,10):
            ind2 = ind + i
            ind2 = self.limit(ind2,0,len(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'])-2)
            total += abs(self.road_info_dict_pingjie[str(ego_lane)]['curvature'][ind2])
        self.k_lane_turn = total / 10
        total = 0
        for i in range(-5,20):
            ind2 = ind + i
            ind2 = self.limit(ind2,0,len(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'])-2)
            total += abs(self.road_info_dict_pingjie[str(ego_lane)]['curvature'][ind2])
        k_lane = total / 20
        for i in range(-1,1):
            ind2 = ind + i
            ind2 = self.limit(ind2,0,len(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'])-2)
            total += abs(self.road_info_dict_pingjie[str(ego_lane)]['curvature'][ind2])
        k_lane2 = total / 2
        self.k_lane2 = k_lane2
        if self.map_type == 'HangZhouWan':
            for i in range(-6,70):
                ind2 = ind + i
                ind2 = self.limit(ind2,0,len(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'])-2)
                total += abs(self.road_info_dict_pingjie[str(ego_lane)]['curvature'][ind2])
            k_lane = total / 40
            
        if self.map_type == 'AI_town':
            # if k_lane <= 0.001:
            #     v_max_des = self.v_max
            # elif k_lane > 0.017:
            #     v_max_des = 5
            # else:
            #     v_max_des = -((self.v_max-5)/0.017) * k_lane + self.v_max
            # if k_lane2 > 0.53:
            #     v_max_des = 4
            if k_lane <= 0.001:
                v_max_des = self.v_max
            elif k_lane > 0.025:
                v_max_des = 5
            else:
                v_max_des = -((self.v_max-5)/0.025) * k_lane + self.v_max
            if self.k_lane3 > 0.1:
                v_max_des = 4
            elif self.k_lane3 < 0.03:
                v_max_des *= 1.2
            elif self.k_lane3 < 0.05:
                v_max_des *= 1.13
            elif self.k_lane3 < 0.07:
                v_max_des *= 1.06

        elif self.map_type == 'HangZhouWan':
            if k_lane > 0.0006:
                v_max2 = 33
            else:
                v_max2 = self.v_max
            if k_lane <= 0.001:
                v_max_des = v_max2
            elif k_lane > 0.013:
                v_max_des = 14
            else:
                v_max_des = -((v_max2-14)/0.013) * k_lane + v_max2
                if v_max_des > v_max2 * 0.8:
                    v_max_des = v_max2*0.8
        else:
            if k_lane <= 0.001:
                v_max_des = self.v_max
            elif k_lane > 0.017:
                v_max_des = 16
            else:
                v_max_des = -((self.v_max-16)/0.017) * k_lane + self.v_max
        #print(k_lane , v_max_des)
        self.k_lane =k_lane
        self.v_max1 = self.limit(v_max_des,2,self.v_max)

        if agent_par['shushidu']:
            min_R = 9999999999999
            stop_dis = self.ego_v**2/4
            point = math.ceil(stop_dis / 2)
            for i in range(0,point + 10):
                ind2 = ind + i
                ind2 = self.limit(ind2,0,len(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'])-2)
                if self.road_info_dict_pingjie[str(ego_lane)]['curvature'][ind2] > 0:
                    R = 1/self.road_info_dict_pingjie[str(ego_lane)]['curvature'][ind2]
                    if R < min_R:
                        min_R = R
            
            max_v_w = 0.5*min_R
            max_v_ha = 0.5*np.sqrt(min_R)
            self.v_max1 = max(min(max_v_w*0.995, self.v_max1, max_v_ha*0.995), 0)
            

        # if self.ego_v > 10 or self.gg:
        #     if self.ego_v > 10:
        #         print(1111, self.ego_x, self.ego_y)
        #     self.v_max1 = 0
        #     self.gg = True
        
        # if self.ego_v < 3:
        #     print(22222, self.ego_x, self.ego_y)
        
        if agent_par['podar'] and self.is_intersection:
            time1 = time.time()
            podar=PODAR()

            #自车轨迹计算
            traj = self.get_ego_traj(lane=self.target_lane_pre,v0=ego_v,a=ego_a,ind=self.ego_info_dict['ego']['rel_pos_ind'],dt=0.1,size=41)#五次多项式规划
            traj_2 = self.get_ego_traj_road(lane=self.target_lane_pre,v0=ego_v,a=ego_a,ind=self.ego_info_dict['ego']['rel_pos_ind'],dt=0.1,size=41)#直接采用路点
            
            #风险评估

            if self.lastrisk is not None:
                if not self.danger_flag:
                    if self.lastrisk.risk > self.highrisk:
                        self.danger_flag = True
                else:
                    if self.lastrisk.risk < self.lowrisk:
                        self.danger_flag = False
                self.lastrisk.danger_flag = self.danger_flag
                    
            

        
            # print('frame info: ')
            # print(frame)

            #replay
            # podar.parse_frame_to_podar(frame,traj,lastres=self.lastrisk)

            #fragment
            podar.parse_frame_to_podar_fragment(self.npc_info_dict,traj,lastres=self.lastrisk, ego_frame = self.ego_info_dict)

            
            ego,max_risk_obj = podar.get_max_risk_obj_info()
            self.lastrisk = podar.ego

            self.safe = podar.lon_res






        if len(frame_dict_obj)>0:
            self.frame_list.append(1)
            self.last_frame_dict_obj = frame_dict_obj
        else:
            self.frame_list.append(0)

        def predict_vehicle_state(df, delta_t=0.2):
            """
            基于匀速直线运动模型预测车辆未来状态
            参数：
                df : 输入的DataFrame (必须包含x,y,v,a,yaw列)
                delta_t : 预测时间间隔(秒)
            返回：
                预测后的完整DataFrame (保持原始索引和列顺序)
            """
            # 创建副本避免修改原始数据
            df_pred = df.copy()
            
            # 计算位移变化量
            theta = df_pred['yaw']
            dx = df_pred['v'] * np.cos(theta) * delta_t
            dy = df_pred['v'] * np.sin(theta) * delta_t
            
            # 更新坐标
            df_pred['x'] += dx
            df_pred['y'] += dy
            
            
            return df_pred

        if len(frame_dict_obj)==0 and sum(self.frame_list)>0:
            # frame1 = predict_vehicle_state(self.last_frame)
            # print(9999)
            # print(frame1)
            frame_dict_obj = self.last_frame_dict_obj
        self.npc_info_dict = frame_dict_obj
        # print(frame_dict_obj)
        # print(self.frame_list)
        pedestrain_dis = 2.8
        # if self.k_lane_turn > 0.01 or self.ego_v > 14:
        #     pedestrain_dis = 5
        # elif self.ego_v > 10:
        #     pedestrain_dis = 4.2
        # if self.is_intersection:
        #     pedestrain_dis = 2.5
        same = {}
        self.front = {}
        self.front_pedestrain = {}
        self.behind = {}
        if len(frame_dict_obj)>0:
            for key, value in frame_dict_obj.items():
                if value['type'] == 1 and self.v_max1 > 7.2:
                    self.v_max1 *= 0.8
                if value['lane'] == int(self.ego_info_dict['ego']['lane']) or (value['type'] == 1 and value['dis_ind'] < pedestrain_dis):
                    same[key] = value

        if len(same) > 0:
            for key, value in same.items():
                if value['rel_pos_ind'] >= int(self.ego_info_dict['ego']['rel_pos_ind']):
                    self.front[key] = value
                    if value['type'] == 1:
                        self.front_pedestrain[key] = value
                else:
                    self.behind[key]= value
                    
        if self.left_lane_id is not None:
            left = {}
            self.left_front = {}
            self.left_behind = {}
            if len(frame_dict_obj)>0:
                for key, value in frame_dict_obj.items():
                    if value['lane'] == int(self.left_lane_id):
                        left[key] = value

            if len(left) > 0:
                for key, value in left.items():
                    if value['rel_pos_ind'] >= int(self.ego_info_dict['ego']['rel_pos_ind']):
                        self.left_front[key] = value
                    else:
                        self.left_behind[key]= value
        else:
            self.left_front = {}
            self.left_behind = {}

        if self.right_lane_id is not None:
            right = {}
            self.right_front = {}
            self.right_behind = {}
            if len(frame_dict_obj)>0:
                for key, value in frame_dict_obj.items():
                    if value['lane'] == int(self.right_lane_id):
                        right[key] = value

            if len(right) > 0:
                for key, value in right.items():
                    if value['rel_pos_ind'] >= int(self.ego_info_dict['ego']['rel_pos_ind']):
                        self.right_front[key] = value
                    else:
                        self.right_behind[key]= value
        else:
            self.right_front = {}
            self.right_behind = {}


        if len(self.front) > 0:
            self.front_list.append(1)
        else:
            self.front_list.append(0)

        if len(self.front_pedestrain) > 0:
            self.front_pedestrain_list.append(1)
            self.last_front_pedestrain = self.front_pedestrain
        else:
            self.front_pedestrain_list.append(0)

        if sum(self.front_pedestrain_list) > 0 and len(self.front_pedestrain) == 0:
            self.front_pedestrain = self.last_front_pedestrain
            for key, value in self.front_pedestrain.items():
                self.front[key] = value


        if sum(self.front_list) > 40 and self.ego_v < 5:
            self.follow = True
            self.follow_lane = ego_lane
            # print(sum(self.front_list), self.ego_v, self.follow)
            
        if sum(self.front_list) == 0 and self.ego_v > 10 and self.follow and self.follow_lane != ego_lane and self.follow_lane != -1:
            self.follow = False
            self.follow_lane = -1
            self.change_lane_success = True
            print('success change lane')
            

        if np.sqrt((ego_x - self.x_start)**2 + (ego_y - self.y_start)**2) <= 7  and not self.start:
            self.start = True
            self.target_lane_pre = self.ego_lane

        
        library = frame_dict_obj
        x_1 , x_2 , x_3 , x_4 , x_5 , x_6 = (0 for i in range(6))
        y_1 , y_2 , y_3 , y_4 , y_5 , y_6 = (0 for i in range(6))
        v_1 , v_2 , v_3 , v_4 , v_5 , v_6 = (0 for i in range(6))
        angle_1, angle_2, angle_3, angle_4, angle_5, angle_6 = (0 for i in range(6))
        length_1 , length_2 , length_3 , length_4 , length_5, length_6 = (0 for i in range(6))
        width_1 , width_2 , width_3 , width_4 , width_5, width_6 = (0 for i in range(6))
        acc_1 , acc_2 , acc_3 , acc_4 , acc_5, acc_6 = (0 for i in range(6))
        self.x = ego_x
        self.y = ego_y
        self.head = self.adjust_angle(ego_yaw)
        goal_xiangdui = self.transform(np.append([self.goal_center_x,self.goal_center_y], [0]))[:2]
        self.goal_xiangdui_x = goal_xiangdui[0]
        self.goal_xiangdui_y = goal_xiangdui[1]    

        if self.use_epre_dsac and self.epre_state:
        
            x_1 , x_2 , x_3 , x_4 , x_5 , x_6 = (0 for i in range(6))
            y_1 , y_2 , y_3 , y_4 , y_5 , y_6 = (0 for i in range(6))
            vx_1 , vx_2 , vx_3 , vx_4 , vx_5 , vx_6 = (0 for i in range(6))
            vy_1 , vy_2 , vy_3 , vy_4 , vy_5 , vy_6 = (0 for i in range(6))
            v_1 , v_2 , v_3 , v_4 , v_5 , v_6 = (0 for i in range(6))

            angle_1, angle_2, angle_3, angle_4, angle_5, angle_6 = (0 for i in range(6))
            length_1, length_2, length_3, length_4, length_5, length_6 = (0 for i in range(6))
            width_1, width_2, width_3, width_4, width_5, width_6 = (0 for i in range(6))
            a_1, a_2, a_3, a_4, a_5, a_6 =  (0 for i in range(6))
            index = []
            if len(library)>0:
                
                rel_des_values = {key: value['rel_des'] for key, value in library.items()}  
                  
                if len(rel_des_values) <= 6:
                    # 对字典按值排序，返回排序后的索引
                    sorted_keys = sorted(rel_des_values, key=rel_des_values.get)
                    index = sorted_keys
                else:
                    # 找出 rel_des 中最小的 6 个值，并按值排序
                    smallest_items = sorted(rel_des_values.items(), key=lambda item: item[1])[:6]
                    index = sorted(smallest_items, key=lambda item: item[1])
                    index = [item[0] for item in index]
                    

            # print(obstacles)
            # print(frame)
            # print(index)


            def append_vehicle_dict(vehicle_id):
                # print(666,vehicle_id)
                vehicle_data = np.zeros((self.history_steps,self.features))
                vehicle_dict = self.vehicle_dict[vehicle_id]
                # print(vehicle_dict)
                vehicle_dict_step_next = None
                if vehicle_id == 'ego':
                    length = ego_length
                    width = ego_width
                else:
                    length = frame_dict_obj[vehicle_id]['length']
                    width = frame_dict_obj[vehicle_id]['width']

                def fit_vehicle_features(values):
                    if len(values) < self.features:
                        values = values + [0.0] * (self.features - len(values))
                    return np.array(values[:self.features])

                for i in range(self.history_steps):
                    step = self.episode_step - i
                    if step <= 0:
                        break
                    try:
                        vehicle_dict_step = vehicle_dict[step]
                        vehicle_dict_step_next = vehicle_dict_step
                    except:
                        if vehicle_dict_step_next is not None and step > 1:
                            try:
                                vehicle_dict_step_last = vehicle_dict[step-1]
                                vehicle_dict_step = {
                                'step': (vehicle_dict_step_last['step'] + vehicle_dict_step_next['step']) / 2,               
                                'x': (vehicle_dict_step_last['x'] + vehicle_dict_step_next['x']) / 2,                       
                                'y': (vehicle_dict_step_last['y'] + vehicle_dict_step_next['y']) / 2,                  
                                'v': (vehicle_dict_step_last['v'] + vehicle_dict_step_next['v']) / 2,                  
                                'angle': (vehicle_dict_step_last['angle'] + vehicle_dict_step_next['angle']) / 2,            
                                'a': (vehicle_dict_step_last['a'] + vehicle_dict_step_next['a']) / 2,                        
                                'lane': (vehicle_dict_step_last['lane'] + vehicle_dict_step_next['lane']) / 2,            
                                'zuoyou': vehicle_dict_step_last['zuoyou'],        
                                'rel_pos_ind': (vehicle_dict_step_last['rel_pos_ind'] + vehicle_dict_step_next['rel_pos_ind']) / 2 
                                    }
                                vehicle_dict_step_next = None
                            except:
                                try:
                                    vehicle_dict_step_last_last = vehicle_dict[step-2]
                                    vehicle_dict_step = {
                                    'step': vehicle_dict_step_last_last['step'] + (vehicle_dict_step_next['step'] - vehicle_dict_step_last_last['step'])*2/3,               
                                    'x': vehicle_dict_step_last_last['x'] + (vehicle_dict_step_next['x'] - vehicle_dict_step_last_last['x'])*2/3,                       
                                    'y': vehicle_dict_step_last_last['y'] + (vehicle_dict_step_next['y'] - vehicle_dict_step_last_last['y'])*2/3,                  
                                    'v': vehicle_dict_step_last_last['v'] + (vehicle_dict_step_next['v'] - vehicle_dict_step_last_last['v'])*2/3,                  
                                    'angle': vehicle_dict_step_last_last['angle'] + (vehicle_dict_step_next['angle'] - vehicle_dict_step_last_last['angle'])*2/3,            
                                    'a': vehicle_dict_step_last_last['a'] + (vehicle_dict_step_next['a'] - vehicle_dict_step_last_last['a'])*2/3,                        
                                    'lane': vehicle_dict_step_last_last['lane'] + (vehicle_dict_step_next['lane'] - vehicle_dict_step_last_last['lane'])*2/3,            
                                    'zuoyou': vehicle_dict_step_last_last['zuoyou'],        
                                    'rel_pos_ind': vehicle_dict_step_last_last['rel_pos_ind'] + (vehicle_dict_step_next['rel_pos_ind'] - vehicle_dict_step_last_last['rel_pos_ind'])*2/3}
                                    vehicle_dict_step_next = vehicle_dict_step
                                except:
                                    break
                        else:
                            if step == self.episode_step:
                                try:
                                    vehicle_dict_step = vehicle_dict[step-1]
                                except:
                                    break
                            else:
                                break
                                

                    vehicle_pos = self.transform(np.append([vehicle_dict_step['x'],vehicle_dict_step['y']], [0]))[:2]
                    head = self.adjust_angle(vehicle_dict_step['angle'])
                    a = vehicle_dict_step['a']
                    
                        
                    vehicle_head = self.adjust_heading(head)
                    vehicle_x = vehicle_pos[0]
                    vehicle_y = vehicle_pos[1]
                    if a > 15:
                        a = 15
                    elif a < -15:
                        a = -15
                    
                    path_features = self._normalized_path_features(
                        vehicle_dict_step['x'],
                        vehicle_dict_step['y'],
                    )

                    if agent_par['use_other_direction']:#旁车方位信息
                        position = 7
                        rel_pos_ind = vehicle_dict_step['rel_pos_ind']
                        car_station = np.array([vehicle_dict_step['x'],vehicle_dict_step['y']])
                        if rel_pos_ind > len(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices']) - 2:
                            rel_pos_ind =  len(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices']) - 2

                        xiangliang_car = car_station - self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][rel_pos_ind,:]
                        xiangliang_line = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][rel_pos_ind+1,:] - self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][rel_pos_ind,:]
                        if xiangliang_car[0]*xiangliang_line[1] - xiangliang_car[1]*xiangliang_line[0] > 0:
                            zuoyou = 1 #车在道路中心线右侧
                        elif xiangliang_car[0]*xiangliang_line[1] - xiangliang_car[1]*xiangliang_line[0] < 0:
                            zuoyou = 2 #车在道路中心线左侧
                        else:
                            zuoyou = 0 #车在道路中心线

                        lane = vehicle_dict_step['lane']
                        # zuoyou = vehicle_dict_step['zuoyou']
                        if rel_pos_ind >  self.ego_info_dict['ego']['rel_pos_ind']:#前
                            if lane == int(self.ego_info_dict['ego']['lane']) or zuoyou == 0: #正前
                                position = 2
                            else:
                                if zuoyou == 1: #右前
                                    position = 3
                                elif zuoyou == 2: #左前
                                    position = 1

                        else:#后
                            if lane == int(self.ego_info_dict['ego']['lane']) or zuoyou == 0: #正后
                                position = 5

                            else:
                                if zuoyou == 1: #右后
                                    position = 6
                                elif zuoyou == 2: #左后
                                    position = 4

                        if vehicle_id == 'ego':
                            position = 0
                        
                        vehicle_data[-(i+1), :] = fit_vehicle_features([
                            vehicle_x,
                            vehicle_y,
                            vehicle_head,
                            vehicle_dict_step['v']*np.cos(self.adjust_heading(head)),
                            vehicle_dict_step['v']*np.sin(self.adjust_heading(head)),
                            a,
                            length,
                            width,
                            position,
                        ] + path_features)
                    else:
                        vehicle_data[-(i+1), :] = fit_vehicle_features([
                            vehicle_x,
                            vehicle_y,
                            vehicle_head,
                            vehicle_dict_step['v']*np.cos(self.adjust_heading(head)),
                            vehicle_dict_step['v']*np.sin(self.adjust_heading(head)),
                            a,
                            length,
                            width,
                        ] + path_features)
                    #vehicle_data[-(i+1), :] = np.array([vehicle_x,vehicle_y,vehicle_head, vehicle_dict_step['v_x']*np.cos(self.adjust_heading(head)),vehicle_dict_step['v_x']*np.sin(self.adjust_heading(head)),vehicle_dict_step['p']])
                # print(np.array(vehicle_data))
                return np.array(vehicle_data)

            data_ego = append_vehicle_dict('ego')
            env_state = np.zeros(shape=(self.num_neighbors+1, self.history_steps, self.features))
            env_state[0] = data_ego

            env_map = np.zeros(shape=(self.num_neighbors+1, self.num_lanes, self.num_waypoints, 4))
            last_ind = self.ego_info_dict['ego']['rel_pos_ind']+18
            if last_ind >= len(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices']):
                last_ind = len(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices']) - 1
            self.goal =  self.transform(np.append([self.goal_center_x,self.goal_center_y], [0]))[:2]
            
    
            if agent_par['three_same_lane']:
                if self.right_lane_id is not None:#右侧车道
                    if self.right_lane_id == self.goal_lane:
                        goal_lane = 1
                    else:
                        goal_lane = 0
                    right_lane = self.road_info_dict_pingjie[str(self.right_lane_id)]['center_vertices']
                    right_refer_tree = KDTree(right_lane)
                    ego_station = np.array([self.x,self.y])
                    right_distance, right_ind = right_refer_tree.query(ego_station)
                    last_right_ind = right_ind+18
                    if last_right_ind >= len(right_lane):
                        last_right_ind = len(right_lane) - 1
                    lane_right_1 = right_lane[right_ind:last_right_ind].tolist()
                    lane_right_1_real = []
                    g = 0
                    angle = ego_yaw
                    for i in range(len(lane_right_1) - 1):
                        p1 = lane_right_1[i]
                        p2 = lane_right_1[i + 1]
                        
                        # 在 p1 和 p2 之间插入 3 个点，总共 4 个点，但不包括 p1 和 p2
                        for j in range(1, 4):  # 只插入 3 个新点
                            x_new = p1[0] + (p2[0] - p1[0]) * j / 3  # 插值计算
                            y_new = p1[1] + (p2[1] - p1[1]) * j / 3
                            if g != 0:
                                angle = self.calculate_orientation(last_x, last_y, x_new, y_new)
                            vehicle_pos = self.transform(np.append([x_new,y_new], [0]))[:2]
                            xb = vehicle_pos[0]
                            yb = vehicle_pos[1]
                            head = self.adjust_angle(angle)
                            yawb = self.adjust_heading(head)
                            lane_right_1_real.append([xb, yb, yawb, goal_lane])
                            last_x = x_new
                            last_y = y_new
                            g += 1

                    if len(lane_right_1_real) < 51:
                        for i in range(len(lane_right_1_real), 51):
                            lane_right_1_real.append([0, 0, 0, 0])
                    if goal_lane == 1:
                        lane_right_1_real[-1] = [self.goal[0], self.goal[1], 0, 1]
                else:
                    lane_right_1_real = [[0, 0, 0, 0] for _ in range(51)]
                
                if self.left_lane_id is not None:
                    if self.left_lane_id == self.goal_lane:
                        goal_lane = 1
                    else:
                        goal_lane = 0
                    left_lane = self.road_info_dict_pingjie[str(self.left_lane_id)]['center_vertices']
                    left_refer_tree = KDTree(left_lane)
                    ego_station = np.array([self.x,self.y])
                    left_distance, left_ind = left_refer_tree.query(ego_station)
                    last_left_ind = left_ind+18
                    if last_left_ind >= len(left_lane):
                        last_v_ind = len(left_lane) - 1
                    lane_left_1 = left_lane[left_ind:last_left_ind].tolist()
                    # print(lane_left_1)
                    # print(ego_station)
                    lane_left_1_real = []
                    g = 0
                    angle = ego_yaw
                    for i in range(len(lane_left_1) - 1):
                        p1 = lane_left_1[i]
                        p2 = lane_left_1[i + 1]
                        
                        # 在 p1 和 p2 之间插入 3 个点，总共 4 个点，但不包括 p1 和 p2
                        for j in range(1, 4):  # 只插入 3 个新点
                            x_new = p1[0] + (p2[0] - p1[0]) * j / 3  # 插值计算
                            y_new = p1[1] + (p2[1] - p1[1]) * j / 3
                            if g != 0:
                                angle = self.calculate_orientation(last_x, last_y, x_new, y_new)
                            vehicle_pos = self.transform(np.append([x_new,y_new], [0]))[:2]
                            xb = vehicle_pos[0]
                            yb = vehicle_pos[1]
                            head = self.adjust_angle(angle)
                            yawb = self.adjust_heading(head)
                            lane_left_1_real.append([xb, yb, yawb, goal_lane])
                            last_x = x_new
                            last_y = y_new
                            g += 1
                    if len(lane_left_1_real) < 51:
                        for i in range(len(lane_left_1_real), 51):
                            lane_left_1_real.append([0, 0, 0, 0])
                    if goal_lane == 1:
                        lane_left_1_real[-1] = [self.goal[0], self.goal[1], 0, 1]
                else:
                    lane_left_1_real = [[0, 0, 0, 0] for _ in range(51)]

            else:
                lane_right_1_real = lane_left_1_real = [[0, 0, 0, 0] for _ in range(51)]

            lane_ego_1 = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][self.ego_info_dict['ego']['rel_pos_ind']:last_ind].tolist()
            lane_ego_1_real = []
            g = 0
            angle = ego_yaw
            if str(ego_lane) == self.goal_lane:
                goal_lane = 1
            else:
                goal_lane = 0

            for i in range(len(lane_ego_1) - 1):
                p1 = lane_ego_1[i]
                p2 = lane_ego_1[i + 1]
                
                # 在 p1 和 p2 之间插入 3 个点，总共 4 个点，但不包括 p1 和 p2
                for j in range(1, 4):  # 只插入 3 个新点
                    x_new = p1[0] + (p2[0] - p1[0]) * j / 3  # 插值计算
                    y_new = p1[1] + (p2[1] - p1[1]) * j / 3
                    if g != 0:
                        angle = self.calculate_orientation(last_x, last_y, x_new, y_new)
                    vehicle_pos = self.transform(np.append([x_new,y_new], [0]))[:2]
                    xb = vehicle_pos[0]
                    yb = vehicle_pos[1]
                    head = self.adjust_angle(angle)
                    yawb = self.adjust_heading(head)
                    lane_ego_1_real.append([xb, yb, yawb, goal_lane])
                    last_x = x_new
                    last_y = y_new
                    g += 1
            if len(lane_ego_1_real) < 51:
                for i in range(len(lane_ego_1_real), 51):
                    lane_ego_1_real.append([0, 0, 0, 0])
            if goal_lane == 1:
                lane_ego_1_real[-1] = [self.goal[0], self.goal[1], 0, 1]
            global_lane_ego = self._global_path_env_waypoints()
            if global_lane_ego is not None:
                lane_ego_1_real = global_lane_ego
            # print('eeeeeeeeeeeeeeee',lane_ego_1_real)
            # for i in lane_ego_1_real:
            #     plt.plot(i[0],i[1],'ro', markersize = 2)

            # waypoint_ego = np.array([lane_ego_1_real, [[0, 0, 0, 0] for _ in range(51)], [[0, 0, 0, 0] for _ in range(51)]])
            waypoint_ego = np.array([lane_ego_1_real, lane_left_1_real, lane_right_1_real])

            # # print('eeeeeeeeeeeeeeee',lane_ego_1_real)
            # for i in lane_ego_1_real:
            #     plt.plot(i[0],i[1],'ro', markersize = 2)
            # for i in lane_right_1_real:
            #     plt.plot(i[0],i[1],'bo', markersize = 2)
            # for i in lane_left_1_real:
            #     plt.plot(i[0],i[1],'co', markersize = 2)

            # plt.plot(self.goal[0],self.goal[1] ,'co', markersize = 2)
            # plt.xlim(-50, 300)
            # plt.ylim(-50, 50)
            # plt.show()
            def get_obs_lane(lane_id, x, y, yaw):
                if lane_id == self.goal_lane:
                    goal_lane = 1
                else:
                    goal_lane = 0
                if lane_id != -1:
                    lane = self.road_info_dict_pingjie[str(lane_id)]['center_vertices']
                    lane_refer_tree = KDTree(lane)
                    station = np.array([x,y])
                    distance, ind = lane_refer_tree.query(station)
                    last_ind = ind+18
                    if last_ind >= len(lane):
                        last_ind = len(lane) - 1
                    lane_1 = lane[ind:last_ind].tolist()
                    lane_1_real = []
                    g = 0
                    angle = yaw
                    for i in range(len(lane_1) - 1):
                        p1 = lane_1[i]
                        p2 = lane_1[i + 1]
                        
                        # 在 p1 和 p2 之间插入 3 个点，总共 4 个点，但不包括 p1 和 p2
                        for j in range(1, 4):  # 只插入 3 个新点
                            x_new = p1[0] + (p2[0] - p1[0]) * j / 3  # 插值计算
                            y_new = p1[1] + (p2[1] - p1[1]) * j / 3
                            if g != 0:
                                angle = self.calculate_orientation(last_x, last_y, x_new, y_new)
                            vehicle_pos = self.transform(np.append([x_new,y_new], [0]))[:2]
                            xb = vehicle_pos[0]
                            yb = vehicle_pos[1]
                            head = self.adjust_angle(angle)
                            yawb = self.adjust_heading(head)
                            lane_1_real.append([xb, yb, yawb, goal_lane])
                            last_x = x_new
                            last_y = y_new
                            g += 1

                    if len(lane_1_real) < 51:
                        for i in range(len(lane_1_real), 51):
                            lane_1_real.append([0, 0, 0, 0])
                else:
                    lane_1_real = [[0, 0, 0, 0] for _ in range(51)]
                # for i in lane_1_real:
                #     plt.plot(i[0],i[1],'co', markersize = 2)

                # plt.xlim(-50, 300)
                # plt.ylim(-50, 50)
                # plt.show()
                return lane_1_real
            


            env_map[0] = waypoint_ego
            # print(ego_x, ego_y,ego_yaw, ego_v)
            n = len(index)
            if n >= 1:
                x_1 = library[index[0]]['x']
                y_1 = library[index[0]]['y']
                v_1 = library[index[0]]['v']
                angle_1 = library[index[0]]['yaw']
                a_1 = library[index[0]]['a']
                length_1 = library[index[0]]['length']
                width_1 = library[index[0]]['width']

                if self.use_predict_map:
                    lane1_1 = self.cvar_map(library[index[0]])
                    lane1_2 = lane1_3 = [[0, 0, 0, 0] for _ in range(51)]
                elif agent_par['all_car_map']:
                    lane1_1 = get_obs_lane(library[index[0]]['lane'], x_1, y_1, angle_1)
                    lane1_2 = lane1_3 = [[0, 0, 0, 0] for _ in range(51)]
                else:
                    time1 = time.time()
                    lane1_1, lane1_2, lane1_3 = self.get_future_lane(x_1, y_1, angle_1)
                    print(time.time() - time1)

                waypoint_1 = np.array([lane1_1, lane1_2, lane1_3])
                

                env_map[1] = waypoint_1

                id_1 = index[0]
                data_1 = append_vehicle_dict(str(id_1))
                env_state[1] = data_1
                # print(data_1)
                # print(waypoint_1.shape)
                # for i in waypoint_1[0]:
                #     plt.plot(i[0], i[1], 'bo', markersize = 0.8)
                # plt.xlim(0, 300)
                # plt.ylim(-50, 50)
                # plt.show()
                vehicle_pos = self.transform(np.append([x_1,y_1], [0]))[:2]
                x_1 = vehicle_pos[0]
                y_1 = vehicle_pos[1]
                head = self.adjust_angle(angle_1)
                angle_1 = self.adjust_heading(head)
                vx_1 = library[index[0]]['v']*np.cos(self.adjust_heading(head))
                vy_1 = library[index[0]]['v']*np.sin(self.adjust_heading(head))
            if n >= 2:
                x_2 = library[index[1]]['x']
                y_2 = library[index[1]]['y']
                v_2 = library[index[1]]['v']
                angle_2 = library[index[1]]['yaw']
                a_2 = library[index[1]]['a']
                length_2 = library[index[1]]['length']
                width_2 = library[index[1]]['width']

                if self.use_predict_map:
                    lane2_1 = self.cvar_map(library[index[1]])
                    lane2_2 = lane2_3 = [[0, 0, 0, 0] for _ in range(51)]
                elif agent_par['all_car_map']:
                    lane2_1 = get_obs_lane(library[index[1]]['lane'], x_2, y_2, angle_2)
                    lane2_2 = lane2_3 = [[0, 0, 0, 0] for _ in range(51)]
                else:
                    lane2_1, lane2_2, lane2_3 = self.get_future_lane(x_2, y_2, angle_2)

                waypoint_2 = np.array([lane2_1, lane2_2, lane2_3])
                env_map[2] = waypoint_2

                id_2 = index[1]
                data_2 = append_vehicle_dict(str(id_2))
                env_state[2] = data_2

                vehicle_pos = self.transform(np.append([x_2,y_2], [0]))[:2]
                x_2 = vehicle_pos[0]
                y_2 = vehicle_pos[1]
                head = self.adjust_angle(angle_2)
                angle_2 = self.adjust_heading(head)
                vx_2 = library[index[1]]['v']*np.cos(self.adjust_heading(head))
                vy_2 = library[index[1]]['v']*np.sin(self.adjust_heading(head))
            if n >= 3:
                x_3 = library[index[2]]['x']
                y_3 = library[index[2]]['y']
                v_3 = library[index[2]]['v']
                angle_3 = library[index[2]]['yaw']
                a_3 = library[index[2]]['a']
                length_3 = library[index[2]]['length']
                width_3 = library[index[2]]['width']

                if self.use_predict_map:
                    lane3_1 = self.cvar_map(library[index[2]])
                    lane3_2 = lane3_3 = [[0, 0, 0, 0] for _ in range(51)]
                elif agent_par['all_car_map']:
                    lane3_1 = get_obs_lane(library[index[2]]['lane'], x_3, y_3, angle_3)
                    lane3_2 = lane3_3 = [[0, 0, 0, 0] for _ in range(51)]
                else:
                    lane3_1, lane3_2, lane3_3 = self.get_future_lane(x_3, y_3, angle_3)

                waypoint_3 = np.array([lane3_1, lane3_2, lane3_3])
                env_map[3] = waypoint_3

                id_3 = index[2]
                data_3 = append_vehicle_dict(str(id_3))
                env_state[3] = data_3

                vehicle_pos = self.transform(np.append([x_3,y_3], [0]))[:2]
                x_3 = vehicle_pos[0]
                y_3 = vehicle_pos[1]
                head = self.adjust_angle(angle_3)
                angle_3 = self.adjust_heading(head)
                vx_3 = library[index[2]]['v']*np.cos(self.adjust_heading(head))
                vy_3 = library[index[2]]['v']*np.sin(self.adjust_heading(head))
            if n >= 4:
                x_4 = library[index[3]]['x']
                y_4 = library[index[3]]['y']
                v_4 = library[index[3]]['v']
                angle_4 = library[index[3]]['yaw']
                a_4 = library[index[3]]['a']
                length_4 = library[index[3]]['length']
                width_4 = library[index[3]]['width']

                if self.use_predict_map:
                    lane4_1 = self.cvar_map(library[index[3]])
                    lane4_2 = lane4_3 = [[0, 0, 0, 0] for _ in range(51)]
                elif agent_par['all_car_map']:
                    lane4_1 = get_obs_lane(library[index[3]]['lane'], x_4, y_4, angle_4)
                    lane4_2 = lane4_3 = [[0, 0, 0, 0] for _ in range(51)]
                else:
                    lane4_1, lane4_2, lane4_3 = self.get_future_lane(x_4, y_4, angle_4)

                waypoint_4 = np.array([lane4_1, lane4_2, lane4_3])
                env_map[4] = waypoint_4

                id_4 = index[3]
                data_4 = append_vehicle_dict(str(id_4))
                env_state[4] = data_4

                vehicle_pos = self.transform(np.append([x_4,y_4], [0]))[:2]
                x_4 = vehicle_pos[0]
                y_4 = vehicle_pos[1]
                head = self.adjust_angle(angle_4)
                angle_4 = self.adjust_heading(head)
                vx_4 = library[index[3]]['v']*np.cos(self.adjust_heading(head))
                vy_4 = library[index[3]]['v']*np.sin(self.adjust_heading(head))
            if n >= 5:
                x_5 = library[index[4]]['x']
                y_5 = library[index[4]]['y']
                v_5 = library[index[4]]['v']
                angle_5 = library[index[4]]['yaw']
                a_5 = library[index[4]]['a']
                length_5 = library[index[4]]['length']
                width_5 = library[index[4]]['width']

                if self.use_predict_map:
                    lane5_1 = self.cvar_map(library[index[4]])
                    lane5_2 = lane5_3 = [[0, 0, 0, 0] for _ in range(51)]
                elif agent_par['all_car_map']:
                    lane5_1 = get_obs_lane(library[index[4]]['lane'], x_5, y_5, angle_5)
                    lane5_2 = lane5_3 = [[0, 0, 0, 0] for _ in range(51)]
                else:
                    lane5_1, lane5_2, lane5_3 = self.get_future_lane(x_5, y_5, angle_5)

                waypoint_5 = np.array([lane5_1, lane5_2, lane5_3])
                env_map[5] = waypoint_5

                id_5 = index[4]
                data_5 = append_vehicle_dict(str(id_5))
                env_state[5] = data_5

                vehicle_pos = self.transform(np.append([x_5,y_5], [0]))[:2]
                x_5 = vehicle_pos[0]
                y_5 = vehicle_pos[1]
                head = self.adjust_angle(angle_5)
                angle_5 = self.adjust_heading(head)
                vx_5 = library[index[4]]['v']*np.cos(self.adjust_heading(head))
                vy_5 = library[index[4]]['v']*np.sin(self.adjust_heading(head))
            if n >= 6:
                x_6 = library[index[5]]['x']
                y_6 = library[index[5]]['y']
                v_6 = library[index[5]]['v']
                angle_6 = library[index[5]]['yaw']
                a_6 = library[index[5]]['a']
                length_6 = library[index[5]]['length']
                width_6 = library[index[5]]['width']

                if self.use_predict_map:
                    lane6_1 = self.cvar_map(library[index[5]])
                    lane6_2 = lane6_3 = [[0, 0, 0, 0] for _ in range(51)]
                elif agent_par['all_car_map']:
                    lane6_1 = get_obs_lane(library[index[5]]['lane'], x_6, y_6, angle_6)
                    lane6_2 = lane6_3 = [[0, 0, 0, 0] for _ in range(51)]
                else:
                    lane6_1, lane6_2, lane6_3 = self.get_future_lane(x_6, y_6, angle_6)

                waypoint_6 = np.array([lane6_1, lane6_2, lane6_3])
                env_map[6] = waypoint_6

                id_6 = index[5]
                data_6 = append_vehicle_dict(str(id_6))
                env_state[6] = data_6

                vehicle_pos = self.transform(np.append([x_6,y_6], [0]))[:2]
                x_6 = vehicle_pos[0]
                y_6 = vehicle_pos[1]
                head = self.adjust_angle(angle_6)
                angle_6 = self.adjust_heading(head)
                vx_6 = library[index[5]]['v']*np.cos(self.adjust_heading(head))
                vy_6 = library[index[5]]['v']*np.sin(self.adjust_heading(head))
                
            frenet_state = self._build_frenet_state(ego_x, ego_y, ego_v, library)
            state = [ego_x/10000,ego_y/10000,ego_v/50,ego_a/15,ego_yaw/(math.pi*2),ego_length/10,ego_width/5] + frenet_state + [
                self.x_goal_min/10000,self.x_goal_max/10000,self.y_goal_min/10000,self.y_goal_max/10000] + [
                x_1/self.sensor_range, y_1/self.sensor_range, v_1/50, a_1/15, angle_1/math.pi, length_1/10, width_1/10,
                x_2/self.sensor_range, y_2/self.sensor_range, v_2/50, a_2/15, angle_2/math.pi, length_2/10, width_2/10,
                x_3/self.sensor_range, y_3/self.sensor_range, v_3/50, a_3/15, angle_3/math.pi, length_3/10, width_3/10,
                x_4/self.sensor_range, y_4/self.sensor_range, v_4/50, a_4/15, angle_4/math.pi, length_4/10, width_4/10,
                x_5/self.sensor_range, y_5/self.sensor_range, v_5/50, a_5/15, angle_5/math.pi, length_5/10, width_5/10,
                x_6/self.sensor_range, y_6/self.sensor_range, v_6/50, a_6/15, angle_6/math.pi, length_6/10, width_6/10]
            
            state = np.array(state)

            # print(state)
            # 判断数组中是否存在 NaN 或 Inf
            has_nan_or_inf = np.isnan(state) | np.isinf(state)
            # 将 NaN 和 Inf 替换成 -1
            state[has_nan_or_inf] = -1
            # print(env_map.shape)
            # for i in env_map:
            #     for j in i[0]:
            #         if i == 0:
            #             ccc = 'ro'
            #         else:
            #             ccc = 'bo'
            #         plt.plot(j[0],j[1],ccc, markersize = 2)
            # plt.show()
            
            
            return state, env_state, env_map
        
        elif self.use_epre_dsac:
            return np.zeros(self.state_dim), None, None


        else:

            if not library.empty:
                if len(library['distance']) < 6:
                    index = library['distance'].sort_values().index
                else:
                    index = library['distance'].nsmallest(5).index

                n = len(index)
                if n >= 1:
                    x_1 = library['positionx'][index[0]] - ego_x
                    y_1 = library['positiony'][index[0]] - ego_y
                    v_1 = library['velocity'][index[0]]
                    angle_1 = library['angle'][index[0]]
                    length_1 = library['length'][index[0]]
                    width_1 = library['width'][index[0]]
                    acc_1 = library['acc'][index[0]]
                if n >= 2:
                    x_2 = library['positionx'][index[1]] - ego_x
                    y_2 = library['positiony'][index[1]] - ego_y
                    v_2 = library['velocity'][index[1]]
                    angle_2 = library['angle'][index[1]]
                    length_2 = library['length'][index[1]]
                    width_2 = library['width'][index[1]]
                    acc_2 = library['acc'][index[1]]
                if n >= 3:
                    x_3 = library['positionx'][index[2]] - ego_x
                    y_3 = library['positiony'][index[2]] - ego_y
                    v_3 = library['velocity'][index[2]]
                    angle_3 = library['angle'][index[2]]
                    length_3 = library['length'][index[2]]
                    width_3 = library['width'][index[2]]
                    acc_3 = library['acc'][index[2]]
                if n >= 4:
                    x_4 = library['positionx'][index[3]] - ego_x
                    y_4 = library['positiony'][index[3]] - ego_y
                    v_4 = library['velocity'][index[3]]
                    angle_4 = library['angle'][index[3]]
                    length_4 = library['length'][index[3]]
                    width_4 = library['width'][index[3]]
                    acc_4 = library['acc'][index[3]]
                if n >= 5:
                    x_5 = library['positionx'][index[4]] - ego_x
                    y_5 = library['positiony'][index[4]] - ego_y
                    v_5 = library['velocity'][index[4]]
                    angle_5 = library['angle'][index[4]]
                    length_5 = library['length'][index[4]]
                    width_5 = library['width'][index[4]]
                    acc_5 = library['acc'][index[4]]
                if n >= 6:
                    x_6 = library['positionx'][index[5]] - ego_x
                    y_6 = library['positiony'][index[5]] - ego_y
                    v_6 = library['velocity'][index[5]]
                    angle_6 = library['angle'][index[5]]
                    length_6 = library['length'][index[5]]
                    width_6 = library['width'][index[5]]
                    acc_6 = library['acc'][index[5]]

            state = [(ego_x - self.guiyi_x)/self.guiyi_x1, (ego_y-self.guiyi_y)/self.guiyi_y1, ego_v/50,ego_a/15, ego_yaw/(math.pi*2),
                    ego_length/10, ego_width/5, (self.x_goal- self.guiyi_x)/self.guiyi_x1, (self.y_goal-self.guiyi_y)/self.guiyi_y1,ego_lane/10, self.goal_lane/10, 
                    x_1/self.sensor_range, y_1/self.sensor_range, v_1/50, acc_1/15, angle_1/(math.pi*2), length_1/10, width_1/5,
                    x_2/self.sensor_range, y_2/self.sensor_range, v_2/50, acc_2/15, angle_2/(math.pi*2), length_2/10, width_2/5,
                    x_3/self.sensor_range, y_3/self.sensor_range, v_3/50, acc_3/15, angle_3/(math.pi*2), length_3/10, width_3/5,
                    x_4/self.sensor_range, y_4/self.sensor_range, v_4/50, acc_4/15, angle_4/(math.pi*2), length_4/10, width_4/5,
                    x_5/self.sensor_range, y_5/self.sensor_range, v_5/50, acc_5/15, angle_5/(math.pi*2), length_5/10, width_5/5,
                    x_6/self.sensor_range, y_6/self.sensor_range, v_6/50, acc_6/15, angle_6/(math.pi*2), length_6/10, width_6/5]
            state = np.array(state)
            state = np.clip(state, -1, 1)

            return state
        
    def determine_travel_direction(self,heading, lanes, index_closest_point):
        # 车辆朝向的单位向量
        car_direction = (math.cos(heading), math.sin(heading))
        # 计算道路方向向量
        if index_closest_point + 1 >= len(lanes):
            index_closest_point = len(lanes) - 2
        point_prev = (lanes[index_closest_point - 1][0], lanes[index_closest_point - 1][1])
        point_next = (lanes[index_closest_point + 1][0], lanes[index_closest_point + 1][1])
        
        road_direction = (point_next[0] - lanes[index_closest_point][0], point_next[1] - lanes[index_closest_point][1])
        
        # 计算车辆朝向与道路方向的点积
        dot_product = car_direction[0] * road_direction[0] + car_direction[1] * road_direction[1]
        
        if dot_product > 0:
            return True #"顺着道路行驶"
        else:
            return False #"逆着道路行驶"
    
    def get_future_lane(self, x, y, yaw):
        lanes = self.lane_info.discretelanes
        robot_ego = np.array((x,y))
        dis_ego = 999999
        ind_ego = 0
        lane1 = []
        lane2 = []
        lane3 = []
        # print('ego', robot_ego, yaw)
        lane_other = []
        other_min = 2
        for i in lanes:
            # print("id: ", i.lane_id, "start: ", i.center_vertices[0,:], "end: ", i.center_vertices[-1,:])
            refer_tree = KDTree(i.center_vertices)
            distance_ego, ind = refer_tree.query(robot_ego)
            direction_ego = self.determine_travel_direction(yaw, i.center_vertices, ind)

            if distance_ego < dis_ego and self.determine_travel_direction(yaw, i.center_vertices, ind):
                id_ego = i.lane_id
                dis_ego = distance_ego
                ind_ego = ind
                if not direction_ego:
                    ind_ego = len(i.center_vertices) - 1 - ind
                    centers = i.center_vertices[::-1]
                else:
                    centers = i.center_vertices
            if distance_ego < other_min and ind <= 10 and self.determine_travel_direction(yaw, i.center_vertices, ind):
                lane_other.append([i.lane_id, len(i.center_vertices), ind])
        # print('id_ego', id_ego, direction_ego, ind_ego, len(centers))

        if len(lane_other) >= 2 and any(sublist[0] == id_ego for sublist in lane_other):
            lane1.append([x, y, yaw, 50])
            lane2.append([x, y, yaw, 50])
            lane3.append([x, y, yaw, 50])
            # print('llllllllllllllllllllllllllllll',lane_other)
        else:
            # print(dir(lanes[0]))
            # if not direction_ego:
            #     print('旁车逆行',x, y, yaw)
            g = 0
            angle = yaw
            last_x = 0
            last_y = 0
            for i in range(ind_ego, len(centers)):
                if g != 0:
                    angle = self.calculate_orientation(last_x, last_y, centers[i][0], centers[i][1])
                lane1.append([centers[i][0], centers[i][1], angle, 50])
                lane2.append([centers[i][0], centers[i][1], angle, 50])
                lane3.append([centers[i][0], centers[i][1], angle, 50])
                g += 1
                last_x = centers[i][0]
                last_y = centers[i][1]
                if g >= 51:
                    break


        if len(lane1) < 51 or len(lane2) < 51 or len(lane3) < 51:
            lane1, lane2, lane3 = self.add_point(id_ego, lanes, yaw, lane1, lane2, lane3)

        if lane1 == lane2:
            lane2 =  [[0, 0, 0, 0] for _ in range(51)]
        if lane1 == lane3:
            lane3 =  [[0, 0, 0, 0] for _ in range(51)]
        lane1_1 = []
        lane2_1 = []
        lane3_1 = []
        for i in lane1:
            if i == [0,0,0,0]:
                lane1_1.append(i)
            else:
                vehicle_pos = self.transform(np.append([i[0],i[1]], [0]))[:2]
                xb = vehicle_pos[0]
                yb = vehicle_pos[1]
                head = self.adjust_angle(i[2])
                yawb = self.adjust_heading(head)
                lane1_1.append([xb, yb, yawb, i[3]])

        if lane2[0] == [0,0,0,0] and lane2[1] == [0,0,0,0]:
            lane2_1 = lane2
        else:
            for i in lane2:
                if i[0] == i[1] == i[2] == i[3] == 0:
                    lane2_1.append(i)
                else:
                    vehicle_pos = self.transform(np.append([i[0],i[1]], [0]))[:2]
                    xb = vehicle_pos[0]
                    yb = vehicle_pos[1]
                    head = self.adjust_angle(i[2])
                    yawb = self.adjust_heading(head)
                    lane2_1.append([xb, yb, yawb, i[3]])

        if lane3[0] == [0,0,0,0] and lane3[1] == [0,0,0,0]:
            lane3_1 = lane3
        else:
            for i in lane3:
                if i[0] == i[1] == i[2] == i[3] == 0:
                    lane3_1.append(i)
                else:
                    vehicle_pos = self.transform(np.append([i[0],i[1]], [0]))[:2]
                    xb = vehicle_pos[0]
                    yb = vehicle_pos[1]
                    head = self.adjust_angle(i[2])
                    yawb = self.adjust_heading(head)
                    lane3_1.append([xb, yb, yawb, i[3]])
        
        
        # print(self.head)
        # for i in lane1:
        #     plt.plot(i[0],i[1],'ro', markersize = 2)
        # for i in lane2:
        #     plt.plot(i[0],i[1],'go', markersize = 2)
        # for i in lane3:
        #     plt.plot(i[0],i[1],'ko', markersize = 2)

        # for i in lane1_1:
        #     plt.plot(i[0],i[1],'ro', markersize = 2)
        # for i in lane2_1:
        #     plt.plot(i[0],i[1],'go', markersize = 2)
        # for i in lane3_1:
        #     plt.plot(i[0],i[1],'ko', markersize = 2)

        # for i in lanes:
        #     lane_list = i.lane_id.split('.')
        #     po = 0.2
        #     if i.lane_id == id_ego:
        #         po = 1
        #     elif i.lane_id in lanes[0]._predecessor:
        #         po = 0.5
        #     center = i.center_vertices
        #     for j in center:
                
        #         plt.plot(j[0],j[1],'bo', markersize = po)

        # plt.plot(x,y,'ko', markersize = 2)
        # plt.plot(self.x, self.y, 'ro', markersize = 2)
        # plt.xlim(0, 300)
        # plt.ylim(-50, 50)
        # plt.show()

        lane1 = lane1_1
        lane2 = lane2_1
        lane3 = lane3_1

        return lane1, lane2, lane3

    def add_point2(self, last_id, lanes, car_yaw, lane, direction): 
        # print('addddd_point2')
        done = False
        last_x = lane[-1][0]
        last_y = lane[-1][1]
        if len(lane)>=2:
            last_point = lane[-1]
            last_last_point = lane[-2]
            yaw = self.calculate_orientation(last_last_point[0], last_last_point[1], last_point[0], last_point[1])
        else:
            yaw = car_yaw
            last_point = lane[-1]
        
        next_lane = []
        for i in lanes:
            # print("id: ", i.lane_id, "start: ", i.center_vertices[0,:], "end: ", i.center_vertices[-1,:])
            refer_tree = KDTree(i.center_vertices)
            distance_ego, ind = refer_tree.query(np.array((last_point[0],last_point[1])))
            if distance_ego < 2 and ind <= 10 and self.determine_travel_direction(yaw, i.center_vertices, ind):# and i.lane_id != last_id:
                cross_product = self.calculate_turn_direction(i.center_vertices, yaw)
                next_lane.append([i.lane_id, i.center_vertices, ind, cross_product])
        if len(next_lane)>0:
            mid_lane = min(next_lane, key=lambda x: abs(x[-1]))
            # 创建一个新的列表，去掉绝对值最小的元素
            other_lane = [item for item in next_lane if item != mid_lane]
            negative_values = [item for item in other_lane if item[-1] < 0]
            positive_values = [item for item in other_lane if item[-1] > 0]

            # 找到负值中绝对值最小的元素
            right_lane = min(negative_values, key=lambda x: abs(x[-1]), default=None)
            # 找到正值中绝对值最小的元素
            left_lane = min(positive_values, key=lambda x: abs(x[-1]), default=None)
            if direction == 'mid':
                if mid_lane is not None:
                    for j in mid_lane[1]:
                        if len(lane) < 51:
                            angle = self.calculate_orientation(last_x, last_y, j[0], j[1])
                            lane.append([j[0], j[1], angle, 50])
                            last_x = j[0]
                            last_y = j[1]
                        else:
                            break
                else:
                    while len(lane) < 51:
                        lane.append([0, 0, 0, 0])  
            elif direction == 'left':
                if left_lane is not None:
                    for j in left_lane[1]:
                        if len(lane) < 51:
                            angle = self.calculate_orientation(last_x, last_y, j[0], j[1])
                            lane.append([j[0], j[1], angle, 50])
                            last_x = j[0]
                            last_y = j[1]
                        else:
                            break
                else:
                    while len(lane) < 51:
                        lane.append([0, 0, 0, 0])  
            else:
                if right_lane is not None:
                    for j in right_lane[1]:
                        if len(lane) < 51:
                            angle = self.calculate_orientation(last_x, last_y, j[0], j[1])
                            lane.append([j[0], j[1], angle, 50])
                            last_x = j[0]
                            last_y = j[1]
                        else:
                            break
                else:
                    while len(lane) < 51:
                        lane.append([0, 0, 0, 0])  
        else:
            while len(lane) < 51:
                lane.append([0, 0, 0, 0])  

        if len(lane) >= 51:
                done = True
        
        return lane, done
    
    def calculate_orientation(self,x1, y1, x2, y2):
        # 计算角度（范围 -pi 到 pi）
        angle = math.atan2(y2 - y1, x2 - x1)
        
        # 转换为 0 到 2pi 的范围
        if angle < 0:
            angle += 2 * math.pi
        
        return angle


    def calculate_turn_direction(self,route_points, yaw): #计算该道路与前路夹角
        # 假设route_points是 [(x1, y1), (x2, y2), ..., (x5, y5)]
        # if len(route_points) >= 10:
        #     x1, y1 = route_points[0]  # 自车位置
        #     x5, y5 = route_points[9]  # 第10个点位置
        # else:
        #     x1, y1 = route_points[0]  # 自车位置
        #     x5, y5 = route_points[-1]  # 第10个点位置
        x1, y1 = route_points[0]  # 自车位置
        x5, y5 = route_points[-1]
        
        # 自车朝向的单位向量
        car_direction_x = math.cos(yaw)
        car_direction_y = math.sin(yaw)
        
        # 计算自车到第5个点的向量
        vector_to_point_5_x = x5 - x1
        vector_to_point_5_y = y5 - y1
        
        # # 计算夹角的点积
        # dot_product = car_direction_x * vector_to_point_5_x + car_direction_y * vector_to_point_5_y
        # vector_to_point_5_length = math.sqrt(vector_to_point_5_x**2 + vector_to_point_5_y**2)
        # cos_theta = dot_product / vector_to_point_5_length
        
        # 计算叉积
        cross_product = car_direction_x * vector_to_point_5_y - car_direction_y * vector_to_point_5_x
        
        # 判断转向
        return cross_product #左：大于零， 右：小于零
    
    def add_point(self, last_id, lanes, car_yaw, lane1, lane2, lane3):
        done1, done2, done3 = False, False, False
        first_len = len(lane1)
        if len(lane1) >= 51:
            done1 = True
        if len(lane2) >= 51:
            done2 = True
        if len(lane3) >= 51:
            done3 = True
        while  not (done1 and done2 and done3):
            if not done1:
                lane1, done1 = self.add_point2(last_id, lanes, car_yaw, lane1, 'mid')
            if not done2:
                lane2, done2 = self.add_point2(last_id, lanes, car_yaw, lane2, 'left')
            if not done3:
                lane3, done3 = self.add_point2(last_id, lanes, car_yaw, lane3, 'right')

        len2 = sum(1 for row in lane2 if row[-1] != 0)
        len3 = sum(1 for row in lane3 if row[-1] != 0)
        if len2 == first_len:
            lane2 = [[0, 0, 0, 0] for _ in range(51)]
        if len3 == first_len:
            lane3 = [[0, 0, 0, 0] for _ in range(51)]

        
        return lane1, lane2, lane3
    
    def transform(self, v):
        return position_to_ego_frame(v, [self.x, self.y, 0], self.head)

    def adjust_heading(self, h):
        return wrap_value(h - self.head, -math.pi, math.pi)

    def adjust_angle(self, angle):
        if angle > math.pi:
            angle -= 2*math.pi
        return angle
    
    def cvar_map(self, data_now, pre_time=25.5, dt=0.5):
        # 假设状态量为 (x, y, yaw, yaw_v, a)，且速度固定为 1m/s
        v = 1  # 车辆速度固定为 1 m/s
        trajectory = []  # 用来存储车辆未来轨迹，格式为[[x1, y1], [x2, y2], ..., [x10, y10]]
        ego_trajectory = []
        
        # 获取初始状态
        x = data_now['x']
        y = data_now['y']
        yaw = data_now['yaw']
        yaw_v = data_now['yaw_v']  # 车速与航向角速度
        a = data_now['a']  # 加速度（在这个模型中未用到，但可以扩展）

        # 假设航向角速度yaw_v不变，保持恒定
        for i in range(int(pre_time / dt)):
            # 计算当前时刻的坐标
            if i == 0:
                # 初始时刻，根据速度v和航向角yaw计算初始位置
                x_new = x + (dt * v * math.cos(yaw))  # x坐标
                y_new = y + (dt * v * math.sin(yaw))  # y坐标
            else:
                # 后续时刻，根据上一个时刻的状态来更新坐标
                x_new = trajectory[i-1][0] + (dt * v * math.cos(trajectory[i-1][2]))  # x坐标
                y_new = trajectory[i-1][1] + (dt * v * math.sin(trajectory[i-1][2]))  # y坐标
            
            # 更新航向角（偏航角），假设偏航角变化率是由yaw_v控制的
            yaw_new = yaw + dt * yaw_v
            yaw_new = yaw_new % (2 * math.pi)
            yaw = yaw_new
            
            # 将新的坐标（x, y）加入轨迹中
            trajectory.append([x_new, y_new, yaw_new])  # 存储坐标和航向角
            vehicle_pos = self.transform(np.append([x_new,y_new], [0]))[:2]
            x_1 = vehicle_pos[0]
            y_1 = vehicle_pos[1]
            head = self.adjust_angle(yaw_new)
            angle_1 = self.adjust_heading(head)
            ego_trajectory.append([x_1, y_1, angle_1, 50])
            
        # 只返回x, y坐标，不包括yaw
        return ego_trajectory

    def reset(self, x_start, y_start, x_goal, y_goal, lane_info, map_file, weather):
        self.start = False
        self.weather = weather
        self.pre_jiaodu = 0
        self.last_ego_lane = -1
        self.guiji_list = []
        self.validator = ComfortValidator(wheelbase=2.8, dt=0.1)
        if (self.weather['rain'] >= 0.15 or self.weather['fog'] >= 0.15 or self.weather['snow'] >= 0.15):
            self.bad_weather = True
        if self.bad_weather:
            self.frame_list = deque(maxlen = 30)
        else:
            self.frame_list = deque(maxlen = 20)
        self.front_list = deque(maxlen = 30)
        self.front_pedestrain_list = deque(maxlen = 10)
        self.last_front_pedestrain = {}
        self.follow = False
        self.change_lane_success = False
        self.is_intersection = False
        self.npc_info_dict = {}
        self.follow_lane = -1
        self.dis_goal = math.sqrt((x_goal-x_start)**2 + (y_goal - y_start)**2)
        # print('disdisdisdis', self.dis_goal)
        print(x_goal, y_goal, x_start, y_start, map_file)
        self.map_file = map_file
        self.x_goal = x_goal
        self.y_goal = y_goal
        if x_start == 171.5452088 and y_start == 12.350735:
            x_start += 1
        self.x_start = x_start
        self.y_start = y_start
        self.rot = 0
        self.epre_state = False
        self.big_turn = False
        self.in_goal_lane = False
        self.keep_str = False
        self.max_acc = True
        self.keep_time = 0

        self.episode_step = 0
        self.guikong_step = 0
        self.last_episode_step = 0
        self.change_lane_sleep = 0
        self.car_info_pre_ego = {}
        self.car_info_pre_obs = {}

        if self.use_epre_dsac:
            self.vehicle_dict = {}
        self.lane_info = lane_info
        
        self.goal_center_x = x_goal
        self.goal_center_y = y_goal

        self.x_goal_min = x_goal
        self.x_goal_max = x_goal
        self.y_goal_min = y_goal
        self.y_goal_max = y_goal

        self.keep = False
        self.goal_center_x = self.x_goal
        self.goal_center_y = self.y_goal
        self.bijingdian = []
        self.tongji_path = None
        self.clear_global_path()
        print(1863,map_file)
        self._apply_scene_speed_limit(map_file)
        if map_file == 'AITownReconstructed_V0103_200518.xodr':  #确实这个地图的归一化需要单独拿出来做
            self.map_type = 'AI_town'
            self.guiyi_x = 784400
            self.guiyi_y = 3352400
            self.guiyi_x1 = 600
            self.guiyi_y1 = 600
            self.x_max = 784910
            self.x_min = 784490
            self.y_max = 3352940
            self.y_min = 3352490
            self.time_max = 20000
            self.target_v = 15
            self.v_max = self.scene_speed_limit
            self.zhenlv = 60/45
        elif map_file == 'HangShaoYongMotorway_V0101_20220601.xodr':
            self.map_type = 'HangZhouWan'
            self.guiyi_x = 0
            self.guiyi_y = 0
            self.guiyi_x1 = 6000
            self.guiyi_y1 = 3500
            self.x_max = 4400
            self.x_min = 2300
            self.y_max = -800
            self.y_min = -2125
            self.time_max = 5000
            self.target_v = 33
            self.v_max = self.scene_speed_limit
            self.zhenlv = 60/20
        elif map_file == 'highway_merge_3_2_401.xodr':
            self.map_type = 'Gaosu'
            self.guiyi_x = 0
            self.guiyi_y = 0
            self.guiyi_x1 = 1000
            self.guiyi_y1 = 30
            self.x_max = 910
            self.x_min = -2
            self.y_max = 4
            self.y_min = -25
            self.time_max = 5000
            self.target_v = 33
            self.v_max = self.scene_speed_limit
            self.zhenlv = 60/50
        elif map_file == '0418follow379.xodr':
            self.map_type = 'follow'
            self.guiyi_x = 0
            self.guiyi_y = 0
            self.guiyi_x1 = 2000
            self.guiyi_y1 = 20
            self.x_max = 2000
            self.x_min = -2000
            self.y_max = 20
            self.y_min = -20
            self.time_max = 5000
            self.target_v = 33
            self.v_max = self.scene_speed_limit
            self.zhenlv = 60/40
        elif map_file == 'highway.xodr' or map_file == 'highway_merge_3_2.xodr':
            self.map_type = 'highway'
            self.guiyi_x = 0
            self.guiyi_y = 0
            self.guiyi_x1 = 2000
            self.guiyi_y1 = 20
            self.x_max = 2000000
            self.x_min = -2000000
            self.y_max = 20000
            self.y_min = -20000
            self.time_max = 5000
            self.target_v = 33
            self.v_max = self.scene_speed_limit
            self.zhenlv = 60/40
        elif map_file == 'tongji.xodr':
            self.tongji = True
            self.map_type = 'AI_town'
            self.guiyi_x = 0
            self.guiyi_y = 0
            self.guiyi_x1 = 2000
            self.guiyi_y1 = 20
            self.x_max = 2000000
            self.x_min = -2000000
            self.y_max = 20000
            self.y_min = -20000
            self.time_max = 5000
            self.target_v = 15
            self.v_max = self.scene_speed_limit
            self.zhenlv = 60/40
        elif map_file =='MT_23-rule_complaince.xodr':
            self.map_type = 'AI_town'
            self.guiyi_x = 0
            self.guiyi_y = 0
            self.guiyi_x1 = 2000
            self.guiyi_y1 = 20
            self.x_max = 2000000
            self.x_min = -2000000
            self.y_max = 20000
            self.y_min = -20000
            self.time_max = 5000
            self.target_v = 2    #yuanshi33
            self.v_max = self.scene_speed_limit
            self.zhenlv = 60/40
        else:
            self.map_type = 'AI_town'
            self.guiyi_x = 0
            self.guiyi_y = 0
            self.guiyi_x1 = 2000
            self.guiyi_y1 = 20
            self.x_max = 2000000
            self.x_min = -2000000
            self.y_max = 20000
            self.y_min = -20000
            self.time_max = 5000
            self.target_v = 18   #yuanshi33
            self.v_max = self.scene_speed_limit
            self.zhenlv = 60/40
            print('1962map_file',map_file)
            print('1963jinru mo ren  xian zhi che su ')

        self._apply_scene_speed_limit(map_file)

        if self.tongji:
            
            if x_start == -663.1692961659872:
                base_path = 'samples_epre_wutfsd/maps/tongjitest/tj02/case_cfg.json'
                base_road = 'samples_epre_wutfsd/maps/tongjitest/tongji_road/tj02.txt'
                print('tongji 02')
            elif x_start == -747.1526916689911:
                base_path = 'samples_epre_wutfsd/maps/tongjitest/tj03/case_cfg.json'
                base_road = 'samples_epre_wutfsd/maps/tongjitest/tongji_road/tj03.txt'
                print('tongji 03')
            elif x_start == -669.8665385170518:
                base_path = 'samples_epre_wutfsd/maps/tongjitest/tj04/case_cfg.json'
                base_road = 'samples_epre_wutfsd/maps/tongjitest/tongji_road/tj04.txt'
                print('tongji 04')
            elif x_start == -748.8692474518673:
                base_path = 'samples_epre_wutfsd/maps/tongjitest/tj05/case_cfg.json'
                base_road = 'samples_epre_wutfsd/maps/tongjitest/tongji_road/tj05.txt'
                print('tongji 05')
            else:
                base_path = 'samples_epre_wutfsd/maps/tongjitest/tj06/case_cfg.json'
                base_road = 'samples_epre_wutfsd/maps/tongjitest/tongji_road/tj06.txt'
                print('tongji 06')


            with open(base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.tongji_path = self.extract_path_points(data)


            tongji_start = [self.tongji_path['init_state']['x'],self.tongji_path['init_state']['y']]
            tongji_target = [self.tongji_path['target_state']['x'],self.tongji_path['target_state']['y']]
            self.bijingdian = []
            for i, pt in enumerate(self.tongji_path['way_points'], 1):
                self.bijingdian.append([pt['x'], pt['y']])

            with open(base_road, 'r') as f:
                tongji_lanes = eval(f.read())
                


        self.zhenlv=1
        self.light_num = 2
        if self.x_start==784785 or self.x_start==784897:
            self.x_start = 784785
            self.y_start = 3352917.5
            self.goal_center_x = 784874.1
            self.goal_center_y = 3352876
        self.start_dis = math.sqrt(math.pow(self.x_start-self.goal_center_x,2) + math.pow(self.y_start-self.goal_center_y,2))
        # self.v_max *= 1.26
        #换道连续
        if self.x_goal > 784850:
            self.big = 1
        else:
            self.big = 0
        self.step_pre = 0
        self.time_pre = time.time()
        self.action_pre = 0
        self.v_des_pre = 0
        self.v_des_steer_pre = 0
        self.v_des = 0
        self.target_lane_pre = 0
        self.action_now = 0
        self.front = {}
        self.front_pedestrain = {}
        self.behind = {}
        self.left_front = {}
        self.left_behind = {}
        self.right_front = {}
        self.right_behind = {}
        self.front_all = {}
        if self.train == False:
            self.time_max*=2

        if not self.tongji:
            #道路信息
            lanes = lane_info.discretelanes

            self.road_info_dict = {}
            self.road_info_index = {}
            self.road_info_pingjie = {}

            #查找自车初始车道和目标所在车道
            robot_goal = np.array([self.goal_center_x,self.goal_center_y])
            robot_ego = np.array((self.x_start,self.y_start))
            dis_goal = 999999
            dis_ego = 999999
            id_goal = '-1.-1.-1.-1'
            id_ego = '-1.-1.-1.-1'
            goal_id = []
            ego_id = []
            temp_goal_dict = {}

            for i in lanes:
                refer_tree = KDTree(i.center_vertices)
                distance_goal, ind_goal = refer_tree.query(robot_goal)
                distance_ego, ind_ego = refer_tree.query(robot_ego)
                if distance_goal < dis_goal:
                    id_goal = i.lane_id
                    dis_goal = distance_goal
                    
                if distance_ego < dis_ego:
                    id_ego = i.lane_id
                    dis_ego = distance_ego

            

            goal_id.append(id_goal)
            ego_id.append(id_ego)
            
            temp_dict_goal = {}
            temp_id_goal = []
            

            temp_dict_ego = {}
            temp_id_ego = []
            for i in lanes:
                dict_test_goal = {}
                dict_test_ego = {}
                if int(i.lane_id.split('.')[0]) == int(id_goal.split('.')[0]) and int(i.lane_id.split('.')[1]) == int(id_goal.split('.')[1]):
                    dict_test_goal['center_vertices'] = copy(i.center_vertices)
                    dict_test_goal['node_start'] = np.array(i.center_vertices[0,:])
                    dict_test_goal['node_end'] = np.array(i.center_vertices[-1,:])
                    left = np.array(copy(i.left_vertices))
                    right = np.array(copy(i.right_vertices))
                    k = (left - right) ** 2
                    dict_test_goal['width'] = np.sqrt(k[:,0] + k[:,1])
                    temp_dict_goal[i.lane_id] = dict_test_goal
                    temp_id_goal.append(i.lane_id)
                if int(i.lane_id.split('.')[0]) == int(id_ego.split('.')[0]) and int(i.lane_id.split('.')[1]) == int(id_ego.split('.')[1]):
                    dict_test_ego['center_vertices'] = copy(i.center_vertices)
                    dict_test_ego['node_start'] = np.array(i.center_vertices[0,:])
                    dict_test_ego['node_end'] = np.array(i.center_vertices[-1,:])
                    left = np.array(copy(i.left_vertices))
                    right = np.array(copy(i.right_vertices))
                    k = (left - right) ** 2
                    dict_test_ego['width'] = np.sqrt(k[:,0] + k[:,1])
                    temp_dict_ego[i.lane_id] = dict_test_ego
                    temp_id_ego.append(i.lane_id)
            #取出所有车道与车道id（除开goal或者ego）
            temp_dict_g = {}
            temp_id_g = []
            temp_dict_e = {}
            temp_id_e = []
            for i in lanes:
                dict_test_g = {}
                dict_test_e = {}
                if i.lane_id not in temp_id_goal:
                    dict_test_g['center_vertices'] = copy(i.center_vertices)
                    dict_test_g['node_start'] = np.array(i.center_vertices[0,:])
                    dict_test_g['node_end'] = np.array(i.center_vertices[-1,:])
                    left = np.array(copy(i.left_vertices))
                    right = np.array(copy(i.right_vertices))
                    k = (left - right) ** 2
                    dict_test_g['width'] = np.sqrt(k[:,0] + k[:,1])
                    temp_dict_g[i.lane_id] = dict_test_g
                    temp_id_g.append(i.lane_id)
                if i.lane_id not in temp_id_ego:
                    dict_test_e['center_vertices'] = copy(i.center_vertices)
                    dict_test_e['node_start'] = np.array(i.center_vertices[0,:])
                    dict_test_e['node_end'] = np.array(i.center_vertices[-1,:])
                    left = np.array(copy(i.left_vertices))
                    right = np.array(copy(i.right_vertices))
                    k = (left - right) ** 2
                    dict_test_e['width'] = np.sqrt(k[:,0] + k[:,1])
                    temp_dict_e[i.lane_id] = dict_test_e
                    temp_id_e.append(i.lane_id)
            
            road_info_dict_temp = {}
            road_info_id_temp = {}
            
            #车道拼接
            error = 0.5
            m = 0
            l_t=[]
            #车道拼接(根据ego)
            for x_ego,y_ego in temp_dict_ego.items():
                while True:
                    k = {}
                    l_test = []
                    k['center_vertices'] = y_ego['center_vertices']
                    k['width'] = y_ego['width']
                    l_test.append(x_ego)
                    while True:
                        i = 0
                        k1 = 0
                        num = 0
                        # angle_temp = math.pi * 4
                        dis_temp = 999
                        id_temp = '-1.-1.-1.-1'
                        for x,y in temp_dict_e.items():
                            if abs(np.linalg.norm(k['center_vertices'][-1,:]-y['center_vertices'][0,:]))<=error:
                                dis = abs(np.linalg.norm(y['center_vertices'][-1,:]-[self.goal_center_x,self.goal_center_y]))
                                if dis_temp > dis:
                                    dis_temp = dis
                                    if id_temp != id_goal:
                                        id_temp = x
                                
                                if x == id_goal:
                                    id_temp = id_goal
                                num += 1

                        for x,y in temp_dict_e.items():
                            if abs(np.linalg.norm(k['center_vertices'][0,:]-y['center_vertices'][-1,:]))<=error:
                                if x in l_test:
                                    break
                                k['center_vertices'] = np.vstack((y['center_vertices'],k['center_vertices']))
                                k['width'] = np.concatenate((y['width'],k['width']))
                                l_test.insert(0,x)
                                k1 += 1
                            if abs(np.linalg.norm(k['center_vertices'][-1,:]-y['center_vertices'][0,:]))<=error:
                                if x in l_test:
                                    break
                                if num <= 1:
                                    k['center_vertices'] = np.vstack((k['center_vertices'],y['center_vertices']))
                                    k['width'] = np.concatenate((k['width'],y['width']))
                                    l_test.append(x)
                                    i += 1
                                else:
                                    if x == id_temp:
                                        k['center_vertices'] = np.vstack((k['center_vertices'],y['center_vertices']))
                                        k['width'] = np.concatenate((k['width'],y['width']))
                                        l_test.append(x)
                                        i += 1
                            if i == 1:
                                break
                        if i == 0 and k1 == 0:
                            break
                    if l_test not in l_t:
                        road_info_dict_temp[str(m)] = k
                        road_info_id_temp[str(m)] = l_test
                        m += 1
                        l_t.append(l_test)
                    break
            
            #车道拼接(根据goal)
            for x_goal,y_goal in temp_dict_goal.items():
                while True:
                    k = {}
                    l_test = []
                    k['center_vertices'] = y_goal['center_vertices']
                    k['width'] = y_goal['width']
                    l_test.append(x_goal)
                    while True:
                        i = 0
                        k1 = 0
                        num = 0
                        dis_temp = 99999
                        id_temp = '-1.-1.-1.-1'
                        for x,y in temp_dict_g.items():
                            if abs(np.linalg.norm(k['center_vertices'][0,:]-y['center_vertices'][-1,:]))<=error:
                                dis = abs(np.linalg.norm(y['center_vertices'][0,:]-[self.goal_center_x,self.goal_center_y]))
                                if dis_temp > dis:
                                    dis_temp = dis
                                    if id_temp != id_ego:
                                        id_temp = x
                                
                                if x == id_ego:
                                    id_temp = id_ego
                                num += 1
                        
                        for x,y in temp_dict_g.items():
                            if abs(np.linalg.norm(k['center_vertices'][0,:]-y['center_vertices'][-1,:]))<=error:
                                if x in l_test:
                                    break
                                if num <= 1:
                                    k['center_vertices'] = np.vstack((y['center_vertices'],k['center_vertices']))
                                    # k['width'] = np.vstack((y['width'],k['width']))
                                    k['width'] = np.concatenate((y['width'],k['width']))
                                    l_test.insert(0,x)
                                    i += 1
                                else:
                                    if x == id_temp:
                                        k['center_vertices'] = np.vstack((y['center_vertices'],k['center_vertices']))
                                        k['width'] = np.concatenate((y['width'],k['width']))
                                        l_test.insert(0,x)
                                        i += 1
                            if abs(np.linalg.norm(k['center_vertices'][-1,:]-y['center_vertices'][0,:]))<=error:
                                if x in l_test:
                                    break
                                k['center_vertices'] = np.vstack((k['center_vertices'],y['center_vertices']))
                                k['width'] = np.concatenate((k['width'],y['width']))
                                l_test.append(x)
                                k1 += 1
                            if i == 1:
                                break
                        if i == 0 and k1 == 0:
                            break
                    if l_test not in l_t:
                        road_info_dict_temp[str(m)] = k
                        road_info_id_temp[str(m)] = l_test
                        m += 1
                        l_t.append(l_test)
                    break
            


            #拼接好的道路后处理（去除包含车道)
            del_num = []
            for i in range(len(road_info_id_temp)):
                j = i+1
                while j < len(road_info_id_temp):
                    if self.check_containment(road_info_id_temp[str(i)],road_info_id_temp[str(j)]):
                        if len(road_info_id_temp[str(i)]) < len(road_info_id_temp[str(j)]):
                            del_num.append(str(i))
                        else:
                            del_num.append(str(j))
                    j += 1


            del_num = list(set(del_num))

            for key in del_num:
                if key in road_info_dict_temp:
                    del road_info_dict_temp[key]
                if key in road_info_id_temp:
                    del road_info_id_temp[key]
            
            #复制
            road_info_dict_temp_temp_temp = {}
            road_info_id_temp_temp_temp = {}
            for x,y in road_info_id_temp.items():
                road_info_id_temp_temp_temp[str(x)] = y
            for x,y in road_info_dict_temp.items():
                road_info_dict_temp_temp_temp[str(x)] = y


            #拼接好的道路后处理（留下包含ego和goal的车道)
            del_num = []
            for x,y in road_info_id_temp.items():
                if not (self.check_common_elements(y,temp_id_goal)  or  self.check_common_elements(y,temp_id_ego)):
                    if not (self.check_common_elements(y,goal_id)  or  self.check_common_elements(y,ego_id)):
                        del_num.append(x)
            if self.map_type == 'AI_town':
                for key in del_num:
                    if key in road_info_dict_temp:
                        del road_info_dict_temp[key]
                    if key in road_info_id_temp:
                        del road_info_id_temp[key]
            
            road_info_dict_temp_temp = {}
            road_info_id_temp_temp = {}
            i = 0
            for x,y in road_info_id_temp.items():
                road_info_id_temp_temp[str(i)] = y
                i += 1
            i = 0
            # for x,y in road_info_dict_temp.items():
            #     road_info_dict_temp_temp[str(i)] = y
            #     i += 1

            # for key, value in road_info_dict_temp_temp.items():
            #     road_1 =value['center_vertices']
            #     road_1 = np.array(road_1)
            #     # 提取x坐标和y坐标
            #     x = road_1[:,0]
            #     y = road_1[:,1]
            #     plt.plot(x, y,markersize = 0.2)
            # if self.tongji:
            #     for j in bijingdian:
            #         plt.plot(j[0], j[1], 'ko', zorder = 10,markersize = 3)
            # plt.plot(self.x_start, self.y_start, 'ko', zorder = 10,markersize = 5)
            # plt.plot(self.x_goal, self.y_goal, 'ko',zorder = 10,markersize = 7)
            # plt.show()
            
            #不同时包含goal和ego
            if len(road_info_dict_temp_temp) == 0:
                road_info_dict_temp_temp = road_info_dict_temp_temp_temp
                road_info_id_temp_temp = road_info_id_temp_temp_temp

            road_info_dict_temp = {}
            road_info_id_temp = {}
            i = 0
            for x,y in road_info_id_temp_temp.items():
                road_info_id_temp[str(i)] = y
                i += 1
            i = 0
            for x,y in road_info_dict_temp_temp.items():
                road_info_dict_temp[str(i)] = y
                i += 1
                 
            #车道id赋值，按照自车位置，在自车左方的最大
            for i in range(len(road_info_dict_temp)):
                for j in range(0, len(road_info_dict_temp)-i-1):
                    a = road_info_dict_temp[str(j)]['center_vertices'][1,:] - road_info_dict_temp[str(j)]['center_vertices'][0,:]
                    b = road_info_dict_temp[str(j+1)]['center_vertices'][0,:] - road_info_dict_temp[str(j)]['center_vertices'][0,:]
                    if b[0] == 0 and b[1] == 0:
                        a = road_info_dict_temp[str(j)]['center_vertices'][-2,:] - road_info_dict_temp[str(j)]['center_vertices'][-1,:]
                        b = road_info_dict_temp[str(j+1)]['center_vertices'][-1,:] - road_info_dict_temp[str(j)]['center_vertices'][-1,:]
                    if a[0] * b[1] - a[1] * b[0] < 0:
                        road_info_dict_temp[str(j)], road_info_dict_temp[str(j+1)] = road_info_dict_temp[str(j+1)], road_info_dict_temp[str(j)]
                        road_info_id_temp[str(j)], road_info_id_temp[str(j+1)] = road_info_id_temp[str(j+1)], road_info_id_temp[str(j)]

            def remove_overlap(road1, road2):
                # 获取第一条道路和第二条道路的路点列表
                vertices1 = [tuple(vertex) for vertex in road1['center_vertices']]
                vertices2 = [tuple(vertex) for vertex in road2['center_vertices']]
                
                # 找到两条道路中的重叠路点，并将其从第一条道路中删除
                overlap_vertices = set(vertices1).intersection(vertices2)
                road1['center_vertices'] = [list(vertex) for vertex in vertices1 if tuple(vertex) not in overlap_vertices]
                
                # 删除第一条道路中的重叠路点对应的宽度信息
                indices_to_delete = [i for i, vertex in enumerate(vertices1) if vertex in overlap_vertices]
                road1['width'] = np.delete(road1['width'], indices_to_delete)

                # 将道路中心点和宽度信息重新转换为 NumPy 数组
                road1['center_vertices'] = np.array(road1['center_vertices'])
                road1['width'] = np.array(road1['width'])

            for i in range(len(road_info_dict_temp)):
                for j in range(i+1, len(road_info_dict_temp)):
                    remove_overlap(road_info_dict_temp[str(j)], road_info_dict_temp[str(i)])


            #得到self.road_info_pingjie
            for x,y in road_info_dict_temp.items():
                dict_test = {}
                dict_test['node_start'] = np.array(y['center_vertices'][0,:])
                dict_test['node_end'] = np.array(y['center_vertices'][-1,:])
                # dict_test['width'] = copy(y['width'])
                for i,j in road_info_id_temp.items():
                    if i == x:
                        dict_test['id'] = j
                self.road_info_pingjie[x] = dict_test
            

            #路点稀疏化
            self.road_info_dict_pingjie = road_info_dict_temp
            for x, y in self.road_info_dict_pingjie.items():
                self.road_info_dict_pingjie[x]['center_vertices'] = y['center_vertices'][::4]
                self.road_info_dict_pingjie[x]['width'] = y['width'][::4]
            
            #计算phi_road, curvature, station
            for x, y in self.road_info_dict_pingjie.items():
                self.road_info_dict_pingjie[x]['phi_road'], self.road_info_dict_pingjie[x]['curvature'], self.road_info_dict_pingjie[x]['station'] = self.refer_cacul(y['center_vertices'][:,0],y['center_vertices'][:,1])
       
        else:
            self.road_info_dict_pingjie = {
                '0': {
                    'center_vertices': np.array(tongji_lanes, dtype=np.float64),
                    'width': np.full(len(tongji_lanes), 3.45)  # 更高效的初始化
                }
            }
            for x, y in self.road_info_dict_pingjie.items():
                self.road_info_dict_pingjie[x]['phi_road'], self.road_info_dict_pingjie[x]['curvature'], self.road_info_dict_pingjie[x]['station'] = self.refer_cacul(y['center_vertices'][:,0],y['center_vertices'][:,1])

        ludian_len = 2
        self.x_start = x_start
        self.y_start = y_start
        if len(self.road_info_dict_pingjie) != 0:
            if len(self.road_info_dict_pingjie[str(0)]['station']) >=2:
                ludian_len = self.road_info_dict_pingjie[str(0)]['station'][1] - self.road_info_dict_pingjie[str(0)]['station'][0]
        self.guikong = kongzhi(self.road_info_dict_pingjie,self.scenario_dt,ludian_len, self.x_start, self.y_start, self.x_goal, self.y_goal, self.map_type, map_file)
        self.guikong.goal_state = False
        
        

        m = 999999
        self.goal_lane = 0
        self.goal_dis = 0
        goal_ind = 0
        goal_station = np.array([self.x_goal, self.y_goal])
        start_station = np.array([self.x_start, self.y_start])
        self.goal_center_x = self.x_goal
        self.goal_center_y = self.y_goal
        for k1,v1 in self.road_info_dict_pingjie.items():
            refer_tree = KDTree(v1['center_vertices'])
            distance, ind = refer_tree.query(goal_station)
            if distance <= m:
                self.goal_lane = int(k1)
                self.goal_dis = distance
                goal_ind = ind
                m = distance

        xiangliang_goal = goal_station - self.road_info_dict_pingjie[str(self.goal_lane)]['center_vertices'][goal_ind,:]
        xiangliang_line = self.road_info_dict_pingjie[str(self.goal_lane)]['center_vertices'][goal_ind,:] - self.road_info_dict_pingjie[str(self.goal_lane)]['center_vertices'][goal_ind-1,:]
        if xiangliang_goal[0]*xiangliang_line[1] - xiangliang_goal[1]*xiangliang_line[0] > 0:
            zuoyou = 1 #车在道路中心线右侧
        elif xiangliang_goal[0]*xiangliang_line[1] - xiangliang_goal[1]*xiangliang_line[0] < 0:
            zuoyou = 2 #车在道路中心线左侧
        else:
            zuoyou = 0 #车在道路中心线
        self.goal_zuoyou = zuoyou
        self._save_global_plan_visualization()
    

        if self.show_map:

            for i in range(len(self.road_info_dict_pingjie)):
                road_1 =self.road_info_dict_pingjie[str(i)]['center_vertices']
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

            plt.plot(self.x_start, self.y_start, 'wo', zorder = 10)
            plt.plot(self.x_goal, self.y_goal, 'ko',zorder = 10)
            plt.gca().set_aspect('equal')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Lane Center Points')
            plt.grid(True)
            plt.ion()
            plt.show()

    def extract_path_points(self,json_data):
        path_data = {
            'init_state': None,
            'way_points': [],
            'target_state': None
        }
        
        # 遍历所有角色
        for role_id, role_data in json_data['roles'].items():
            if 'schedule' in role_data:
                schedule = role_data['schedule']
                
                # 提取初始状态
                if 'init_state' in schedule:
                    init = schedule['init_state']
                    path_data['init_state'] = {
                        'x': init['x'],
                        'y': init['y'],
                        'z': init['z'],
                        'orientation_z': init['orientation_z'],
                        'v': init['v']
                    }
                
                # 提取必经点
                if 'way_point_state' in schedule:
                    for point in schedule['way_point_state']:
                        path_data['way_points'].append({
                            'x': point['x'],
                            'y': point['y'],
                            'z': point['z'],
                            'orientation_z': point['orientation_z'],
                            'v': point['v']
                        })
                
                # 提取目标状态
                if 'target_state' in schedule:
                    target = schedule['target_state']
                    path_data['target_state'] = {
                        'x': target['x'],
                        'y': target['y'],
                        'z': target['z'],
                        'orientation_z': target['orientation_z'],
                        'v': target['v']
                    }
        
        return path_data

    def _continuous_action(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.size < 2:
            target_speed = float(action[0]) if action.size else 0.0
            lateral_offset = 0.0
        else:
            target_speed = float(action[0])
            lateral_offset = float(action[1])

        speed_limit = getattr(self, "v_max1", self.v_max)
        target_speed = self.limit(target_speed, 0.0, min(8.0, speed_limit))
        lateral_limit = float(agent_par.get("action_high_limit", [8.0, 5.0])[1])
        lateral_offset = self.limit(lateral_offset, -abs(lateral_limit), abs(lateral_limit))
        return target_speed, lateral_offset

    def _cal_continuous_control(self, target_speed, lateral_offset):
        target_lane = int(getattr(self, "ego_lane", 0))
        if str(target_lane) not in self.road_info_dict_pingjie:
            target_lane = int(getattr(self, "target_lane_pre", 0))
        if str(target_lane) not in self.road_info_dict_pingjie:
            return [0.0, 0.0]

        self.target_lane_pre = target_lane
        self.action_pre = 0
        self.action_now = 0

        lane_info = self.road_info_dict_pingjie[str(target_lane)]
        center_vertices = lane_info["center_vertices"]
        if len(center_vertices) < 2:
            return [0.0, 0.0]

        ego_x_houlun = self.ego_x - self.ego_length / 1.7 / 2 * math.cos(self.ego_yaw)
        ego_y_houlun = self.ego_y - self.ego_length / 1.7 / 2 * math.sin(self.ego_yaw)
        ego_station = np.array([ego_x_houlun, ego_y_houlun])
        _, ind = KDTree(center_vertices).query(ego_station)

        station = lane_info.get("station")
        if station is not None and len(station) > 1:
            ds = max(0.1, float(station[min(ind + 1, len(station) - 1)] - station[ind]))
        else:
            ds = 0.5
        lookahead = max(3.0, min(20.0, self.ego_v * 0.8 + 4.0))
        target_ind = self.limit(int(ind + lookahead / ds), 0, len(center_vertices) - 1)

        x_ref = float(center_vertices[target_ind][0])
        y_ref = float(center_vertices[target_ind][1])
        phi_road = lane_info.get("phi_road")
        if phi_road is not None and len(phi_road) > target_ind:
            yaw_ref = float(phi_road[target_ind])
        else:
            prev_ind = max(0, target_ind - 1)
            dx = center_vertices[target_ind][0] - center_vertices[prev_ind][0]
            dy = center_vertices[target_ind][1] - center_vertices[prev_ind][1]
            yaw_ref = math.atan2(dy, dx)

        lane_widths = lane_info.get("width")
        if lane_widths is not None and len(lane_widths) > ind:
            lane_bound = max(0.5, float(lane_widths[ind]) * 0.5 - 0.2)
            lateral_offset = self.limit(lateral_offset, -lane_bound, lane_bound)

        x_ref = x_ref - lateral_offset * math.sin(yaw_ref)
        y_ref = y_ref + lateral_offset * math.cos(yaw_ref)

        alpha = math.atan2(y_ref - ego_y_houlun, x_ref - ego_x_houlun) - self.ego_yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        ld = max(0.1, math.hypot(y_ref - ego_y_houlun, x_ref - ego_x_houlun))
        rot = math.atan2(2 * self.ego_length * math.sin(alpha), ld)
        acc = self.guikong.pid_lon.cal_output(self.ego_v - target_speed)

        if self.safe != 0:
            acc = -6 if self.safe == -5 else -4

        self.rot = rot
        self.v_des = target_speed
        self.target_lateral_offset = lateral_offset
        self.rl_lateral_offset = lateral_offset
        return [self.limit(acc, -6.0, 3.0), self.limit(rot, -0.85, 0.85)]

    def cal_control(self, action, step):
        target_speed, lateral_offset = self._continuous_action(action)
        self.v_des = target_speed
        self.rl_lateral_offset = lateral_offset
        self.target_lateral_offset = lateral_offset
        self.target_lane_d = lateral_offset
        self.rl_target_d = lateral_offset
        return self._cal_continuous_control(target_speed, lateral_offset)

    def front_or_back(self, waypoint):
        direction = 'None'
        ego_station = np.array([self.ego_x,self.ego_y])
        previous_ego_station = np.array([self.ego_pre_x,self.ego_pre_y])
        waypoint = np.array(waypoint)
        v = ego_station - previous_ego_station
        u = waypoint - previous_ego_station
        w = np.cross(v, u)
        if w > 0:
            direction = 'left'
        elif w<0:
            direction = 'right'
        return direction

    def left_or_right(self, waypoint):
        direction = 'None'
        
        waypoint = np.array(waypoint)
        try:
            ego_station = self.road_info_dict_pingjie[str(self.ego_lane)]['center_vertices'][self.ego_info_dict['ego']['rel_pos_ind']]
            previous_ego_station = self.road_info_dict_pingjie[str(self.ego_lane)]['center_vertices'][self.ego_info_dict['ego']['rel_pos_ind']-1]
        except:
            try:
                ego_station = self.road_info_dict_pingjie[str(self.ego_lane)]['center_vertices'][self.ego_info_dict['ego']['rel_pos_ind']+1]
                previous_ego_station = self.road_info_dict_pingjie[str(self.ego_lane)]['center_vertices'][self.ego_info_dict['ego']['rel_pos_ind']]
            except:
                ego_station = np.array([self.ego_x,self.ego_y])
                previous_ego_station = np.array([self.ego_pre_x,self.ego_pre_y])

        v = ego_station - previous_ego_station
        u = waypoint - previous_ego_station
        w = np.cross(v, u)
        if w > 0:
            direction = 'left'
        elif w<0:
            direction = 'right'
        return direction
    
    def shun_ni(self, waypoint, last_waypoint, next_waypoint):

        try:
            ego_station = self.road_info_dict_pingjie[str(self.ego_lane)]['center_vertices'][self.ego_info_dict['ego']['rel_pos_ind']]
            previous_ego_station = self.road_info_dict_pingjie[str(self.ego_lane)]['center_vertices'][self.ego_info_dict['ego']['rel_pos_ind']-1]
        except:
            try:
                ego_station = self.road_info_dict_pingjie[str(self.ego_lane)]['center_vertices'][self.ego_info_dict['ego']['rel_pos_ind']+1]
                previous_ego_station = self.road_info_dict_pingjie[str(self.ego_lane)]['center_vertices'][self.ego_info_dict['ego']['rel_pos_ind']]
            except:
                ego_station = np.array([self.ego_x,self.ego_y])
                previous_ego_station = np.array([self.ego_pre_x,self.ego_pre_y])
                
        waypoint = np.array(waypoint)
        v = ego_station - previous_ego_station
    
        
        if last_waypoint is not None:
            last_waypoint = np.array(last_waypoint)
            lane_direction = waypoint - last_waypoint
        elif next_waypoint is not None:
            next_waypoint = np.array(next_waypoint)
            lane_direction = next_waypoint - waypoint
        else:
            # 如果没有参考点，默认逆行
            return -1
        v_norm = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v
        lane_norm = lane_direction / np.linalg.norm(lane_direction) if np.linalg.norm(lane_direction) > 0 else lane_direction
        # 计算向量夹角余弦值
        cos_theta = np.dot(v_norm, lane_norm)
        
        # 判断顺逆（余弦值接近1表示顺行，接近-1表示逆行）
        if cos_theta > 0:  # 夹角小于90度
            return 1  # 顺行
        else:
            return -1  # 逆行

    
    def angle_between_vectors(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        cosine_theta = dot_product / (magnitude1 * magnitude2)
        theta_rad = np.arccos(cosine_theta)
        theta_deg = np.degrees(theta_rad)
        return theta_deg

    def jiajiao(self, waypoint, pre_waypoint):
        ego_station = np.array([self.ego_x,self.ego_y])
        previous_ego_station = np.array([self.ego_pre_x,self.ego_pre_y])
        waypoint = np.array(waypoint)
        pre_waypoint = np.array(pre_waypoint)
        v = ego_station - previous_ego_station
        u = waypoint - pre_waypoint
        angle = self.angle_between_vectors(v, u)
        return angle

    def jiajiao_goal(self, waypoint, pre_waypoint):
        goal_station = np.array([self.goal_center_x,self.goal_center_y])
        waypoint = np.array(waypoint)
        pre_waypoint = np.array(pre_waypoint)
        v = goal_station - pre_waypoint
        u = waypoint - pre_waypoint
        angle = self.angle_between_vectors(v, u)
        return angle

    def get_jiaodu(self):
        ego_lane = self.ego_info_dict['ego']['lane']
        ego_station = np.array([self.ego_x,self.ego_y])
        
        refer_tree = KDTree(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'])
        distance, ind = refer_tree.query(ego_station)
        waypoint = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind]
        pre_waypoint = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind - 1]
        jiaodu = self.jiajiao(waypoint, pre_waypoint)
        if math.isnan(jiaodu):
            jiaodu = self.pre_jiaodu
            
        else:
            self.pre_jiaodu = jiaodu
        
        return jiaodu
    
    def get_goal_jiaodu(self):
        ego_lane = self.ego_info_dict['ego']['lane']
        try:
            waypoint = self.road_info_dict_pingjie[str(self.ego_lane)]['center_vertices'][self.ego_info_dict['ego']['rel_pos_ind']]
            pre_waypoint = self.road_info_dict_pingjie[str(self.ego_lane)]['center_vertices'][self.ego_info_dict['ego']['rel_pos_ind']-1]
        except:
            try:
                waypoint = self.road_info_dict_pingjie[str(self.ego_lane)]['center_vertices'][self.ego_info_dict['ego']['rel_pos_ind']+1]
                pre_waypoint = self.road_info_dict_pingjie[str(self.ego_lane)]['center_vertices'][self.ego_info_dict['ego']['rel_pos_ind']]
            except:
                waypoint = np.array([self.ego_x,self.ego_y])
                pre_waypoint = np.array([self.ego_pre_x,self.ego_pre_y])
                
        jiaodu = self.jiajiao_goal(waypoint, pre_waypoint)
        return jiaodu

    def get_other_pos(self, other_x, other_y):
        ego_lane = self.ego_info_dict['ego']['lane']
        other_station = np.array([other_x,other_y])
        refer_tree = KDTree(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'])
        distance, ind = refer_tree.query(other_station)
        waypoint = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind]
        try:
            next_waypoint = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind + 1]
        except:
            next_waypoint = waypoint + np.array([1,0])
        jiaodu, diraction = self.jiajiao_other(waypoint, next_waypoint, other_station)
        if jiaodu >= 90:
            if distance < self.road_info_dict_pingjie[str(ego_lane)]['width'][ind]/2:
                position = "hou"
            elif distance > self.road_info_dict_pingjie[str(ego_lane)]['width'][ind]/2 and distance < self.road_info_dict_pingjie[str(ego_lane)]['width'][ind]*3/2:
                if diraction == "left":
                    position = "zuo_hou"
                elif diraction == "right":
                    position = "you_hou"
                else:
                    position = None
            else:
                position = 'out'
        elif jiaodu < 90:
            if distance < self.road_info_dict_pingjie[str(ego_lane)]['width'][ind]/2:
                position = "qian"
            elif distance > self.road_info_dict_pingjie[str(ego_lane)]['width'][ind]/2 and distance < self.road_info_dict_pingjie[str(ego_lane)]['width'][ind]*3/2:
                if diraction == "left":
                    position = "zuo_qian"
                elif diraction == "right":
                    position = "you_qian"
                else:
                    position = None
            else:
                position = 'out'
        return position


    def jiajiao_other(self, waypoint, next_waypoint ,other_station):
        ego_station = np.array([self.ego_x,self.ego_y])
        waypoint = np.array(waypoint)
        next_waypoint = np.array(next_waypoint)
        v = other_station - waypoint
        u = next_waypoint - waypoint
        v1 = other_station - ego_station
        u1 = next_waypoint - ego_station
        angle = self.angle_between_vectors(v1, u1)
        w = np.cross(u, v)
        if w > 0:
            direction = 'left'
        elif w<0:
            direction = 'right'
        else:
            direction = 'same'

        return angle, direction

    

    def diff(self,val):
        differences = val[1:] - val[:-1]
        return differences
    
    def get_ego_traj_road(self, lane, v0, a, ind, dt, size): #直接按照索引取点，导致规划路径过长。不符合预测
    
        #此处的ind是在自车车道上索引
        # refer_tree = KDTree(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'])
        
        road_data = self.road_info_dict_pingjie[str(lane)] 
        center_vertices = road_data['center_vertices']
        phi_road = road_data['phi_road']

       

        if lane != self.ego_info_dict['ego']['lane']:
            refer_tree = KDTree(center_vertices)
            ego_station = np.array([self.ego_info_dict['ego']['x'],self.ego_info_dict['ego']['y']])
            distance,ind_traget = refer_tree.query(ego_station)
            # print('target lane dis ind : ',distance, ind_traget)
            ind = ind_traget
            



        max_len = len(phi_road)
        end_ind = ind + size

       

        # 限制在最大长度内
        valid_end_ind = min(end_ind, max_len)

        # 提取有效区段
        x_valid = center_vertices[ind:valid_end_ind, 0]
        y_valid = center_vertices[ind:valid_end_ind, 1]
        phi_valid = phi_road[ind:valid_end_ind] % (2 * np.pi)  # ⬅ 标准化为 [0, 2π)

        # 获取当前长度
        current_len = valid_end_ind - ind
        pad_len = size - current_len

        if len(x_valid) == 0:
            print("路点获取错误")
            print(center_vertices[:,0],center_vertices[:,1])
            print(valid_end_ind)
            print("自车",ind)

        if pad_len > 0:
            # 用最后一个值填充剩余
            x_pad = np.full(pad_len, x_valid[-1])
            y_pad = np.full(pad_len, y_valid[-1])
            phi_pad = np.full(pad_len, phi_valid[-1])

            x_list = np.concatenate([x_valid, x_pad])
            y_list = np.concatenate([y_valid, y_pad])
            phi_list = np.concatenate([phi_valid, phi_pad])
        else:
            x_list = x_valid
            y_list = y_valid
            phi_list = phi_valid

        # 计算速度序列
        indices = np.arange(size)
        v_list = v0 + a * dt * indices

        # print('road',x_list,y_list)

        # 拼接 4×size 矩阵
        # traj_matrix = np.vstack([x_list, y_list, phi_list, v_list])
        traj_matrix = np.column_stack([x_list, y_list, phi_list, v_list])


        return traj_matrix
    
    def get_ego_traj(self, lane, v0, a, ind, dt, size=41):

        # 计算路径长度
        T = 4  
        ego_v = self.ego_info_dict['ego']['v']
        ego_a = self.limit(self.ego_info_dict['ego']['a'], -3, 3)
        # traj_lenth = ego_v * T + 0.5 * ego_a * T ** 2  # 自车预测行驶的路径长度
        traj_lenth = ego_v * T  # 自车预测行驶的路径长度

        if traj_lenth < 6:
            traj_lenth = 6
        
        # print('规划轨迹长度： ',traj_lenth,ego_v,ego_a)
            


        
        # 获取道路数据
        road_data = self.road_info_dict_pingjie[str(lane)]
        center_vertices = road_data['center_vertices']  # 道路中心点坐标
        phi_road = road_data['phi_road']  # 车道角度

        #换道索引纠正
        if lane != self.ego_info_dict['ego']['lane']:
            refer_tree = KDTree(center_vertices)
            ego_station = np.array([self.ego_info_dict['ego']['x'],self.ego_info_dict['ego']['y']])
            distance,ind_traget = refer_tree.query(ego_station)
            # print('target lane dis ind : ',distance, ind_traget)
            ind = ind_traget

        # 计算每个点到起始点的路径长度
        total_length = 0
        lengths = [0]  # 起始点的路径长度为0
        for i in range(1, len(center_vertices)):
            dx = center_vertices[i, 0] - center_vertices[i - 1, 0]
            dy = center_vertices[i, 1] - center_vertices[i - 1, 1]
            segment_length = np.sqrt(dx**2 + dy**2)  # 计算每段路点之间的距离
            total_length += segment_length
            lengths.append(total_length)

        # 根据自车的预测路径长度来确定应该截取的路段
        valid_end_ind = np.searchsorted(lengths, traj_lenth)  # 截取到当前的路径长度



        x_0 = self.ego_info_dict['ego']['x']
        y_0 = self.ego_info_dict['ego']['y']
        phi_0 = self.ego_info_dict['ego']['yaw'] % (2 * np.pi)

        x_b = x_0 - self.ego_info_dict['ego']['length'] *0.5 * math.cos(phi_0)
        y_b = y_0 - self.ego_info_dict['ego']['length'] *0.5 * math.sin(phi_0)



        change_id = self.limit(valid_end_ind - 2,0,5)
        mid_ind = self.limit(ind+change_id,0,len(center_vertices)-1)


        x_mid = center_vertices[mid_ind,0]
        y_mid = center_vertices[mid_ind,1]
        phi_mid = phi_road[mid_ind] % (2 * np.pi)

        end_ind = self.limit(ind+valid_end_ind,0,len(center_vertices)-1)


        x_T = center_vertices[end_ind, 0]
        y_T = center_vertices[end_ind, 1]
        phi_T = phi_road[end_ind] % (2 * np.pi)

        # print("规划起点： ",x_0,y_0,phi_0)
        # print("规划中点： ",x_mid,y_mid,phi_mid)
        # print("规划终点： ",x_T,y_T,phi_T)
        # print("规划路径长度： ",traj_lenth)
        # print("自车索引： ",ind)
        # print("换道索引： ",mid_ind)
        # print("终点索引： ",end_ind)
       
        planner = TrajectoryPlanner(num_change=21, num_straight=21)

    
        #换道轨迹预测
        
        
        traj_matrix = planner.plan_by_command(
            'lane_change',
            start=(x_0,y_0,phi_0,ego_v), #从质心处规划
            # start=(x_b,y_b,phi_0,ego_v),  #从自车尾部处规划
            mid = (x_mid,y_mid,phi_mid,ego_v),
            end  =(x_T,y_T,phi_T,ego_v)
           
        )
        # print("规划轨迹",[traj_matrix[:,0],traj_matrix[:,1]])
        
        
        return traj_matrix


    def sign(self,para):
        if para > 0:
            return 1
        elif para < 0:
            return -1
        else:
            return 0

    def refer_cacul(self, x, y):
        x = np.array(x)
        y = np.array(y)
        dx = self.diff(x)
        dy = self.diff(y)

        dx_pre = deepcopy(dx)
        dx_after = deepcopy(dx)
        dx_pre = np.insert(dx_pre, 0, dx[0])
        dx_after = np.append(dx_after,dx[-1])
        dy_pre = deepcopy(dy)
        dy_after = deepcopy(dy)
        dy_pre = np.insert(dy_pre,0, dy[0])
        dy_after = np.append(dy_after, dy[-1])

        s_r = []
        s_r.append(0.0)
        dx_final = (dx_pre + dx_after) / 2.0
        dy_final = (dy_pre + dy_after) / 2.0
        ds = np.sqrt(dx_final**2 + dy_final**2)
        h_r = np.arctan2(dy_final, dx_final)
        for ii in range(len(dx_pre) - 1):
            s_r.append(s_r[ii] + math.sqrt(math.pow(dx[ii], 2) + math.pow(dy[ii], 2)))
        
        for ii in range(1,len(h_r)):
            if abs(h_r[ii] - h_r[ii - 1]) > math.pi:
                h_r[ii] = h_r[ii] - 2 * math.pi * self.sign(h_r[ii] - h_r[ii - 1])

        dh = self.diff(h_r)
        dh_p = deepcopy(dh)
        dh_a = deepcopy(dh)
        dh_p = np.insert(dh_p,0,dh[0])
        dh_a = np.append(dh_a, dh[-1])
        dh_f = (dh_p + dh_a) / 2.0
        k_r = np.sin(dh_f) / ds
        
        return h_r, k_r, s_r
    
    #检查两个数组中是否有相同元素
    def check_common_elements(self, arr1, arr2):
        set1 = set(arr1)
        set2 = set(arr2)
        common_elements = set1.intersection(set2)
        if common_elements:
            return True
        else:
            return False

    #检查两个数组中是否有包含关系
    def check_containment(self, arr1, arr2):
        set1 = set(arr1)
        set2 = set(arr2)
        if set1.issubset(set2) or set2.issubset(set1):
            return True
        else:
            return False
        
    def limit(self,para,low,up):
        if para >= up:
            return up
        elif para <= low:
            return low
        else:
            return para

    
