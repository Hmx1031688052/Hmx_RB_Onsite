# samples_epre_wutfsd 运行逻辑说明

本文档基于 `run.py` 及其直接调用链整理，重点说明从场景准备、全局规划、感知、决策规划到控制输出的整体流程。

## 入口

主入口是 `run.py`。程序启动后会：

1. 自动把工程根目录加入 `sys.path`，以便加载 `modules`、`chassis`、`main` 等外部模块。
2. 创建组播通道，绑定 `lidar`、`notify`、`vehiclecontrol`、`prepare`、`ins`、`camera`。
3. 初始化 `dsac_main = Main(use_epre_dsac=True)`，内部创建强化学习环境 `Env` 和后台策略 Worker。
4. 初始化 `model = Predictor()`，内部加载 PointPillars 检测模型、AB3DMOT 跟踪器、VTS Planner 以及可选 ROS2 规划桥。
5. 进入 `main()` 无限循环，按准备、开始、感知、规划、控制的顺序推进。

运行参数：

```bash
python samples_epre_wutfsd/run.py \
  --config_center 47.110.233.70:52009 \
  --field_id field-zd-test1-22-0331134113-874 \
  --net_interface lo
```

## 总体数据流

```text
ActorPrepare
  -> run.get_prepare()
  -> Main.change_map()
  -> Env.reset()
  -> Predictor.change_map()
  -> Predictor.set_destination()
  -> publish /global_plan_request
  -> ActorPrepareResult
  -> Notify START
  -> INS + lidar
  -> Predictor.infer()
  -> PointPillars/AB3DMOT perception
  -> Main.get_action()
  -> Env._get_features()
  -> RL Worker action
  -> Env.cal_control()
  -> local ControlCommand
  -> optional ROS /ctrl_info override
  -> vehiclecontrol channel
```

## 准备阶段

`run.get_prepare()` 监听 `prepare` 通道中的 `MT_ACTOR_PREPARE` 消息，并解析 `archive_info.brief_data`。

准备阶段做四件关键事情：

1. 读取场景信息：天气、地图文件、测试车辆初始状态、目标状态和 `role_id`。
2. 调用 `dsac_main.change_map(brief_data, weather)`，解析 OpenDRIVE 地图并重置强化学习环境。
3. 调用 `model.change_map(zjl_odv_file)` 和 `model.set_destination(x, y, theta)`，更新 VTS Planner 的地图和目标点。
4. 调用 `publish_global_plan_request_from_brief_data(brief_data)`，把起点、终点和地图名发给 ROS2 全局规划节点。

地图准备完成后，`run.py` 会等待第一帧有效 INS。只有地图准备完成并且自车位姿可用，才会发送 `ActorPrepareResult`。

## 全局规划

代码里有两条全局规划相关路径。

第一条是 ROS2 全局规划请求。`run.publish_global_plan_request_from_brief_data()` 会发布 `PoseArray` 到：

```text
/global_plan_request
```

消息格式：

- `header.frame_id` 是当前 `.xodr` 地图名。
- `poses[0]` 是测试车初始状态。
- `poses[1]` 是目标状态。
- `orientation_z` 会从角度转成四元数的 `z/w`。

第二条是 `Env.reset()` 内部的地图路线构建。`Main.change_map()` 会通过 `parse_opendrive()` 解析地图，然后把离散车道信息传入 `Env.reset()`。环境会：

- 找到起点和终点最近的车道。
- 基于车道首尾点拼接候选道路。
- 删除重复或包含关系的道路片段。
- 保留和起点/终点相关的道路集合。
- 对中心线降采样，并计算航向角 `phi_road`、曲率 `curvature` 和累计里程 `station`。
- 计算目标所在车道 `goal_lane` 和目标相对中心线的左右位置。

因此，ROS2 全局规划用于外部规划节点交互；`Env.reset()` 的路线集合则直接服务于本地特征构建、RL 决策和几何控制。

## 启动与主循环

`run.process_notify()` 处理 `notify` 通道。

- `NT_START_TEST`：设置 `start_test=True`，并把 `model.start` 置为 `1`。
- `NT_FINISH_TEST` 或 `NT_ABORT_TEST`：调用 `dsac_main.finish()` 结算，重置 `Predictor` 和环境中的缓存状态。
- 碰撞、超时、到达终点等事件会在 `Predictor.infer()` 中参与奖励和结束条件计算。

测试开始后，`run.main()` 每轮执行：

1. `get_vehicle_pose()`：读取 INS，更新 `Predictor.ego`。
2. `get_pointcloud_msg()`：读取点云并调用 `Predictor.infer()`。
3. 如果返回 `done_out=True`，发送 `NT_ABORT_TEST`。
4. 如果返回控制命令，调用 `send_control_cmd(acc, speed, steer)` 写入 `vehiclecontrol` 通道。
5. `get_vehicle_feedback()`：读取底盘反馈，用于下一轮规划。

## 感知

感知入口是 `Predictor.process_pointcloud_msg()`。

主要处理流程：

1. 读取 lidar 点云，过滤自车点云和有效范围。
2. 按前向/后向区域拆分点云；恶劣天气下会做角度旋转和扇区恢复补偿。
3. 用 mmdet3d 的 PointPillars 模型分别检测前后点云。
4. 合并检测框，并按类别置信度过滤车辆、行人等目标。
5. 根据配置选择直接构造检测结果，或通过 AB3DMOT 做多目标跟踪。
6. `process_pubrole()` 把检测框从车体系转换到世界坐标系，输出 `Box2d` 障碍物列表。

`Predictor.infer()` 还会对障碍物做短时 ID 关联和速度估计，维护 `obstacles_id_dict`、`pre_obstacles` 等缓存，减少帧间跳变。

## 决策与局部规划

`Predictor.infer()` 会把自车状态、障碍物、奖励和结束标志传给：

```python
Main.get_action(...)
```

`Main.get_action()` 的核心流程：

1. 把障碍物列表整理成固定 ID 字典。
2. 调用 `Env._get_features()`，根据自车、障碍物和地图生成策略输入。
3. 对 EPre-DSAC 模式，向后台 Worker 的 `input_queue` 放入 `['act', state, reward, done, ...]`。
4. 从 `output_queue` 获取策略动作。
5. 调用 `Env.cal_control(action, step)` 把动作转换成局部控制量。

当前连续动作由两部分组成：

- `target_speed`：目标速度。
- `lateral_offset`：相对目标车道中心线的横向偏移。

`Env.cal_control()` 会调用 `_cal_continuous_control()`，逻辑是：

1. 选取当前自车车道或上一次目标车道。
2. 根据自车速度计算预瞄距离。
3. 在车道中心线上找到目标点，并应用横向偏移。
4. 用 pure pursuit 类几何关系计算转角 `rot`。
5. 用纵向 PID 根据目标速度计算加速度 `acc`。
6. 如果安全风险标志触发，则强制制动。

返回值是 `[acc, rot]`。

## ROS2 规划桥

`Predictor` 内部创建 `RosPlanningBridge`。默认启用，除非设置：

```bash
E2E_ENABLE_ROS_PLANNING=0
```

启用后会：

- 发布自车状态到 `/global_info`。
- 发布障碍物到 `/obs_info_local`。
- 发布 RL 决策到 `/RL_ctrl_info`，其中包含目标速度和横向偏移。
- 订阅 `/ctrl_info`，接收外部规划/控制节点返回的转向、速度、制动等控制信息。

`Predictor.infer()` 会先生成本地 `ControlCommand`，然后检查 `RosPlanningBridge.get_control_command()`。如果 `/ctrl_info` 在超时时间内有有效消息，ROS 控制命令会覆盖本地几何控制命令。

## 当前 ROS2 启动指令

从 `samples_epre_wutfsd/run.py` 进入时，ROS2 侧需要两个常驻节点：

- `ros2_map` 负责全局路径。它监听 `run.py` 发布的 `/global_plan_request`，按请求里的 `.xodr` 地图、起点和终点规划路径，然后发布 `/global_plan`，同时响应 `/map_info_request` 并发布 `/map_info_res`。
- `ros2_sanjiang` 负责局部规划和控制。当前应启动 `planning_wl` 包里的 `AutoDrive_onsite`，它订阅 `/global_plan`、`/global_info`、`/RL_ctrl_info`、`/obs_info_local` 和 `/map_info_res`，最终发布 `/ctrl_info` 给 `run.py` 覆盖本地控制命令。

如果工作空间还没有编译，先编译一次：

```bash
cd /path/to/e2e_wutfsd_zw/ros2_sanjiang
source /opt/ros/humble/setup.bash
colcon build --symlink-install

cd /path/to/e2e_wutfsd_zw/ros2_map
source /opt/ros/humble/setup.bash
source ../ros2_sanjiang/install/setup.bash
colcon build --symlink-install
```

`ros2_map` 当前应该执行：

```bash
cd /path/to/e2e_wutfsd_zw/ros2_map
source /opt/ros/humble/setup.bash
source ../ros2_sanjiang/install/setup.bash
source install/setup.bash
ros2 run gloplan opendrive_planner
```

这里要启动的是 `opendrive_planner`。正常联调时不要再手动执行 `ros2 run gloplan send_plan_request`，因为 `run.py` 已经会在 prepare 阶段自动发布 `/global_plan_request`；`send_plan_request` 只适合脱离仿真单独调试全局规划。

`ros2_sanjiang` 当前应该执行：

```bash
cd /path/to/e2e_wutfsd_zw/ros2_sanjiang
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run planning_wl AutoDrive_onsite
```

注意需要在 `ros2_sanjiang` 工作空间根目录启动，因为 `AutoDrive_onsite` 会用相对路径读取 `src/planning_wl/conf/AutoDriveConf.csv`、`DataTransferConf.csv` 和 `brake_lut_speed_pressure.csv`。`rs_ilqr_1` 的 `launch_rs_ilqr_rviz.launch.py` 会启动 `RS_part_1`、`ILQR_part_1` 和 RViz，主要是另一套 RS/ILQR 可视化链路；当前 WUT-FSD 这条链路应优先使用 `planning_wl AutoDrive_onsite`。

`AutoDrive_onsite` 收到 `/global_plan` 后会发布一次 `/glo_path`，这是它内部实际加载并重采样后的全局路径。检查 `ros2_map` 的全局规划是否成功时，应优先看 `/global_plan`、`/planner_status` 和 `global_path.png`；`/glo_path` 更适合检查 `AutoDrive_onsite` 是否收到了并成功加载了 `/global_plan`。如果 `/global_plan` 有数据但 `/glo_path` 一直为空，优先检查 `AutoDrive_onsite` 是否打印了 `Loaded /global_plan and reset onsite planner`；如果没有，说明它没有收到 `/global_plan` 或启动顺序不对。

当前 `ros2_map` 按“每个 prepare/回合都重新规划”的方式工作：

- `reload_map_on_each_plan_request=true`：每个被接受的 `/global_plan_request` 都会重新解析 `.xodr` 并重建拓扑，再发布新的 `/global_plan`。
- `duplicate_request_skip_window_sec=1.0`：`run.py` 在同一个 prepare 中会连续发布 3 次相同请求；1 秒内相同请求只规划第 1 次，避免同一回合重复算 3 遍。
- 如果下一回合地图和起终点完全相同，只要距离上一次成功发布超过 1 秒，仍然会重新加载地图并重新发布 `/global_plan`，`AutoDrive_onsite` 会在收到新的 `/global_plan` 后执行 `reset_for_new_global_plan()`。

推荐启动顺序：

1. 启动 `ros2_map` 的 `opendrive_planner`，等待日志出现 `waiting for /global_plan_request frame_id` 或地图加载成功。
2. 启动 `ros2_sanjiang` 的 `AutoDrive_onsite`，等待它订阅 `/global_plan` 并输出等待全局路径或基础信息的日志。
3. 启动 `samples_epre_wutfsd/run.py`。收到 `ActorPrepare` 后，`run.py` 会发布 `/global_plan_request`，`ros2_map` 发布 `/global_plan`，`AutoDrive_onsite` 再根据 `/global_info`、障碍物和 RL 决策发布 `/ctrl_info`。

如果 `opendrive_planner` 打印：

```text
can not reach ...
ERROR: no reachable global path for segment 0
```

这通常不是 ROS2 topic 没通，而是起终点坐标已经匹配到地图 lane，但最近中心线匹配到的 lane 在 OpenDRIVE 拓扑里不可达。当前 `road_node.py` 已加候选 lane fallback：原始最近 lane 失败后，会结合起终点 yaw 在附近 driving lane 中重试。修改后需要重新编译并重新 source：

```bash
cd /path/to/e2e_wutfsd_zw/ros2_map
source /opt/ros/humble/setup.bash
source ../ros2_sanjiang/install/setup.bash
colcon build --symlink-install --packages-select gloplan
source install/setup.bash
ros2 run gloplan opendrive_planner
```

## 控制输出

本地控制命令生成位置在 `Predictor.infer()` 末尾。

处理步骤：

1. 从 `Env.cal_control()` 得到 `acc` 和 `rot`。
2. 如果 `agent_par['shushidu']` 启用，则限制纵向加加速度、横向加速度变化等舒适性指标。
3. 将 `rot` 转成方向盘角：

```python
steer = rot / 0.0085
steer = max(-100, min(100, steer))
```

4. 按速度限制进一步限制转角。
5. 构造 `ControlCommand(acc, speed, steer)`。
6. 若 ROS2 `/ctrl_info` 有新控制，则用 ROS 命令覆盖。
7. `run.send_control_cmd()` 把最终命令写到 `vehiclecontrol` 通道。

## 结束与重置

结束信号来源包括：

- `NT_FINISH_TEST`
- `NT_ABORT_TEST`
- 碰撞通知
- 到达终点
- 最大时间
- 驶出边界

结束时会调用 `Main.finish()`，同时重置 `Predictor` 中的障碍物缓存、时间缓存、轨迹状态、碰撞标志、超时标志和 ROS 控制缓存，为下一轮场景准备。

## 关键文件职责

| 文件 | 作用 |
| --- | --- |
| `run.py` | 程序入口，组播通道管理，prepare/start/finish 时序，调用感知规划并发送控制 |
| `predictor.py` | 点云感知、障碍物跟踪、VTS Planner 地图/目标更新、ROS2 规划桥、本地控制输出 |
| `dsac_main.py` | 强化学习主控，地图切换，动作请求，episode 结算 |
| `env.py` | OpenDRIVE 路线拼接，状态特征构建，奖励相关环境状态，动作到控制的局部规划 |
| `guikong.py` | PID、避障判断、多项式轨迹/控制辅助 |
| `PODAR.py` | 风险评估、轨迹预测、换道/直行轨迹生成 |
| `global_plan_visualizer.py` | 可选的全局路线可视化输出 |

## 常用调试开关

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `E2E_DEBUG_SYNC` | `0` | 打印主循环、INS、控制发送等同步日志 |
| `E2E_ENABLE_ROS_PLANNING` | `1` | 是否启用 ROS2 规划桥 |
| `E2E_ROS_CTRL_TIMEOUT` | `0.5` | `/ctrl_info` 控制命令有效超时时间 |
| `E2E_ROS_STEER_SCALE` | `1.0` | ROS 返回转向的缩放系数 |
| `E2E_RL_LATERAL_LIMIT` | `5.0` | 发布给 ROS 的横向偏移限幅 |
| `E2E_VIS_GLOBAL_PLAN` | `0` | 是否保存 `Env.reset()` 构建的全局路线可视化 |
| `E2E_VIS_GLOBAL_PLAN_FULL_MAP` | `0` | 可视化时是否显示完整地图 |
| `E2E_VIS_GLOBAL_PLAN_DIR` | 未设置 | 全局路线可视化输出目录 |
| `DETECT_DEVICE` | `cuda:0` | mmdet3d 检测模型运行设备 |

## RViz2 联调显示

`ros2_sanjiang/src/rviz/src/rviz2.cpp` 现在用于显示当前 run 链路里的自车、障碍物、地图信息和全局/局部规划。它订阅 `/global_info`、`/obs_info_local`、`/global_plan_request`、`/global_plan`、`/glo_path`、`/map_info_res` 等话题，并统一发布到：

```text
/env_viz_markers
```

启动方式：

```bash
cd ros2_sanjiang
colcon build --packages-select veh_interfaces rviz
source install/setup.bash
ros2 run rviz rviz --ros-args -p fixed_frame:=map
```

然后启动 `rviz2` GUI，把 `Global Options -> Fixed Frame` 设置为 `map`，添加 `/env_viz_markers` 的 `MarkerArray`。建议在 `run.py` 发布 `/global_plan_request` 之前启动这个节点，否则 `/global_plan` 或 `/glo_path` 这种一次性路径可能已经发过而错过。

当前 run 链路主要看 `/obs_info_local` 的橙色障碍物；旧经纬度链路才看 `/obs_info` 的绿色障碍物。完整说明见 `ros2_sanjiang/src/rviz/README.md`。
