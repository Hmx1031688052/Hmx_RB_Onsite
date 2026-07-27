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

---

# 强化学习算法、训练与测试配置

本工程当前包含三种连续动作强化学习配置：`DSAC`、`STT_DSAC` 和 `DSAC-FDPI`。三种配置共用相同的自动驾驶环境、奖励信号、动作定义和底层控制接口，主要区别在于状态编码方式以及是否引入安全可行性约束和对偶策略。

## 三种算法与代码对应关系

| 算法名称 | 状态编码 | 算法类 | 主要代码 | 使用场景 |
| --- | --- | --- | --- | --- |
| `DSAC` | 53 维基础状态，经轻量卷积集合编码器压缩 | `Agent` | `agent.py`、`model.py` | 基础分布式软演员评论家基线 |
| `STT_DSAC` | 自车与 6 辆周车的历史状态、道路参考线和车辆交互，经时空 Transformer 编码 | `Epre_dsac_agent` | `epre_dsac/Epre_dsac.py`、`epre_dsac/Epre_dsac_model.py` | 验证结构化时空交互表征的增益 |
| `DSAC-FDPI` | 与 `STT_DSAC` 相同的时空 Transformer 编码 | `EpreDSACFDPIAgent` | `epre_dsac/Epre_dsac_fdpi.py`、`epre_dsac/fdpi_sampling.py` | 在任务奖励优化基础上加入显式安全可行域和 dual 风险探索策略 |

本文档中的 `STT_DSAC` 指采用时空 Transformer（Spatio-Temporal Transformer）状态编码器的 DSAC。它与基础 `DSAC` 使用相同的分布式双 Q 学习和最大熵策略优化，但策略输入不再是简单压缩后的基础状态，而是结构化的多车历史与地图交互特征。

## 算法模式选择

当前工程通过 `run.py` 中的 `use_epre_dsac` 和 `epre_dsac/parameters.py` 中的 `fdpi_enabled` 共同选择算法。

### DSAC

在 `run.py` 中设置：

```python
use_epre_dsac = False
```

在 `epre_dsac/parameters.py` 中设置：

```python
agent_par["fdpi_enabled"] = False
agent_par["two_agent"] = False
```

此时 `Worker_agent.py` 和 `Worker_agent_train.py` 创建 `agent.py` 中的 `Agent`。

### STT_DSAC

在 `run.py` 中设置：

```python
use_epre_dsac = True
```

在 `epre_dsac/parameters.py` 中设置：

```python
agent_par["fdpi_enabled"] = False
agent_par["two_agent"] = False
```

此时创建 `epre_dsac/Epre_dsac.py` 中的 `Epre_dsac_agent`。

### DSAC-FDPI

在 `run.py` 中设置：

```python
use_epre_dsac = True
```

在 `epre_dsac/parameters.py` 中设置：

```python
agent_par["fdpi_enabled"] = True
agent_par["fdpi_dual_enabled"] = True
agent_par["fdpi_full_policy_loss"] = True
agent_par["two_agent"] = False
```

此时创建 `epre_dsac/Epre_dsac_fdpi.py` 中的 `EpreDSACFDPIAgent`。

`two_agent` 表示路口策略和直行策略分开训练，不表示 FDPI 的主策略和 dual 策略。当前 FDPI 路径要求：

```python
agent_par["two_agent"] = False
```

## 训练与测试开关

训练模式需要保持下面两处一致：

```python
# dsac_main.py
self.train = True
```

```python
# epre_dsac/parameters.py
agent_par["train"] = True
```

测试模式设置为：

```python
# dsac_main.py
self.train = False
```

```python
# epre_dsac/parameters.py
agent_par["train"] = False
```

其中 `dsac_main.py` 中的 `self.train` 决定是否创建后台训练进程、是否向 replay buffer 写入 transition，以及策略动作采用随机采样还是确定性均值。`agent_par["train"]` 主要控制训练进程中的模型重置和保存逻辑。为避免训练进程与动作进程状态不一致，二者应同时修改。

## 公共环境与动作设置

三种算法共用二维连续动作：

```text
a[0]：target_speed，目标速度
a[1]：lateral_offset，相对参考车道中心线的横向偏移
```

动作范围由 `config.py` 和 `epre_dsac/parameters.py` 给出：

```python
action_low_limit = [0.0, -5.0]
action_high_limit = [8.0, 5.0]
```

策略网络输出高斯分布的均值和标准差，经过 `tanh` 和线性缩放映射到实际动作范围。训练时从分布中采样，测试时使用分布均值对应的确定性动作。

动作进入 `Env.cal_control()` 后：

1. 根据目标速度计算纵向 PID 控制量；
2. 根据横向偏移构造目标预瞄点；
3. 通过几何跟踪计算前轮转角；
4. 输出加速度和转向控制；
5. 若风险规则触发，可由底层安全逻辑覆盖为制动控制；
6. 若 ROS2 `/ctrl_info` 在有效期内有数据，则 ROS2 局部规划控制可以覆盖本地控制。

## 当前生效的公共训练超参数

当前 `Agent`、`Epre_dsac_agent` 和 `EpreDSACFDPIAgent` 的主要训练参数实际来自 `config.py`：

| 参数 | 当前值 | 说明 |
| --- | ---: | --- |
| `Config.lr` | `1e-5` | Q 网络、策略网络、安全网络的基础学习率 |
| `Config.discount_factor` | `0.995` | 奖励折扣因子 |
| `Config.reply_buffer_size` | `30000` | replay buffer 容量 |
| `Config.batch_size` | `64` | batch 大小 |
| `Config.tau` | `0.005` | target 网络 Polyak 更新系数 |
| `delay_update` | `2` | 策略和温度参数延迟更新周期 |
| `target_entropy` | `-2` | 二维动作空间的目标熵 |
| `weight_decay` | `1e-4` | Adam 优化器权重衰减 |
| `gradient clip` | `40` | FDPI 更新中的梯度范数裁剪 |

`Epre_dsac_agent` 和 `EpreDSACFDPIAgent` 创建时使用：

```python
hrl=True
h_rl=1e-5
```

时空编码器 `HNet` 使用独立学习率，并配置：

```python
StepLR(step_size=50000, gamma=0.8)
```

当前训练调用链没有直接使用 `epre_dsac/parameters.py` 中的 `learning_rate`、`memory_size` 和 `batch_size` 作为主训练参数。调整学习率、buffer 容量和 batch 大小时，应优先修改 `config.py`，避免只修改 `parameters.py` 后训练参数没有实际变化。

## DSAC 算法设置

### 状态输入

基础 DSAC 使用 `Env` 输出的 53 维状态。`agent.py` 将状态拆分为：

```text
前 11 维：自车、目标和道路相关基础状态
后 42 维：周车集合状态
```

后 42 维由 `model.py` 中的轻量一维卷积 `HNet` 编码为 32 维，再与前 11 维拼接，得到 43 维 Q 网络和策略网络输入。

### 分布式双 Q 网络

`QNet` 不仅输出 Q 均值，还输出非负标准差：

```text
Q(s,a) -> [mean, std]
```

训练时从价值分布采样，并使用双 Q 最小值降低高估偏差。目标值包括最大熵项：

```text
y = r + gamma * (min(Q1_target, Q2_target) - alpha * log pi(a'|s'))
```

价值分布采样偏差采用约 3 倍标准差范围裁剪，减少极端随机样本造成的更新不稳定。

### 策略与温度参数

策略目标为：

```text
L_actor = E[alpha * log pi(a|s) - min(Q1(s,a), Q2(s,a))]
```

温度参数 `alpha` 自动学习，使策略熵接近二维动作空间的目标熵 `-2`。

## STT_DSAC 算法设置

STT_DSAC 保留 DSAC 的双分布式 Q、最大熵策略和自动温度更新，仅替换状态编码部分。

### 结构化时空输入

环境为自车和最多 6 辆周车构造：

```text
env_input shape = [7, 11, 11]
```

含义为：

```text
7：1 辆自车 + 6 辆周车
11：连续 11 个历史时刻
11：8 个车辆运动/几何特征 + 3 个 Frenet/全局路径特征
```

同时为每个交通参与者构造地图参考线：

```text
env_map shape = [7, 3, 51, 4]
```

含义为：

```text
7：自车与 6 辆周车
3：候选车道或参考线
51：每条参考线的离散路点数
4：路点位置、航向或有效性相关特征
```

### HNet 时空交互编码

`epre_dsac/Epre_dsac_model.py` 中的 `HNet` 依次执行：

1. `AgentEncoder`：对每辆车的 11 步历史进行位置编码和时序自注意力；
2. `MapEncoder`：编码对应车辆的未来道路参考线；
3. `Agent2Map`：通过交叉注意力融合车辆行为与地图；
4. `Agent2Agent`：通过两层自注意力描述自车和周车之间的交互；
5. 将 7 个交通参与者的 128 维表示展开为 `7 × 128 = 896` 维策略状态。

当前 `use_h()` 和 `use_target_h()` 直接使用 `HNet(env_input, env_map)` 的输出，原始状态向量主要用于 replay 对齐和环境信息传递，不再与 Transformer 输出拼接。

### 与 DSAC 的公平对比

进行 DSAC 与 STT_DSAC 对比时，建议保持以下设置完全一致：

- 相同训练场景和场景出现顺序；
- 相同 reward；
- 相同动作范围；
- 相同训练步数；
- 相同 batch size、buffer 容量、折扣因子和 target 更新系数；
- 相同随机种子和测试场景；
- 唯一主要变量为状态编码器。

## DSAC-FDPI 算法设置

DSAC-FDPI 在 STT_DSAC 的共享时空编码器和奖励 DSAC 主干上增加安全可行性学习以及训练阶段的 dual 策略。

### 网络组成

奖励优化部分：

```text
main_policy
q1 / q2
q1_target / q2_target
```

主策略安全部分：

```text
g1 / g2：未来进入安全违反状态的可行性价值
gr1 / gr2：已经违反后恢复到安全状态的恢复价值
```

对偶策略部分：

```text
dual_policy：主动搜索约束边界和风险样本
dual_g1 / dual_g2：dual 策略对应的风险可达价值
```

所有安全 critic 均有 target 网络。

### reward 与 cost 分离

FDPI 使用独立的任务奖励 `reward` 和安全代价 `cost`：

```text
reward：用于优化通行效率、任务完成、舒适性等目标
cost：用于表示是否触发碰撞、近碰撞或 TTC 风险约束
```

`predictor.py` 中已经提供：

```python
compute_safety_cost(
    collision_done,
    near_collision=False,
    min_ttc=None,
    ttc_threshold=1.5,
)
```

当前实际调用为：

```python
cost = compute_safety_cost(collision_done)
```

因此当前训练中的 `cost` 实际只由碰撞产生。函数已经预留近碰撞和 TTC 接口，但只有在调用处传入 `near_collision` 或 `min_ttc` 后，这两类风险才会进入 FDPI 安全学习。

建议先用二值 cost 保证训练稳定：

```text
无违反：cost = 0
发生安全违反：cost = 1
```

不要将碰撞奖励惩罚值直接作为 cost。

### 主策略区域化更新

当 `fdpi_full_policy_loss=True` 时，主策略根据 `g` 和当前 cost 被划分为四类：

1. 可行内部区域：主要最大化任务 Q；
2. 可行临界区域：同时优化 Q 和安全可行性；
3. 不可行区域：提高安全约束权重；
4. 已违反区域：优先最大化恢复价值 `gr`。

安全阈值由：

```python
agent_par["fdpi_pf"] = 0.10
```

控制。

### dual 策略目标

`dual_policy` 不直接追求高任务奖励。它最大化 `dual_g` 所预测的风险，同时通过双向 KL 约束保持在主策略附近：

```text
最小化：-dual_g
        + lambda3 * KL(dual || main)
        + lambda4 * KL(main || dual)
```

因此 dual 策略的作用是为安全 critic 提供接近当前主策略、但更靠近安全边界的训练样本。测试和部署阶段不能使用 dual 策略。

### 单环境 episode 级交替采样

当前仿真系统无法同时运行两套独立环境，因此 FDPI 采用单环境、按 episode 固定行为策略的方式采样：

```text
Episode 1：main policy
Episode 2：main policy
Episode 3：dual policy
Episode 4：main policy
...
```

同一个 episode 中不会逐步切换行为策略。这样每条 transition 的行为分布明确，可以正确计算主策略和 dual 策略的双向重要性采样权重。

### dual 激活条件

当前配置：

| 参数 | 值 | 说明 |
| --- | ---: | --- |
| `fdpi_warmup_steps` | `10000` | 训练更新不足该步数时只使用主策略 |
| `fdpi_dual_threshold` | `0.90` | 平均可行率高于该值后允许启用 dual |
| `fdpi_feasible_window` | `1000` | 可行率滑动窗口 |
| `fdpi_dual_sample_ratio` | `0.50` | dual 激活后，每个新 episode 选择 dual 的概率 |

激活逻辑为：

```text
global_step >= 10000
且最近窗口 mean_feasible_ratio > 0.90
且 fdpi_dual_enabled = True
```

测试模式下无论传入何种行为策略，`take_fdpi_action(..., train=False)` 都会强制选择主策略。

### 重要性采样

当 transition 来自主策略时：

```text
log_is_to_main = 0
log_is_to_dual = log pi_dual(a|s) - log pi_main(a|s)
```

当 transition 来自 dual 策略时：

```text
log_is_to_main = log pi_main(a|s) - log pi_dual(a|s)
log_is_to_dual = 0
```

当前配置为：

| 参数 | 值 |
| --- | ---: |
| `fdpi_beta` | `0.50` |
| `fdpi_min_is_weight` | `0.10` |
| `fdpi_max_is_weight` | `10.0` |

权重在 log 空间累计并裁剪，避免主策略和 dual 策略差异过大时出现梯度爆炸。

### 其他 FDPI 参数

| 参数 | 当前值 | 说明 |
| --- | ---: | --- |
| `fdpi_cost_gamma` | `0.97` | 安全可行性价值折扣因子 |
| `fdpi_target_kl` | `5.0` | dual 与 main 的目标 KL 距离 |
| `fdpi_cg_init` | `0.01` | 临界可行域宽度初值 |
| `fdpi_lambda_lr` | `3e-4` | 约束乘子的学习率 |
| `fdpi_full_policy_loss` | `True` | 是否启用完整 FDPI 区域化主策略损失 |

## 强化学习训练流程

### 进程结构

训练时包含两个主要后台进程：

```text
Worker_agent
  ├─ 接收状态并生成动作
  ├─ 维护当前 episode transition
  └─ episode 结束后发送数据和接收新策略参数

Worker_agent_train
  ├─ 维护全局 replay buffer
  ├─ 在空闲时间执行梯度更新
  ├─ 定期保存模型和 buffer
  └─ 将最新 HNet、main policy 和可选 dual policy 返回动作进程
```

Q 网络和安全 critic 只保留在训练进程中；动作进程主要同步状态编码器和策略网络，减少进程间传输开销。

### transition 生成

每个有效 RL 决策步保存：

```text
state
action
reward
next_state
done 或 terminated/truncated
```

STT_DSAC 还保存：

```text
env_input
next_env_input
env_map
next_env_map
```

DSAC-FDPI 额外保存：

```text
cost
behavior_policy
logp_main
logp_dual
log_is_to_main
log_is_to_dual
terminated
truncated
```

`terminated` 用于表示碰撞、到达终点等真实终止；`truncated` 用于表示超时。FDPI 的价值 bootstrap 仅由 `terminated` 截断，单纯超时仍允许估计下一状态价值。

### replay buffer 与更新时机

当前 replay buffer 容量为 30000，batch size 为 64。当 buffer 中有效 transition 数量达到约 70 条以上后，后台训练进程开始更新。

每次 episode 结束并完成模型同步后，`train_time` 重新置零。训练进程在下一次 episode 数据到达前的空闲时间内，最多执行约 501 次 `update_model()` 调用。实际更新数量取决于：

- 当前 episode 持续时间；
- CPU/GPU 训练速度；
- replay buffer 是否达到 batch 要求；
- 动作进程与训练进程的同步等待时间。

因此论文实验中建议同时记录环境交互步数和参数更新步数，避免只按 episode 数量比较三种算法。

### episode 数据保留差异

当前 `Worker_agent_train.send_model()` 中：

- FDPI 会保留所有 `real=True` 的真实 episode 数据；
- 非 FDPI 的旧路径会过滤 `done_type == 'termination'` 或 `done_type == 'out'` 的 episode。

若要进行严格公平的 DSAC、STT_DSAC 和 DSAC-FDPI 对比，建议统一三种算法的 episode 数据保留规则。否则 FDPI 可能看到更多碰撞或越界样本，而基础算法丢弃了这些数据，导致比较不仅包含算法差异，还包含训练数据差异。

### 模型保存

训练更新次数每达到 2000 的整数倍时保存一次模型。输出目录按启动时间建立：

```text
logs/YYYY-MM-DD_HHMM/
  ├─ model/
  ├─ model_temp/
  ├─ logagent/
  └─ logtrain/
```

DSAC 和 STT_DSAC 主要保存：

```text
q1_local
q2_local
policy_local
h_local
对应 target 网络
optimizer 状态
log_alpha
replay buffer
```

DSAC-FDPI 除上述文件外，还保存：

```text
*_fdpi_checkpoint.pth
```

该完整 checkpoint 包含：

- 主策略、dual 策略；
- Q、g、gr、dual_g 及所有 target 网络；
- alpha、cg、lambda1 至 lambda4；
- 所有 optimizer；
- `global_step`、`episode`；
- `feasible_ratio_history`。

恢复 FDPI 训练时应优先使用完整 `_fdpi_checkpoint.pth`。仅加载传统四文件模型时，代码会从主策略初始化 dual 策略，但安全 critic 仍为随机参数，不等价于完整续训。

## 继续训练

在 `epre_dsac/parameters.py` 中设置：

```python
agent_par["continue_train"] = True
agent_par["train_data"]["folder_path"] = "已有 model_temp 目录"
agent_par["train_data"]["episode"] = 已完成的episode数
agent_par["train_data"]["update_time"] = 已完成的更新次数
agent_par["train_data"]["global_step"] = 已完成的FDPI更新步数
```

DSAC-FDPI 会优先搜索目录中最新的：

```text
*_fdpi_checkpoint.pth
```

如果没有完整 FDPI checkpoint，则退化为加载旧 DSAC/STT_DSAC 权重，并重新初始化安全网络。

## 强化学习测试流程

### 确定性动作

测试时必须设置：

```python
self.train = False
agent_par["train"] = False
```

此时 `TanhGaussDistribution.sample(train=False)` 直接使用高斯均值，不再加入随机采样噪声。

对于 DSAC-FDPI：

```text
测试只使用 main_policy
不启用 dual episode
不计算用于采样的累计重要性权重
不写入 replay buffer
不执行参数更新
```

### 模型加载

当前 `run.py` 不会自动加载指定模型。基础 DSAC 和 STT_DSAC 可以在创建 `Main` 后调用：

```python
dsac_main.load_net(
    policy_path,
    q1_path,
    q2_path,
    h_path,
)
```

DSAC-FDPI 测试阶段只需要部署主策略，因此使用上述四文件接口加载 `main_policy`、Q 和 HNet 即可。此路径不会恢复完整 dual 和安全网络，但不会影响主策略的确定性测试。

若需要继续训练或分析 FDPI 的安全 critic、可行率和 dual 策略，则必须使用：

```python
agent.load_checkpoint(fdpi_checkpoint_path)
```

或通过 `continue_train_model()` 加载完整 checkpoint。

### 推荐测试设置

三种算法应使用完全相同的测试协议：

1. 使用训练期间未参与参数更新的固定测试场景集；
2. 每种算法使用相同起点、终点、背景车数量和交通流随机种子；
3. 使用确定性策略，不在测试中继续学习；
4. 每种场景重复多次并报告均值和标准差；
5. 除任务 reward 外，单独记录安全、效率和舒适性指标；
6. DSAC-FDPI 只部署主策略，不能选择 dual 策略；
7. 测试控制周期、ROS2 节点、感知模型和安全覆盖规则保持一致。

建议至少统计：

| 指标 | 说明 |
| --- | --- |
| 成功率 | 到达目标且未发生碰撞的 episode 比例 |
| 碰撞率 | 发生碰撞的 episode 比例 |
| 超时率 | 未在规定时间内完成任务的比例 |
| 平均回报 | episode 累计 reward 均值 |
| 平均安全代价 | episode cost 累计值或违反率 |
| 最小 TTC | 有 TTC 数据时统计最危险交互程度 |
| 平均速度/通行时间 | 评价效率性 |
| 纵向加速度、横向加速度和 jerk | 评价舒适性 |
| 交规违反率 | 评价规则符合性 |

### 公平对比注意事项

为了使三种算法的结果具有可解释性，建议：

- 三种算法使用相同 reward 和相同安全规则；
- DSAC 与 STT_DSAC 不使用 FDPI cost 更新，但测试时仍统计相同 cost 指标；
- 不允许 FDPI 使用更多训练场景或更多环境交互步数；
- 统一终止 episode 是否写入 replay buffer；
- 统一模型选择规则，例如按照固定交互步数的 checkpoint，而不是分别选取各自最优测试模型；
- 至少使用 3 个随机种子，报告均值和标准差；
- 同时报告环境交互步数和梯度更新次数。

## 日志与 TensorBoard

动作进程日志写入：

```text
logs/<time>/logagent/
```

训练进程日志写入：

```text
logs/<time>/logtrain/
```

可以运行：

```bash
tensorboard --logdir samples_epre_wutfsd/logs
```

DSAC 和 STT_DSAC 主要日志包括：

```text
q_loss
policy_loss
new_log_prob
alpha
average_episode_reward
average_goal
average_collision
average_episode_speed
```

DSAC-FDPI 额外包括：

```text
train/q1_loss
train/q2_loss
train/main_policy_loss
train/g1_loss
train/g2_loss
train/gr1_loss
train/gr2_loss
train/dual_g1_loss
train/dual_g2_loss
train/dual_policy_loss
fdpi/feasible_ratio
fdpi/dual_active
fdpi/kl_dual_to_main
fdpi/kl_main_to_dual
fdpi/main_is_weight_mean
fdpi/dual_is_weight_mean
fdpi/main_dual_action_l2_gap
fdpi/cost_rate
```

训练初期 `dual_active=0` 是正常现象。只有达到 warmup 步数且平均可行率超过阈值后，dual 才会被激活。

## 最小测试

在工程根目录运行：

```bash
python -m unittest tests.test_fdpi_minimal -v
```

测试覆盖：

- tanh 高斯动作范围和 log probability；
- 主策略与 dual 策略输出形状；
- 测试时强制使用主策略；
- 重要性采样约定和权重裁剪；
- episode 级策略调度；
- FDPI replay buffer schema；
- 单 batch 更新是否产生有限 loss；
- 完整 FDPI checkpoint 保存和恢复；
- 关闭 FDPI 后旧 DSAC agent 是否仍可训练。

在 CPU 上运行完整 FDPI 单 batch 更新可能较慢，建议在实际训练机上设置：

```bash
RL_DEVICE=cuda:0 python -m unittest tests.test_fdpi_minimal -v
```


## 当前实现需要重点检查的事项

1. 基础 DSAC 的训练进程在 `continue_train=False` 时仍会把 `self.folder_path` 传给 `Agent`。正式运行基础 DSAC 前，应确认 `Worker_agent_train.py` 已为该变量设置默认值，避免未定义变量。
2. 当前 `run.py`、`dsac_main.py` 和 `parameters.py` 分别保存算法和训练开关，修改模式时必须保持一致。后续建议统一为一个 `algorithm_name` 和一个 `train_mode` 配置入口。
3. 当前 FDPI cost 只接入碰撞事件。若论文中声称使用近碰撞或 TTC 约束，需要在 `Predictor.infer()` 调用 `compute_safety_cost()` 时真正传入对应变量。
4. 三种算法当前对终止 episode 的 replay 保留规则不完全一致，正式对比实验前建议统一。
5. 测试前应确认加载的是对应算法和对应状态编码器的模型。基础 DSAC 的 HNet 与 STT_DSAC/FDPI 的 HNet 结构不同，不能互换 checkpoint。
