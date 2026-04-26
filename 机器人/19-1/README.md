# 机器人

## 任务目标

小车自主运动到固定位置的四个不同方块前，通过腕部深度摄像头识别方块类别，抓取后放置在固定区域，再回到起点。

厂商已提供逆运动学解算接口，需要手写的代码有：

- 目标检测（YOLO）
- 深度相机三维定位（像素坐标 -> 相机坐标 -> 机器人基座坐标）
- 麦轮控制
- 机械臂抓放

## 文件

- [src](src)：ROS 2 功能包目录
  - [src/yolov11_detect](src/yolov11_detect)：YOLO 推理节点与 launch 文件
  - [src/object_3d_locator_task](src/object_3d_locator_task)：目标三维定位与可视化
  - [src/motor_task](src/motor_task)：麦轮控制相关节点
  - [src/arm_task](src/arm_task)：机械臂抓取相关节点
- [yolo](yolo)：训练数据、训练脚本、推理测试脚本

## 检测类别

每组拿到的小方块类别不同。类别定义来自 [yolo/classes.txt](yolo/classes.txt#L1)：

- brick
- grass
- lucky
- stone

同样可在 [yolo/notes.json](yolo/notes.json#L1) 中看到类别 id 映射。

## YOLO 训练与导出

训练脚本在 [yolo/train_yolo.py](yolo/train_yolo.py#L1)，流程是：

1. 扫描 [yolo/images](yolo/images) 与 [yolo/labels](yolo/labels)
2. 自动生成 train/val 划分文件和 data.yaml
3. 使用 yolov11n 训练
4. 导出 OpenVINO 模型

运行示例：

```bash
cd yolo
python train_yolo.py
```

随机抽样推理脚本在 [yolo/demo_random_test.py](yolo/demo_random_test.py#L1)：

```bash
cd yolo
python demo_random_test.py --num 8
```

## ROS 2 一键启动

一键启动入口在 [src/object_3d_locator_task/launch/detect_3d.launch.py](src/object_3d_locator_task/launch/detect_3d.launch.py#L1)。

包含：

1. YOLO 检测（含深度相机 launch）
2. 三维定位节点
3. 目标 Marker 可视化（可选）
4. 底盘键盘控制（可选）
5. rqt_image_view
6. RViz（可选）

示例：

```bash
ros2 launch object_3d_locator_task detect_3d.launch.py \
  target_class:=stone \
  start_marker_viz:=false \
  start_motor_teleop:=false \
  start_rviz:=false
```

## 节点与话题

### YOLO 节点

代码见 [src/yolov11_detect/yolov11_detect/yolov11_node.py](src/yolov11_detect/yolov11_detect/yolov11_node.py#L82)。

- 输入图像：/ascamera/camera_publisher/rgb0/image
- 检测输出：/yolo_node/object_detect
- 可视化图像：/yolo_node/object_image

### 三维定位节点

代码见 [src/object_3d_locator_task/object_3d_locator_task/object_3d_locator.py](src/object_3d_locator_task/object_3d_locator_task/object_3d_locator.py#L1)。

- 订阅相机内参：/ascamera/camera_publisher/rgb0/camera_info
- 订阅深度图：/ascamera/camera_publisher/depth0/image_raw
- 订阅检测框：/yolo_node/object_detect
- 发布目标点：/object_3d_locator/target_point

### Marker 可视化节点

代码见 [src/object_3d_locator_task/object_3d_locator_task/target_marker_viz.py](src/object_3d_locator_task/object_3d_locator_task/target_marker_viz.py#L1)。

- 订阅：/object_3d_locator/target_point
- 发布：/object_3d_locator/markers

### 底盘控制节点（motor_task）

代码见 [src/motor_task/motor_task/motor_control.py](src/motor_task/motor_task/motor_control.py#L1)。

- 节点名：keyboard_teleop_node
- 发布电机命令：/ros_robot_controller/set_motor

### 机械臂抓放节点（arm_task）

代码见：
- [src/arm_task/arm_task/pick_place.py](src/arm_task/arm_task/pick_place.py#L1)
- [src/arm_task/arm_task/auto_pick_place.py](src/arm_task/arm_task/auto_pick_place.py#L1)
- [src/arm_task/arm_task/align_pick_place.py](src/arm_task/arm_task/align_pick_place.py#L1)

入口可执行名（见 [src/arm_task/setup.py](src/arm_task/setup.py#L1)）：

- pick_place_node：终端输入抓取/放置坐标后执行搬运
- auto_pick_place_node：订阅 `/object_3d_locator/target_point` 自动执行抓放
- align_pick_place_node：先底盘闭环对齐，再执行固定抓放流程，是最终使用的方案

机械臂相关通信接口：

- 逆解服务：/kinematics/set_pose_target
- 舵机控制发布：/ros_robot_controller/bus_servo/set_position
