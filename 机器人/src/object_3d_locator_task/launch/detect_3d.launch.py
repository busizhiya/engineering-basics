# 一键启动三维定位功能
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, LogInfo, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # 获取yolo节点的功能包路径
    yolo_path = get_package_share_directory('yolov11_detect')
    locator_path = get_package_share_directory('object_3d_locator_task')
    target_class = LaunchConfiguration('target_class')
    # 第一步：启动深度相机与 YOLO 识别节点
    step1_log = LogInfo(msg="""
        1/5: 正在启动YOLO 识别节点...
        """)
    yolo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(yolo_path, 'launch/yolov11_detect.launch.py')
        ),
        # 传递YOLO识别检测的参数
        launch_arguments={
            'model': 'bset2',
            'classes': "['brick', 'grass', 'lucky', 'stone']",
            'conf': '0.7'
        }.items()
    )
    # 第二步：启动三维空间定位节点
    # 等待YOLO话题稳定发布后，再启动定位节点
    locator_launch = TimerAction(
        period=8.0,  # 延时8秒启动
        actions=[
            LogInfo(msg="2/5:  正在启动目标三维空间定位节点（Object 3D Locator）..."),
            Node(
                package='object_3d_locator_task',
                executable='object_3d_locator_node',
                name='object_3d_locator_node',
                parameters=[{'target_class': target_class}],
                output='screen'
            ),
            LogInfo(msg="定位节点已启动，正在等待YOLO识别目标 ...")
        ]
    )

    # arm_pick_place_launch = TimerAction(
    #     period=10.0,
    #     actions=[
    #         Node(
    #             package='arm_task',
    #             executable='auto_pick_place_node',
    #             name='auto_pick_place_node',
    #             output='screen',
    #             parameters=[
    #                 {'place_pose': [0.16, 0.0, 0.06, -90.0]},
    #                 {'lift_height': 0.08},
    #                 {'pick_pitch': -90.0},
    #                 {'min_pick_interval': 2.0},
    #             ],
    #         ),
    #     ],
    # )

    marker_viz_launch = TimerAction(
        period=10.5,
        actions=[
            LogInfo(msg='3/5: 启动目标坐标可视化节点（Marker Viz）...'),
            Node(
                package='object_3d_locator_task',
                executable='target_marker_viz_node',
                name='target_marker_viz_node',
                output='screen',
            ),
        ],
        condition=IfCondition(LaunchConfiguration('start_marker_viz')),
    )

    motor_teleop_launch = TimerAction(
        period=13.0,
        actions=[
            LogInfo(msg='可选: 启动底盘键盘控制节点（motor_control_node）...'),
            Node(
                package='motor_task',
                executable='motor_control_node',
                name='motor_control_node',
                output='screen',
            ),
        ],
        condition=IfCondition(LaunchConfiguration('start_motor_teleop')),
    )
    # 第三步：启动图像查看器 RQT
    image_view_launch = TimerAction(
        period=12.0,  # 延时12秒启动
        actions=[
            LogInfo(msg="4/5:  正在启动图像查看器（rqt_image_view）..."),
            Node(
                package='rqt_image_view',
                executable='rqt_image_view',
                name='rqt_image_view',
                output='screen'
            ),
            LogInfo(msg="""
        启动完毕！
        请在 rqt_image_view 中选择 /yolo_node/image_raw 查看识别画面。
        """)
        ]
    )

    rviz_launch = TimerAction(
        period=14.0,
        actions=[
            LogInfo(msg='5/5: 启动 RViz 可视化（TF/Marker/Point）...'),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', os.path.join(locator_path, 'rviz', 'object_3d_pick.rviz')],
            ),
        ],
        condition=IfCondition(LaunchConfiguration('start_rviz')),
    )

    # 返回 LaunchDescription，交由ROS 2 底层引擎执行节点
    return LaunchDescription([
        DeclareLaunchArgument('target_class', default_value='brick'),
        DeclareLaunchArgument('start_motor_teleop', default_value='false'),
        DeclareLaunchArgument('start_marker_viz', default_value='true'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        step1_log,
        yolo_launch,
        locator_launch,
        # arm_pick_place_launch,
        marker_viz_launch,
        motor_teleop_launch,
        image_view_launch,
        rviz_launch
    ])
