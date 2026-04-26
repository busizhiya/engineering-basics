import os
import ast
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    OpaqueFunction,
)


def launch_setup(context):
    compiled = os.environ.get("need_compile", "True")
    conf = float(LaunchConfiguration("conf").perform(context))
    model = LaunchConfiguration("model").perform(context)
    classes_raw = LaunchConfiguration("classes").perform(context)

    try:
        classes = ast.literal_eval(classes_raw)
    except (ValueError, SyntaxError):
        classes = []

    if not isinstance(classes, list):
        classes = []
    if compiled == "True":
        peripherals_package_path = get_package_share_directory("peripherals")
    else:
        peripherals_package_path = "/home/ubuntu/ros2_ws/src/peripherals"

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, "launch/depth_camera.launch.py")
        ),
    )

    yolov11_node = Node(
        package="yolov11_detect",
        executable="yolov11_node",
        output="screen",
        parameters=[
            {"classes": classes},
            {"model": model, "conf": conf, "start": True},
        ],
    )

    return [
        camera_launch,
        yolov11_node,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("model", default_value="garbage_classification"),
            DeclareLaunchArgument(
                "classes",
                default_value="['BananaPeel','BrokenBones','CigaretteEnd','DisposableChopsticks','Ketchup','Marker','OralLiquidBottle','PlasticBottle','Plate','StorageBattery','Toothbrush','Umbrella']",
            ),
            DeclareLaunchArgument("conf", default_value="0.65"),
            OpaqueFunction(function=launch_setup),
        ]
    )


if __name__ == "__main__":
    # 创建一个LaunchDescription对象(create a LaunchDescription object)
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
