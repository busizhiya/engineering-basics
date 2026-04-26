import time

import rclpy
from geometry_msgs.msg import PointStamped
from kinematics_msgs.srv import SetRobotPose
from rclpy.node import Node
from ros_robot_controller_msgs.msg import ServoPosition, ServosPosition


class AutoPickPlace(Node):
    def __init__(self):
        super().__init__("auto_pick_place_node")
        self.ik_client = self.create_client(SetRobotPose, "/kinematics/set_pose_target")
        self.arm_pub = self.create_publisher(
            ServosPosition, "/ros_robot_controller/bus_servo/set_position", 10
        )

        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("正在等待运动学解算服务启动...")

        self.declare_parameter("pick_pitch", 0.0)
        self.declare_parameter("lift_height", 0.08)
        self.declare_parameter("pick_z_offset", 0.0)
        self.declare_parameter("min_pick_interval", 2.0)
        self.declare_parameter("place_pose", [0.16, 0.0, 0.06, 0.0])

        self.pick_pitch = float(self.get_parameter("pick_pitch").value)
        self.lift_height = float(self.get_parameter("lift_height").value)
        self.pick_z_offset = float(self.get_parameter("pick_z_offset").value)
        self.min_pick_interval = float(self.get_parameter("min_pick_interval").value)

        place_pose = self.get_parameter("place_pose").value
        if not isinstance(place_pose, (list, tuple)) or len(place_pose) != 4:
            place_pose = [0.16, 0.0, 0.06, 0.0]
        self.place_pose = [float(v) for v in place_pose]

        self.latest_target = None
        self.busy = False
        self.last_pick_time = 0.0

        self.create_subscription(
            PointStamped,
            "/object_3d_locator/target_point",
            self.target_callback,
            10,
        )
        self.get_logger().info(
            "自动抓放节点已启动，等待 /object_3d_locator/target_point ..."
        )

    def target_callback(self, msg):
        self.latest_target = [
            float(msg.point.x),
            float(msg.point.y),
            float(msg.point.z) + self.pick_z_offset,
            self.pick_pitch,
        ]

    def try_pick(self):
        if self.busy or self.latest_target is None:
            return
        if time.time() - self.last_pick_time < self.min_pick_interval:
            return

        self.busy = True
        pick_data = self.latest_target
        self.latest_target = None

        try:
            self.get_logger().info(
                f"开始自动抓取: x={pick_data[0]:.3f}, y={pick_data[1]:.3f}, z={pick_data[2]:.3f}"
            )
            self.execute_task(pick_data, self.place_pose, self.lift_height)
            self.last_pick_time = time.time()
        except Exception as e:
            self.get_logger().error(f"自动抓放执行失败: {e}")
        finally:
            self.busy = False

    def move_to_target(self, x, y, z, pitch, duration=1.5):
        if not self.ik_client.service_is_ready():
            self.get_logger().error("逆运动求解服务未启动!")
            return False

        req = SetRobotPose.Request()
        req.position = [float(x), float(y), float(z)]
        req.pitch = float(pitch)
        req.pitch_range = [-180.0, 180.0]
        req.resolution = 1.0

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done():
            return False

        res = future.result()
        if not (res and res.success and len(res.pulse) > 0):
            self.get_logger().error(f"坐标[{x:.3f}, {y:.3f}, {z:.3f}]无法到达")
            return False

        msg = ServosPosition()
        msg.duration = float(duration)
        msg.position = [
            ServoPosition(id=i + 1, position=int(p)) for i, p in enumerate(res.pulse)
        ]
        self.arm_pub.publish(msg)
        time.sleep(duration + 0.2)
        return True

    def set_gripper(self, state):
        msg = ServosPosition()
        msg.duration = 0.5
        pos_value = 200 if state == "open" else 474

        s = ServoPosition()
        s.id = 10
        s.position = pos_value
        msg.position.append(s)

        self.arm_pub.publish(msg)
        time.sleep(0.8)

    def execute_task(self, pick_data, place_data, lift_height=0.08):
        px, py, pz, pp = pick_data
        lx, ly, lz, lp = place_data

        pz = 0.025
        px += 0.02

        self.set_gripper("open")

        if not self.move_to_target(px, py, pz + lift_height, pp):
            return
        if not self.move_to_target(px, py, pz, pp, duration=1.0):
            return
        self.set_gripper("close")

        if not self.move_to_target(px, py, pz + lift_height, pp, duration=1.0):
            return
        if not self.move_to_target(lx, ly, lz + lift_height, lp):
            return
        if not self.move_to_target(lx, ly, lz, lp, duration=1.0):
            return
        self.set_gripper("open")
        self.move_to_target(lx, ly, lz + lift_height, lp, duration=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = AutoPickPlace()
    try:
        while rclpy.ok():
            # Keep subscription callbacks responsive and trigger pick logic
            # in the main loop, similar to pick_place.py's blocking flow.
            rclpy.spin_once(node, timeout_sec=0.1)
            node.try_pick()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
