# 利用系统内置功能实现机械臂运动控制
# 导入需要的库
import rclpy
from rclpy.node import Node
import time
from kinematics_msgs.srv import SetRobotPose
from ros_robot_controller_msgs.msg import ServosPosition, ServoPosition


class UniversalPickPlace(Node):
    def __init__(self):
        super().__init__("universal_pick_place_node")
        # 初始化逆运动求解客户端，向set_pose_target发送请求
        self.ik_client = self.create_client(SetRobotPose, "/kinematics/set_pose_target")
        # 初始化总线舵机发布者，使用bus_servo/set_position话题控制手臂和爪子
        self.arm_pub = self.create_publisher(
            ServosPosition, "/ros_robot_controller/bus_servo/set_position", 10
        )
        # 等待运动学求解服务启动
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("正在等待运动学解算服务启动...")

    def get_input_data(self, prompt_text):
        # 引导用户在终端输入正确的数据
        while True:
            try:
                print(f"\n--- {prompt_text} ---")
                user_input = input("请输入 x y z pitch（例如 0.15 0 0.05 -90)：")
                data = list(map(float, user_input.split()))
                if len(data) != 4:
                    print("错误：必须输入4个数字！")
                    continue
                return data
            except ValueError:
                print("输入格式错误！请确保输入的是数字。")

    def move_to_target(self, x, y, z, pitch, duration=1.5):
        # 检查逆运动求解服务是否启动
        if not self.ik_client.service_is_ready():
            self.get_logger().error("逆运动求解服务未启动!")
            return False
        # 逆运动求解服务调用
        req = SetRobotPose.Request()
        req.position = [float(x), float(y), float(z)]  # 设置目标位置坐标
        req.pitch = float(pitch)  # 设置俯仰角（修正原截图笔误）
        req.pitch_range = [-180.0, 180.0]  # 设置俯仰角范围
        req.resolution = 1.0  # 设置求解俯仰角允许误差
        future = self.ik_client.call_async(req)  # 通过异步向逆运动求解服务发送请求
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.done():
            res = future.result()
            if res and res.success and len(res.pulse) > 0:
                # 将逆运动求解服务的返回结果打包成ServosPosition消息并发布
                msg = ServosPosition()
                msg.duration = duration
                msg.position = [
                    ServoPosition(id=i + 1, position=int(p))
                    for i, p in enumerate(res.pulse)
                ]
                self.arm_pub.publish(msg)
                time.sleep(duration + 0.2)  # 等待动作完成
                return True
            else:
                self.get_logger().error(f"坐标[{x}, {y}, {z}]无法到达")
                return False
        return False  # 服务调用未完成时返回失败

    # 控制爪子状态
    def set_gripper(self, state):
        msg = ServosPosition()
        msg.duration = 0.5
        pos_value = 150 if state == "open" else 350
        s = ServoPosition()
        s.id = 10
        s.position = pos_value
        msg.position.append(s)
        self.arm_pub.publish(msg)
        self.get_logger().info(f"夹爪控制: {state} ({pos_value})")
        time.sleep(0.8)

    # 设定连贯的执行动作
    def execute_task(self, pick_data, place_data, lift_height=0.08):
        px, py, pz, pp = pick_data
        lx, ly, lz, lp = place_data

        print("\n开始执行连贯动作序列")

        # 1. 初始张开爪子
        self.set_gripper("open")

        # 2. 移动到抓取点上方
        self.get_logger().info("准备抓取: 移动到目标上方...")
        if not self.move_to_target(px, py, pz + lift_height, pp):
            return

        # 3. 下降抓取
        self.get_logger().info("执行抓取: 下降并闭合爪子...")
        if not self.move_to_target(px, py, pz, pp, duration=1.0):
            return
        self.set_gripper("close")

        # 4. 垂直抬升
        self.get_logger().info("抬升物块...")
        if not self.move_to_target(px, py, pz + lift_height, pp, duration=1.0):
            return

        # 5. 高位水平平移到放置点上方
        self.get_logger().info("水平跨越：移动到放置位置上方...")
        if not self.move_to_target(lx, ly, lz + lift_height, lp):
            return

        # 6. 下降放置
        self.get_logger().info("执行放置：下降并张开爪子...")
        if not self.move_to_target(lx, ly, lz, lp, duration=1.0):
            return
        self.set_gripper("open")

        # 7. 抬起撤离
        self.get_logger().info("安全撤离...")
        self.move_to_target(lx, ly, lz + lift_height, lp, duration=1.0)
        print("\n搬运任务完成!")


def main():
    rclpy.init()
    node = UniversalPickPlace()
    print("\n机械臂搬运任务程序启动")
    try:
        while True:
            pick_data = node.get_input_data("设置【抓取】目标坐标")
            place_data = node.get_input_data("设置【放置】目标坐标")
            # 执行连贯序列，默认抬升8厘米
            node.execute_task(pick_data, place_data, lift_height=0.08)
            if input("\n是否继续下一次任务？(y/n): ").lower() != "y":
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
