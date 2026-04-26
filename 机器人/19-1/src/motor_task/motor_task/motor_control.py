#!/usr/bin/env python3
import rclpy
import time
import math
import sys
import termios
import tty
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from ros_robot_controller_msgs.msg import MotorState, MotorsState

class MecanumChassis:
    def __init__(self, wheelbase=0.1368, track_width=0.1410, wheel_diameter=0.065):
        self.wheelbase = wheelbase
        self.track_width = track_width
        self.wheel_diameter = wheel_diameter

    def speed_convert(self, speed):
        return speed / (math.pi * self.wheel_diameter)
    
    def set_velocity(self, linear_x, linear_y, angular_z=0.0):
        motor1 = linear_x - linear_y - angular_z * (self.wheelbase + self.track_width) / 2
        motor2 = linear_x + linear_y - angular_z * (self.wheelbase + self.track_width) / 2
        motor3 = linear_x + linear_y + angular_z * (self.wheelbase + self.track_width) / 2
        motor4 = linear_x - linear_y + angular_z * (self.wheelbase + self.track_width) / 2
        v_s = [self.speed_convert(v) for v in [-motor1, -motor2, motor3, motor4]]
        data = []
        for i in range(len(v_s)):
            msg = MotorState()
            msg.id = i + 1
            msg.rps = float(v_s[i])
            data.append(msg)

        msg = MotorsState()
        msg.data = data
        return msg

class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop_node')
        self.pub = self.create_publisher(MotorsState, '/ros_robot_controller/set_motor', 10)
        self.chassis = MecanumChassis()
        
        # 速度参数
        self.linear_speed = 0.2  # 线速度 (m/s)
        self.angular_speed = 1.0  # 角速度 (rad/s)
        
        # 当前速度
        self.current_linear_x = 0.0
        self.current_linear_y = 0.0
        self.current_angular_z = 0.0
        
        # 保存终端设置
        self.old_settings = termios.tcgetattr(sys.stdin)
        
        # 创建定时器，50ms发布一次速度
        self.timer = self.create_timer(0.05, self.publish_velocity)
        
        # 按键说明
        self.print_instructions()
        
        # 开始键盘监听
        self.get_logger().info("开始键盘控制，按 'q' 退出")
        self.keyboard_loop()
    
    def print_instructions(self):
        print("\n" + "="*50)
        print("        麦轮机器人键盘遥操作控制")
        print("="*50)
        print("移动控制:")
        print("  w: 前进")
        print("  s: 后退")
        print("  a: 左移")
        print("  d: 右移")
        print("  q: 左转")
        print("  e: 右转")
        print("  x: 停止所有电机")
        print("\n速度控制:")
        print("  [+]: 增加线速度 (当前: {:.2f} m/s)".format(self.linear_speed))
        print("  [-]: 减小线速度")
        print("  [*]: 增加角速度 (当前: {:.2f} rad/s)".format(self.angular_speed))
        print("  [/]: 减小角速度")
        print("\n其他:")
        print("  [space]: 急停")
        print("  [Esc] 或 [Ctrl+C]: 退出程序")
        print("="*50)
        print("注意: 按键需要按一次生效，持续按可连续移动")
        print("="*50 + "\n")
    
    def get_key(self):
        """获取单个按键，不等待回车"""
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        return ch
    
    def keyboard_loop(self):
        """键盘监听主循环"""
        try:
            while rclpy.ok():
                # 获取按键
                key = self.get_key()
                
                # 处理退出
                if key == '\x1b':  # ESC键
                    break
                
                # 处理按键
                self.handle_key(key)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.emergency_stop()
            self.cleanup()
    
    def handle_key(self, key):
        """处理按键输入"""
        if key.lower() == 'q' and key != 'Q':  # 小写q是左转
            self.current_angular_z = self.angular_speed
            self.get_logger().info("左转: angular_z = {:.2f} rad/s".format(self.angular_speed))
        elif key.lower() == 'e':  # 右转
            self.current_angular_z = -self.angular_speed
            self.get_logger().info("右转: angular_z = {:.2f} rad/s".format(-self.angular_speed))
        elif key.lower() == 'w':  # 前进
            self.current_linear_x = self.linear_speed
            self.get_logger().info("前进: linear_x = {:.2f} m/s".format(self.linear_speed))
        elif key.lower() == 's':  # 后退
            self.current_linear_x = -self.linear_speed
            self.get_logger().info("后退: linear_x = {:.2f} m/s".format(-self.linear_speed))
        elif key.lower() == 'a':  # 左移
            self.current_linear_y = self.linear_speed
            self.get_logger().info("左移: linear_y = {:.2f} m/s".format(self.linear_speed))
        elif key.lower() == 'd':  # 右移
            self.current_linear_y = -self.linear_speed
            self.get_logger().info("右移: linear_y = {:.2f} m/s".format(-self.linear_speed))
        elif key.lower() == 'x':  # 停止
            self.current_linear_x = 0.0
            self.current_linear_y = 0.0
            self.current_angular_z = 0.0
            self.get_logger().info("停止所有电机")
        elif key == ' ':  # 空格键急停
            self.emergency_stop()
            self.get_logger().warn("急停激活!")
        elif key == '+':  # 增加线速度
            self.linear_speed = min(0.5, self.linear_speed + 0.05)
            self.get_logger().info("线速度增加: {:.2f} m/s".format(self.linear_speed))
        elif key == '-':  # 减小线速度
            self.linear_speed = max(0.05, self.linear_speed - 0.05)
            self.get_logger().info("线速度减小: {:.2f} m/s".format(self.linear_speed))
        elif key == '*':  # 增加角速度
            self.angular_speed = min(2.0, self.angular_speed + 0.1)
            self.get_logger().info("角速度增加: {:.2f} rad/s".format(self.angular_speed))
        elif key == '/':  # 减小角速度
            self.angular_speed = max(0.1, self.angular_speed - 0.1)
            self.get_logger().info("角速度减小: {:.2f} rad/s".format(self.angular_speed))
        elif key.lower() == 'h':  # 显示帮助
            self.print_instructions()
    
    def publish_velocity(self):
        """定时发布速度命令"""
        msg = self.chassis.set_velocity(
            self.current_linear_x,
            self.current_linear_y,
            self.current_angular_z
        )
        self.pub.publish(msg)
    
    def emergency_stop(self):
        """急停功能"""
        self.current_linear_x = 0.0
        self.current_linear_y = 0.0
        self.current_angular_z = 0.0
        
        # 立即发布停止命令
        msg = self.chassis.set_velocity(0.0, 0.0, 0.0)
        self.pub.publish(msg)
        time.sleep(0.1)  # 确保停止命令发送
        self.get_logger().warn("急停: 所有电机已停止")
    
    def cleanup(self):
        """清理资源"""
        # 恢复终端设置
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        
        # 发布停止命令
        msg = self.chassis.set_velocity(0.0, 0.0, 0.0)
        self.pub.publish(msg)
        time.sleep(0.1)
        self.get_logger().info("清理完成，安全退出")

def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = KeyboardTeleop()
    
    try:
        # 由于键盘循环是阻塞的，我们不需要spin
        # 只需等待节点关闭
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("接收到Ctrl+C信号")
    finally:
        node.emergency_stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

