import math
import time

import rclpy
from geometry_msgs.msg import PointStamped
from kinematics_msgs.srv import SetRobotPose
from rclpy.node import Node
from ros_robot_controller_msgs.msg import MotorState, MotorsState, ServoPosition, ServosPosition


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
        for i, speed in enumerate(v_s):
            msg = MotorState()
            msg.id = i + 1
            msg.rps = float(speed)
            data.append(msg)

        out = MotorsState()
        out.data = data
        return out


class AlignPickPlaceController(Node):
    def __init__(self):
        super().__init__('align_pick_place_node')
        self.chassis = MecanumChassis()

        self.motor_pub = self.create_publisher(MotorsState, '/ros_robot_controller/set_motor', 10)
        self.arm_pub = self.create_publisher(ServosPosition, '/ros_robot_controller/bus_servo/set_position', 10)
        self.ik_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')

        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('waiting for /kinematics/set_pose_target ...')

        self.declare_parameter('align_target_x', 0.18)
        self.declare_parameter('align_x_tol', 0.015)
        self.declare_parameter('align_y_tol', 0.008)
        self.declare_parameter('align_x_kp', 0.8)
        self.declare_parameter('align_y_kp', 1.2)
        self.declare_parameter('align_max_vx', 0.10)
        self.declare_parameter('align_max_vy', 0.10)
        self.declare_parameter('align_stable_cycles', 10)
        self.declare_parameter('target_timeout_sec', 1.2)
        self.declare_parameter('target_lost_grace_sec', 1.0)
        self.declare_parameter('cycle_cooldown_sec', 2.0)

        self.declare_parameter('pick_pose_fixed', [0.20, 0.0, 0.03, -45.0])
        self.declare_parameter('place_pose_fixed', [0.16, 0.0, 0.04, 0.0])
        self.declare_parameter('lift_height', 0.08)
        self.declare_parameter('home_pose_fixed', [0.16, 0.0, 0.10, 0.0])

        # Due to camera blind spot near gripper, move base forward after alignment
        # before starting the fixed arm pick sequence.
        self.declare_parameter('pre_pick_move_vx', 0.12)
        self.declare_parameter('pre_pick_move_vy', -0.02)
        self.declare_parameter('pre_pick_move_sec', 0.8)

        self.declare_parameter('post_pick_move_vx', 0.10)
        self.declare_parameter('post_pick_move_vy', 0.0)
        self.declare_parameter('post_pick_move_sec', 1.8)

        # Mission path parameters (meters/degrees)
        self.declare_parameter('mission_forward1_m', 1.50)
        self.declare_parameter('mission_turn_right_deg', 117.0)
        self.declare_parameter('mission_forward2_m', 1.10)
        self.declare_parameter('mission_back_after_grasp_m', 0.20)
        self.declare_parameter('mission_linear_speed', 0.35)
        self.declare_parameter('mission_turn_speed_rad', 1.8)
        self.declare_parameter('turn_backlash_comp_deg', 3.0)
        self.declare_parameter('distance_scale', 0.95)
        self.declare_parameter('grasp_check_wait_sec', 3)
        self.declare_parameter('grasp_retry_max', 8)

        # Dead-reckoning initial pose and final place target in world frame.
        # x forward, y left. "origin right 1.35m" => (0, -1.35).
        self.declare_parameter('initial_x_m', 0.0)
        self.declare_parameter('initial_y_m', 0.0)
        self.declare_parameter('initial_yaw_deg', -45.0)
        self.declare_parameter('place_target_x_m', 0.0)
        self.declare_parameter('place_target_y_m', -1.35)

        self.declare_parameter('gripper_open_pos', 250)
        self.declare_parameter('gripper_close_pos', 500)

        self.align_target_x = float(self.get_parameter('align_target_x').value)
        self.align_x_tol = float(self.get_parameter('align_x_tol').value)
        self.align_y_tol = float(self.get_parameter('align_y_tol').value)
        self.align_x_kp = float(self.get_parameter('align_x_kp').value)
        self.align_y_kp = float(self.get_parameter('align_y_kp').value)
        self.align_max_vx = float(self.get_parameter('align_max_vx').value)
        self.align_max_vy = float(self.get_parameter('align_max_vy').value)
        self.align_stable_cycles = int(self.get_parameter('align_stable_cycles').value)
        self.target_timeout_sec = float(self.get_parameter('target_timeout_sec').value)
        self.target_lost_grace_sec = float(self.get_parameter('target_lost_grace_sec').value)
        self.cycle_cooldown_sec = float(self.get_parameter('cycle_cooldown_sec').value)

        self.pick_pose_fixed = [float(v) for v in self.get_parameter('pick_pose_fixed').value]
        self.place_pose_fixed = [float(v) for v in self.get_parameter('place_pose_fixed').value]
        self.home_pose_fixed = [float(v) for v in self.get_parameter('home_pose_fixed').value]
        self.lift_height = float(self.get_parameter('lift_height').value)

        self.pre_pick_move_vx = float(self.get_parameter('pre_pick_move_vx').value)
        self.pre_pick_move_vy = float(self.get_parameter('pre_pick_move_vy').value)
        self.pre_pick_move_sec = float(self.get_parameter('pre_pick_move_sec').value)

        self.post_pick_move_vx = float(self.get_parameter('post_pick_move_vx').value)
        self.post_pick_move_vy = float(self.get_parameter('post_pick_move_vy').value)
        self.post_pick_move_sec = float(self.get_parameter('post_pick_move_sec').value)

        self.mission_forward1_m = float(self.get_parameter('mission_forward1_m').value)
        self.mission_turn_right_deg = float(self.get_parameter('mission_turn_right_deg').value)
        self.mission_forward2_m = float(self.get_parameter('mission_forward2_m').value)
        self.mission_back_after_grasp_m = float(self.get_parameter('mission_back_after_grasp_m').value)
        self.mission_linear_speed = float(self.get_parameter('mission_linear_speed').value)
        self.mission_turn_speed_rad = float(self.get_parameter('mission_turn_speed_rad').value)
        self.turn_backlash_comp_deg = float(self.get_parameter('turn_backlash_comp_deg').value)
        self.distance_scale = float(self.get_parameter('distance_scale').value)
        self.grasp_check_wait_sec = float(self.get_parameter('grasp_check_wait_sec').value)
        self.grasp_retry_max = int(self.get_parameter('grasp_retry_max').value)

        self.pose_x = float(self.get_parameter('initial_x_m').value)
        self.pose_y = float(self.get_parameter('initial_y_m').value)
        self.pose_yaw = math.radians(float(self.get_parameter('initial_yaw_deg').value))
        self.place_target_x = float(self.get_parameter('place_target_x_m').value)
        self.place_target_y = float(self.get_parameter('place_target_y_m').value)

        self.gripper_open_pos = int(self.get_parameter('gripper_open_pos').value)
        self.gripper_close_pos = int(self.get_parameter('gripper_close_pos').value)

        self.latest_target = None
        self.latest_target_stamp = 0.0
        self.state = 'IDLE'
        self.stable_count = 0
        self.last_finish_time = 0.0
        self.need_home_reset = True
        self.last_lost_log_time = 0.0
        self.startup_route_done = False
        self.last_align_update_time = time.time()

        self.create_subscription(PointStamped, '/object_3d_locator/target_point', self.target_callback, 10)
        self.get_logger().info('align_pick_place_node started: close-loop align then fixed arm sequence')

    def target_callback(self, msg):
        self.latest_target = msg
        self.latest_target_stamp = time.time()

    def clip(self, value, limit):
        return max(-limit, min(limit, value))

    def publish_chassis(self, vx=0.0, vy=0.0, wz=0.0):
        self.motor_pub.publish(self.chassis.set_velocity(vx, vy, wz))

    def stop_chassis(self):
        self.publish_chassis(0.0, 0.0, 0.0)

    def target_age(self):
        if self.latest_target is None:
            return float('inf')
        return time.time() - self.latest_target_stamp

    def target_available(self):
        # Allow short perception dropouts near alignment/grasp transition.
        return self.target_age() <= (self.target_timeout_sec + self.target_lost_grace_sec)

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def update_pose(self, vx, vy, wz, dt):
        if dt <= 0.0:
            return
        c = math.cos(self.pose_yaw)
        s = math.sin(self.pose_yaw)
        vx_world = vx * c - vy * s
        vy_world = vx * s + vy * c
        self.pose_x += vx_world * dt
        self.pose_y += vy_world * dt
        self.pose_yaw = self.normalize_angle(self.pose_yaw + wz * dt)

    def tick(self):
        now = time.time()
        dt = now - self.last_align_update_time
        # Prevent one-shot pose jumps after long blocking sections.
        dt = max(0.0, min(dt, 0.1))
        self.last_align_update_time = now
        if self.state == 'COOLDOWN':
            if now - self.last_finish_time >= self.cycle_cooldown_sec:
                self.state = 'IDLE'
                self.need_home_reset = True
            return

        if not self.startup_route_done:
            self.run_startup_route_once()
            self.startup_route_done = True
            self.need_home_reset = True
            self.last_align_update_time = time.time()
            return

        if not self.target_available():
            self.stop_chassis()
            if time.time() - self.last_lost_log_time > 1.0:
                self.get_logger().info('target lost (timeout), back to IDLE and wait reacquire')
                self.last_lost_log_time = time.time()
            self.state = 'IDLE'
            self.stable_count = 0
            self.need_home_reset = True
            return

        if self.need_home_reset and self.state == 'IDLE':
            self.get_logger().info('returning to home pose before alignment')
            self.stop_chassis()
            self.return_to_home_pose()
            self.need_home_reset = False
            return

        target_x = float(self.latest_target.point.x)
        target_y = float(self.latest_target.point.y)
        err_x = target_x - self.align_target_x
        err_y = target_y

        if abs(err_x) <= self.align_x_tol and abs(err_y) <= self.align_y_tol:
            self.stable_count += 1
            self.stop_chassis()
            if self.stable_count >= self.align_stable_cycles:
                self.get_logger().info('aligned, run fixed pick/place sequence')
                task_ok = self.run_fixed_pick_and_place()
                self.stable_count = 0
                self.last_align_update_time = time.time()
                if task_ok:
                    self.last_finish_time = time.time()
                    self.state = 'COOLDOWN'
                else:
                    self.state = 'IDLE'
            return

        self.state = 'ALIGN'
        self.stable_count = 0
        vx = self.clip(self.align_x_kp * err_x, self.align_max_vx)
        vy = self.clip(self.align_y_kp * err_y, self.align_max_vy)
        self.update_pose(vx, vy, 0.0, dt)
        self.publish_chassis(vx, vy, 0.0)

    def move_to_target(self, x, y, z, pitch, duration=1.2):
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
            self.get_logger().error(f'IK failed: [{x:.3f}, {y:.3f}, {z:.3f}, {pitch:.1f}]')
            return False

        msg = ServosPosition()
        msg.duration = float(duration)
        msg.position = [ServoPosition(id=i + 1, position=int(p)) for i, p in enumerate(res.pulse)]
        self.arm_pub.publish(msg)
        time.sleep(duration + 0.2)
        return True

    def set_gripper(self, is_open):
        msg = ServosPosition()
        msg.duration = 0.5
        servo = ServoPosition()
        servo.id = 10
        servo.position = self.gripper_open_pos if is_open else self.gripper_close_pos
        msg.position.append(servo)
        self.arm_pub.publish(msg)
        time.sleep(0.8)

    def drive_for(self, vx, vy, seconds):
        end_time = time.time() + max(0.0, float(seconds))
        last = time.time()
        while time.time() < end_time and rclpy.ok():
            # Keep callbacks alive while moving so target freshness is updated.
            rclpy.spin_once(self, timeout_sec=0.0)
            now = time.time()
            self.update_pose(vx, vy, 0.0, now - last)
            last = now
            self.publish_chassis(vx, vy, 0.0)
            time.sleep(0.05)
        self.stop_chassis()

    def rotate_for(self, angular_z, seconds):
        end_time = time.time() + max(0.0, float(seconds))
        last = time.time()
        while time.time() < end_time and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)
            now = time.time()
            self.update_pose(0.0, 0.0, angular_z, now - last)
            last = now
            self.publish_chassis(0.0, 0.0, angular_z)
            time.sleep(0.05)
        self.stop_chassis()

    def move_distance(self, distance_m, vx=0.0, vy=0.0, speed=0.12):
        # Apply global distance correction factor (e.g., 0.95 for 95%).
        distance = abs(float(distance_m)) * max(0.0, self.distance_scale)
        if distance <= 1e-6:
            return

        # vx/vy are direction hints; normalize then scale by speed (m/s),
        # consistent with motor_control.py and MecanumChassis.set_velocity.
        norm = math.hypot(vx, vy)
        if norm <= 1e-6:
            return

        speed = max(0.01, abs(float(speed)))
        duration = distance / speed
        dir_sign = 1.0 if distance_m >= 0.0 else -1.0
        cmd_vx = dir_sign * (vx / norm) * speed
        cmd_vy = dir_sign * (vy / norm) * speed
        self.drive_for(cmd_vx, cmd_vy, duration)

    def turn_degrees(self, angle_deg, angular_speed_rad=0.8):
        angle_deg = float(angle_deg)
        # Add backlash compensation in the same direction as requested turn.
        if abs(angle_deg) > 1e-6:
            angle_deg += math.copysign(self.turn_backlash_comp_deg, angle_deg)

        angle_rad = math.radians(angle_deg)
        speed = max(0.1, abs(float(angular_speed_rad)))
        duration = abs(angle_rad) / speed
        wz = speed if angle_rad >= 0.0 else -speed
        self.rotate_for(wz, duration)

    def wait_with_spin(self, seconds):
        end_time = time.time() + max(0.0, float(seconds))
        while time.time() < end_time and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def pick_once_with_forward_grasp(self, px, py, pz, pp):
        self.set_gripper(True)

        if not self.move_to_target(px, py, pz + self.lift_height, pp):
            return False
        if not self.move_to_target(px, py, pz, pp, duration=1.0):
            return False

        # While arm keeps low, move forward and then close gripper.
        self.drive_for(self.pre_pick_move_vx, self.pre_pick_move_vy, self.pre_pick_move_sec)
        self.set_gripper(False)

        if not self.move_to_target(px, py, pz + self.lift_height, pp, duration=1.0):
            return False
        return True

    def place_fixed(self, lx, ly, lz, lp):
        if not self.move_to_target(lx, ly, lz + self.lift_height, lp):
            return False
        if not self.move_to_target(lx, ly, lz, lp, duration=1.0):
            return False
        self.set_gripper(True)
        self.move_to_target(lx, ly, lz + self.lift_height, lp, duration=1.0)
        return True

    def run_fixed_pick_and_place(self):
        px, py, pz, pp = self.pick_pose_fixed
        lx, ly, lz, lp = self.place_pose_fixed

        self.stop_chassis()

        # Retry grasp until object is no longer detectable after retreat.
        grasp_success = False
        for attempt in range(1, self.grasp_retry_max + 1):
            self.get_logger().info(f'grasp attempt {attempt}/{self.grasp_retry_max}')
            if not self.pick_once_with_forward_grasp(px, py, pz, pp):
                continue

            self.return_to_home_pose()
            self.get_logger().info('mission: back 30cm to verify grasp')
            self.move_distance(-self.mission_back_after_grasp_m, vx=1.0, vy=0.0, speed=self.mission_linear_speed)
            self.wait_with_spin(self.grasp_check_wait_sec)

            if self.target_available():
                self.get_logger().info('object still detected after retreat, grasp failed; restart alignment here')
                return False

            grasp_success = True
            self.get_logger().info('grasp success: object not detected after retreat')
            break

        if not grasp_success:
            self.get_logger().warn('grasp failed after max retries, restart alignment at current position')
            return False

        # Compute heading and distance to final place target from tracked pose.
        dx = self.place_target_x - self.pose_x
        dy = self.place_target_y - self.pose_y
        target_yaw = math.atan2(dy, dx)
        turn_needed = self.normalize_angle(target_yaw - self.pose_yaw)
        dist_needed = math.hypot(dx, dy)

        self.get_logger().info(
            f'mission: navigate to place target ({self.place_target_x:.2f}, {self.place_target_y:.2f}), '
            f'current ({self.pose_x:.2f}, {self.pose_y:.2f}), '
            f'turn {math.degrees(turn_needed):.1f}deg, move {dist_needed:.2f}m'
        )
        self.turn_degrees(math.degrees(turn_needed), angular_speed_rad=self.mission_turn_speed_rad)
        self.move_distance(dist_needed, vx=1.0, vy=0.0, speed=self.mission_linear_speed)

        self.place_fixed(lx, ly, lz, lp)

        # After placing, rotate and translate back to origin (0, 0).
        back_dx = -self.pose_x
        back_dy = -self.pose_y
        back_dist = math.hypot(back_dx, back_dy)
        if back_dist > 1e-3:
            back_yaw = math.atan2(back_dy, back_dx)
            back_turn = self.normalize_angle(back_yaw - self.pose_yaw)
            self.get_logger().info(
                f'mission: return to origin, current ({self.pose_x:.2f}, {self.pose_y:.2f}), '
                f'turn {math.degrees(back_turn):.1f}deg, move {back_dist:.2f}m'
            )
            self.turn_degrees(math.degrees(back_turn), angular_speed_rad=self.mission_turn_speed_rad)
            self.move_distance(back_dist, vx=1.0, vy=0.0, speed=self.mission_linear_speed)

        self.return_to_home_pose()
        return True

    def run_startup_route_once(self):
        self.move_distance(self.mission_forward1_m,vx=1.0,vy=0.0,speed=self.mission_linear_speed)
        # self.get_logger().info('startup mission: forward 140cm')
        # self.move_distance(self.mission_forward1_m, vx=1.0, vy=0.0, speed=self.mission_linear_speed)

        # self.get_logger().info('startup mission: turn right 90deg')
        # self.turn_degrees(-self.mission_turn_right_deg, angular_speed_rad=self.mission_turn_speed_rad)

        # self.get_logger().info('startup mission: forward 120cm')
        # self.move_distance(self.mission_forward2_m, vx=1.0, vy=0.0, speed=self.mission_linear_speed)

    def return_to_home_pose(self):
        hx, hy, hz, hp = self.home_pose_fixed
        self.move_to_target(hx, hy, hz, hp, duration=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = AlignPickPlaceController()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            node.tick()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_chassis()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
