### # 基于深度相机的坐标映射与三维定位
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from interfaces.msg import ObjectsInfo
from cv_bridge import CvBridge
import numpy as np

class Object3DLocator(Node):
    def __init__(self):
        super().__init__('object_3d_locator_node')
        self.bridge = CvBridge()  # 实例化图像转换工具
        self.declare_parameter('target_class', 'brick')
        self.target_class = self.get_parameter('target_class').value
        # 创建两个空变量，用于后续对接深度图矩阵和相机内参
        self.latest_depth_img = None
        self.intrinsics = None
        self.target_pub = self.create_publisher(PointStamped, '/object_3d_locator/target_point', 10)
        # 1.订阅相机内参，用于获取焦距和光心
        self.create_subscription(
            CameraInfo,
            '/ascamera/camera_publisher/rgb0/camera_info',
            self.info_callback,
            qos_profile_sensor_data)
        # 2. 订阅深度图像，用于获取相机坐标下的深度信息
        self.create_subscription(
            Image,
            '/ascamera/camera_publisher/depth0/image_raw',
            self.depth_callback,
            qos_profile_sensor_data)

        # 3. 订阅YOLO识别结果，用于获取目标的2D像素坐标（u，v）
        self.create_subscription(
            ObjectsInfo,
            '/yolo_node/object_detect',
            self.detection_callback,
            1)
        self.get_logger().info('目标三维空间定位节点已启动，正在等待 YOLO 识别目标...')
        
    def info_callback(self, msg):
        """获取相机内参"""
        if self.intrinsics is None:
            self.intrinsics = msg.k
            self.get_logger().info(f"深度相机内参获取：fx={self.intrinsics[0]:.1f}, fy={self.intrinsics[4]:.1f}")

    def depth_callback(self, msg):
        """实时将深度信息转换为Python能直接提取的深度信息二维数字矩阵"""
        try:
            # 将msg转换为数字矩阵：latest_depth_img，矩阵元素为深度（单位：毫米）
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, '16UC1')
        except Exception as e:
            self.get_logger().error(f"深度图转换失败：{e}")
            
    def detection_callback(self, msg):
        """当YOLO识别到物体时触发"""
        # 检查相机内参和深度图像是否获取成功
        if self.intrinsics is None or self.latest_depth_img is None:
            return
        # 遍历YOLO识别出的物体
        for obj in msg.objects:
            if obj.class_name == self.target_class:
                # 提取预测框的中心点作为目标的2D像素坐标 (u, v)
                if len(obj.box) >= 4:
                    u = (obj.box[0] + obj.box[2]) / 2
                    v = (obj.box[1] + obj.box[3]) / 2
                    # 将2D像素坐标转换为相机坐标系下的3D坐标
                    camera_coords = self.get_camera_3d_coordinates(u, v)

                    if camera_coords is None:
                        continue
                    x_c, y_c, z_c = camera_coords

                    # 转换为机器人基座坐标系下的3D坐标
                    r_x, r_y, r_z = self.transform_to_robot_frame(x_c, y_c, z_c)

                    point_msg = PointStamped()
                    point_msg.header.stamp = self.get_clock().now().to_msg()
                    point_msg.header.frame_id = 'robot_base'
                    point_msg.point.x = float(r_x)
                    point_msg.point.y = float(r_y)
                    point_msg.point.z = float(r_z)
                    self.target_pub.publish(point_msg)

                    # 实时在终端打印三维坐标
                    self.get_logger().info(
                        f"\n锁定目标: [{obj.class_name}] \n"
                        f"2D像素坐标: (U: {u:.1f}, V: {v:.1f}) \n"
                        f"相机坐标下3D坐标: (X: {x_c:.3f}m, Y: {y_c:.3f}m, Z: {z_c:.3f}m) \n"
                        f"基座坐标下3D坐标: (X: {r_x:.3f}m, Y: {r_y:.3f}m, Z: {r_z:.3f}m)"
                    )
                    
    def get_camera_3d_coordinates(self, u, v):
        """利用小孔成像原理，将2D像素坐标转换为相机坐标系下的3D坐标"""
        # 获取深度图的宽和高，将浮点数坐标转换为整数（像素只能是整数）
        h, w = self.latest_depth_img.shape
        u_int, v_int = int(u), int(v)
        # 边界保护，防止YOLO的预测框超出画面之外影响定位效果
        if u_int < 0 or u_int >= w or v_int < 0 or v_int >= h:
            return None
        # 感兴趣区域（ROI）切片，以目标中心点为圆心，向四周扩展15个像素，切出一个30x30的正方形区域
        radius = 15
        u_min, u_max = max(0, u_int - radius), min(w, u_int + radius + 1)
        v_min, v_max = max(0, v_int - radius), min(h, v_int + radius + 1)
        # roi为装了900个深度值的矩阵
        roi = self.latest_depth_img[v_min:v_max, u_min:u_max]
        # 过滤掉深度为0的无效噪点
        valid_depths = roi[roi > 0]
        if len(valid_depths) == 0:
            return None
        # 计算平均深度，并转换为米（m）
        depth_mm = np.mean(valid_depths)
        z_c = depth_mm / 1000.0

        # 提取相机内参参数（为出厂设定）
        # intrinsics为相机内参矩阵：[fx 0 cx; 0 fy cy; 0 0 1]
        # fx, fy分别为x轴和y轴方向上的焦距
        # cx, cy分别为图像光心在x轴和y轴的像素坐标
        fx, fy = self.intrinsics[0], self.intrinsics[4]
        cx_img, cy_img = self.intrinsics[2], self.intrinsics[5]

        # 利用小孔成像逆投影公式，像素坐标反投影到相机物理坐标
        x_c = (u - cx_img) * z_c / fx
        y_c = (v - cy_img) * z_c / fy
        return x_c, y_c, z_c
    
    def transform_to_robot_frame(self, x_c, y_c, z_c):
        """将相机的3D坐标，转换为机器人基座坐标系的3D坐标"""
        # 相机硬件安装参数（机械臂初始化状态下测量所得）
        TILT_ANGLE = 45.0  # 相机向下倾斜的角度
        CAMERA_HEIGHT = 0.22  # 相机距离地面的高度，单位：米
        CAMERA_OFFSET_X = 0.06  # 相机中心相对于机器人基座中心的向前偏移量，单位：米

        # 将45度转换为弧度
        theta = np.radians(TILT_ANGLE)
        # 机器人基座坐标系下前方距离（Robot X）= 相机Z分量 - 相机Y分量 + 向前偏移
        px = (z_c * np.cos(theta)) - (y_c * np.sin(theta)) + CAMERA_OFFSET_X
        # 机器人基座坐标系下左右距离（Robot Y）= -相机X分量
        py = -x_c
        # 机器人基座坐标系下高度（Robot Z）= 安装高度 - 下降高度
        vertical_drop = (y_c * np.cos(theta)) + (z_c * np.sin(theta))
        pz = CAMERA_HEIGHT - vertical_drop
        return px, py, pz
    
def main(args=None):
    rclpy.init(args=args)
    node = Object3DLocator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
