import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class TargetMarkerViz(Node):
    def __init__(self):
        super().__init__('target_marker_viz_node')
        self.latest_target = None
        self.marker_pub = self.create_publisher(MarkerArray, '/object_3d_locator/markers', 10)
        self.create_subscription(
            PointStamped,
            '/object_3d_locator/target_point',
            self.target_callback,
            10,
        )
        self.create_timer(0.2, self.publish_markers)
        self.get_logger().info('目标可视化节点已启动，发布 /object_3d_locator/markers')

    def target_callback(self, msg):
        self.latest_target = msg

    def _new_marker(self, frame_id, ns, marker_id, marker_type):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0
        return marker

    def publish_markers(self):
        marker_array = MarkerArray()

        # Robot base origin marker
        base = self._new_marker('robot_base', 'robot_base', 0, Marker.SPHERE)
        base.pose.orientation.w = 1.0
        base.scale.x = 0.04
        base.scale.y = 0.04
        base.scale.z = 0.04
        base.color.r = 1.0
        base.color.g = 1.0
        base.color.b = 0.0
        base.color.a = 1.0
        marker_array.markers.append(base)

        base_text = self._new_marker('robot_base', 'robot_base', 1, Marker.TEXT_VIEW_FACING)
        base_text.pose.position.z = 0.06
        base_text.pose.orientation.w = 1.0
        base_text.scale.z = 0.05
        base_text.color.r = 1.0
        base_text.color.g = 1.0
        base_text.color.b = 0.0
        base_text.color.a = 1.0
        base_text.text = 'robot_base (0,0,0)'
        marker_array.markers.append(base_text)

        if self.latest_target is not None:
            frame_id = self.latest_target.header.frame_id or 'robot_base'

            target = self._new_marker(frame_id, 'target', 0, Marker.SPHERE)
            target.pose.position.x = float(self.latest_target.point.x)
            target.pose.position.y = float(self.latest_target.point.y)
            target.pose.position.z = float(self.latest_target.point.z)
            target.pose.orientation.w = 1.0
            target.scale.x = 0.03
            target.scale.y = 0.03
            target.scale.z = 0.03
            target.color.r = 1.0
            target.color.g = 0.2
            target.color.b = 0.2
            target.color.a = 1.0
            marker_array.markers.append(target)

            label = self._new_marker(frame_id, 'target', 1, Marker.TEXT_VIEW_FACING)
            label.pose.position.x = float(self.latest_target.point.x)
            label.pose.position.y = float(self.latest_target.point.y)
            label.pose.position.z = float(self.latest_target.point.z) + 0.05
            label.pose.orientation.w = 1.0
            label.scale.z = 0.04
            label.color.r = 0.9
            label.color.g = 1.0
            label.color.b = 0.9
            label.color.a = 1.0
            label.text = (
                f"target x={self.latest_target.point.x:.3f}, "
                f"y={self.latest_target.point.y:.3f}, "
                f"z={self.latest_target.point.z:.3f}"
            )
            marker_array.markers.append(label)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = TargetMarkerViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
