import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

class AEBNode(Node):
    def __init__(self):
        super().__init__('aeb_node')

        # init variables
        self.speed = 0.0
        self.ttc_threshold = 1.0  # slightly more conservative threshold (seconds)
        self.braking = False       # track brake state to avoid redundant publishes

        # subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # publisher
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.get_logger().info("AEB Node initialized and watching for collisions.")

    def odom_callback(self, msg):
        self.speed = msg.twist.twist.linear.x

        # reset braking flag once car has stopped — allows recovery after AEB trigger
        if abs(self.speed) < 0.05:
            self.braking = False

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)

        # replace invalid/inf readings with a large safe value
        ranges = np.where(np.isfinite(ranges) & (ranges > 0), ranges, 1e6)

        # build angle array for all beams
        num_beams = len(ranges)
        angles = msg.angle_min + np.arange(num_beams) * msg.angle_increment

        # only consider beams pointing generally forward (within ±30 degrees)
        forward_mask = np.abs(angles) <= np.radians(30)
        front_ranges = ranges[forward_mask]
        front_angles = angles[forward_mask]

        # iTTC formula: range_rate = v * cos(angle), TTC = range / range_rate
        # clamp range_rate to avoid division by zero or negative (receding objects)
        range_rate = self.speed * np.cos(front_angles)
        range_rate_clamped = np.maximum(range_rate, 1e-6)

        ttc = front_ranges / range_rate_clamped

        min_ttc = np.min(ttc)

        self.get_logger().info(
            f"Speed: {self.speed:.2f} m/s | Min TTC: {min_ttc:.2f}s",
            throttle_duration_sec=0.5
        )

        if min_ttc < self.ttc_threshold and not self.braking:
            self.emergency_brake()

    def emergency_brake(self):
        self.braking = True

        stop_msg = AckermannDriveStamped()
        stop_msg.header.stamp = self.get_clock().now().to_msg()
        stop_msg.drive.speed = 0.0
        stop_msg.drive.acceleration = -10.0

        self.drive_pub.publish(stop_msg)
        self.get_logger().warn("[AEB TRIGGERED] Emergency brake activated!", throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = AEBNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()