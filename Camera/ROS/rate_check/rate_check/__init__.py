import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

SUBSCRIBER_NAME = "camera_object_detection"

class RateChecker(Node):
    def __init__(self):
        super().__init__('rate_checker')
        self.subscription = self.create_subscription(String, SUBSCRIBER_NAME, self.callback, 10)
        self.last_time_ = time.monotonic()
        self.count_ = 0
        self.max_rate_ = 0.0
        self.timer_ = self.create_timer(1.0, self.timer_callback)

    def callback(self, msg):
        self.count_ += 1

    def timer_callback(self):
        current_time = time.monotonic()
        elapsed_time = current_time - self.last_time_
        if elapsed_time > 0:
            rate = self.count_ / elapsed_time
            if rate > self.max_rate_:
                self.max_rate_ = rate
            print(f"Max publishing rate of {SUBSCRIBER_NAME} topic: {str(self.max_rate_)} Hz")
            self.last_time_ = current_time
            self.count_ = 0

def main(args=None):
    rclpy.init(args=args)
    rate_checker = RateChecker()
    rclpy.spin(rate_checker)
    rate_checker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
