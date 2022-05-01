"""This script is used to collect driving data in LGSVL."""

import os
from datetime import datetime
import errno
import csv
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from lgsvl_msgs.msg import CanBusData
import message_filters


def mkdir_p(path):
    """create directory for saving data."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    print(f'Directory is ready: {path}')


class collect_node(Node):
    """Collect information in LGSVL."""

    def __init__(self):
        """Define the data-collecting path."""
        self.basepath = '/home/yinghanhuang/MyWarehouse/' \
                        'event_based_vision/lgsvl_simu/three_camera'
        self.csv_path = f'{self.basepath}/data/sequence3/dataframe'
        self.center_img_path = f'{self.basepath}/data/sequence3/center_img'
        # Should be adjusted accordingly.

        super().__init__('collect', allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        self.log = self.get_logger()
        self.log.info('Starting Collect node...')
        mkdir_p(self.csv_path)
        mkdir_p(self.center_img_path)
        self.log.info('Center camera topic:/simulator/'
                      'sensor/camera/center/image/compressed')
        self.log.info('Canbus topic: /simulator/canbus')
        # create subscriber for fetching camera images
        sub_center_camera = message_filters.Subscriber(
            self,
            CompressedImage,
            '/simulator/sensor/camera/center/image/compressed'
        )
        # create subscriber for fetching information
        sub_canbus = message_filters.Subscriber(
            self, CanBusData, '/simulator/canbus')
        t_s = message_filters.ApproximateTimeSynchronizer(
            [sub_center_camera, sub_canbus], 1, 0.1)
        # queue size, delay
        t_s.registerCallback(self.callback)
        self.log.info('Up and running...')

    def callback(self, center_camera, canbus):
        """Call back function of subscriber."""
        ts_sec = center_camera.header.stamp.sec
        ts_nsec = center_camera.header.stamp.nanosec
        steer_cmd = canbus.steer_pct
        throttle_cmd = canbus.throttle_pct
        brake_cmd = canbus.brake_pct
        self.log.info(f"[{ts_sec}.{ts_nsec}] Format: {center_camera.format},"
                      f" steering_cmd: {steer_cmd}, "
                      f" throttle_cmd: {throttle_cmd},"
                      f" brake_cmd: {brake_cmd}")
        msg_id = str(datetime.now().isoformat())
        self.save_image(center_camera, msg_id)
        self.save_csv(steer_cmd, throttle_cmd, brake_cmd, msg_id)

    def save_image(self, center_camera, msg_id):
        """Save the image."""
        center_img_np_arr = np.fromstring(bytes(center_camera.data), np.uint8)
        center_img_cv = cv2.imdecode(center_img_np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(
            self.center_img_path,
            f'center-{msg_id}.jpg'), center_img_cv)

    def save_csv(self, steer_cmd, throttle_cmd, brake_cmd, msg_id):
        """Save control commands."""
        with open(os.path.join(self.csv_path, 'steering.csv'),
                  'a+', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([msg_id, steer_cmd, throttle_cmd, brake_cmd])


def main(args=None):
    """Initialize and start ros2 node"""
    rclpy.init(args=args)
    collection_node = collect_node()
    rclpy.spin(collection_node)


if __name__ == '__main__':
    main()
