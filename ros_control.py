"""Control vehicle using ros Node in LGSVL.

Ros node is responsible for subscribing pics and,
publishing predicted action.
"""

import sys
import time
import threading
import os
from datetime import datetime
import csv
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.clock import ClockType
from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage
from lgsvl_msgs.msg import CanBusData, \
    VehicleControlData
import torch
from torchvision import transforms
import numpy as np
import cv2
from wiring import NCPWiring
from nets.cnn_head import ConvolutionHead_Nvidia
from nets.ltc_cell import LTCCell
from nets.models_all import Convolution_Model, \
    LSTM_Model, GRU_Model, CTGRU_Model, NCP_Model
from utils import __crop


class Drive(Node):
    """Ros Node which acts as both subscriber and publisher."""

    def __init__(self, network='CNN'):
        """Initialize important variables."""
        super().__init__('Drive',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        self.network = network
        s_dim = [3, 66, 200]
        a_dim = 1
        seq_length = 1
        self.steering = 0
        self.hidden = None
        self.img = None

        if network == 'CNN':
            self.policy = Convolution_Model(s_dim, a_dim)  # CNN
            self.csv_path = '/home/yinghanhuang/Desktop/' \
                            'xue/trained_models/cnn'
            self.policy.load('/home/yinghanhuang/Desktop/'
                             'xue/trained_models/cnn/')
        elif network == 'LSTM':
            # CNN+LSTM
            conv_head = ConvolutionHead_Nvidia(
                s_dim,
                seq_length,
                num_filters=32,
                features_per_filter=4)
            self.policy = LSTM_Model(
                conv_head=conv_head,
                time_step=seq_length,
                hidden_size=16,
                output=a_dim)
            self.csv_path = '/home/yinghanhuang/Desktop/' \
                            'xue/trained_models/cnn+lstm'
            self.policy.load('/home/yinghanhuang/Desktop/'
                             'xue/trained_models/cnn+lstm/')
        elif network == 'GRU':
            # CNN+GRU
            conv_head = ConvolutionHead_Nvidia(
                s_dim,
                seq_length,
                num_filters=32,
                features_per_filter=4)
            self.policy = GRU_Model(
                conv_head=conv_head,
                time_step=seq_length,
                hidden_size=16,
                output=a_dim)
            self.csv_path = '/home/yinghanhuang/Desktop/' \
                            'xue/trained_models/cnn+gru'
            self.policy.load('/home/yinghanhuang/Desktop/'
                             'xue/trained_models/cnn+gru/')
        elif network == 'CTGRU':
            # CNN+CTGRU
            conv_head = ConvolutionHead_Nvidia(
                s_dim,
                seq_length,
                num_filters=32,
                features_per_filter=4)
            self.policy = CTGRU_Model(16,
                                      conv_head=conv_head,
                                      time_step=1,
                                      output=1,
                                      use_cuda=False)
            self.csv_path = '/home/yinghanhuang/Desktop/' \
                            'xue/trained_models/cnn+ctgru'
            self.policy.load('/home/yinghanhuang/Desktop/'
                             'xue/trained_models/cnn+ctgru/')
        elif network == 'NCP':
            # CNN+NCP
            conv_head = ConvolutionHead_Nvidia(
                s_dim,
                seq_length,
                num_filters=32,
                features_per_filter=4)
            input_shape = (1, 128)
            wiring = NCPWiring(
                inter_neurons=22, command_neurons=12,
                motor_neurons=1, sensory_fanout=16,
                inter_fanout=8, recurrent_command=8, motor_fanin=6)
            wiring.build(input_shape)
            ltc_cell = LTCCell(wiring=wiring, time_interval=0.2)
            self.policy = NCP_Model(ltc_cell=ltc_cell, conv_head=conv_head)
            self.csv_path = '/home/yinghanhuang/Desktop/' \
                            'xue/trained_models/cnn+ncp'
            self.policy.load('/home/yinghanhuang/Desktop/'
                             'xue/trained_models/cnn+ncp/')
        else:
            print("wrong network type")
            sys.exit()

        self.policy.eval()
        print(self.policy.device)

        # transform of the image
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(
                lambda image: __crop(image, (100, 80), (528, 176))
            ),
            transforms.Resize((66, 200)),
            transforms.ToTensor()
           ])

        self.log = self.get_logger()
        self.log.info('Starting Collect node...')
        self.image_lock = threading.RLock()

        # Ros communication
        self.log.info('Camera:/simulator/sensor/'
                      'camera/center/image/compressed')

        # subscribe the image of center camera
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/simulator/sensor/camera/center/image/compressed',
            self.framecallback, 1)
        # subscribe the velocity
        self.vel_sub = self.create_subscription(
            CanBusData, '/simulator/canbus', self.velcallback, 1)

        # publish the command by using vehicle control data
        self.control_pub = self.create_publisher(
            VehicleControlData,
            '/simulator/vehicle_control',
            1
        )

        # Ros timer
        self.publish_period = 0.2  # 5 Hz during test,
        self.timer = self.create_timer(1, self.check_camera_topic)
        self.inference_time = 0.

        # FPS
        self.last_time = time.time()
        self.frames = 0
        self.fps = 0.

        self.log.info('Up and running...')

    def framecallback(self, img):
        """Read the image, predict the action."""
        self.get_fps()
        if self.image_lock.acquire(True):
            self.img = img
            t_start = time.time()
            self.steering = float(self.predict(self.policy, self.img))
            t_end = time.time()
            self.inference_time = t_end - t_start
            self.image_lock.release()

    def velcallback(self, canbus):
        """Get the velocity of vehicle."""
        msg_id = str(datetime.now().isoformat())
        velocity = canbus.speed_mps
        self.save_csv(velocity, msg_id)

    def save_csv(self, velocity, msg_id):
        """Save the vehicle velocity in csv file."""
        with open(os.path.join(self.csv_path, 'velocity.csv'),
                  'a+', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([msg_id, velocity])

    def convert_timestamp(seconds):
        """Convert time stamps."""
        nanosec = seconds * 1e9
        secs = (int)(nanosec // 1e9)
        nsecs = (int)(nanosec % 1e9)
        time_t = Time(seconds=secs, nanoseconds=nsecs,
                      clock_type=ClockType.STEADY_TIME)
        return time_t

    def predict(self, img):
        """Predict actions based on the input."""
        byte_input = np.fromstring(bytes(img.data), np.uint8)
        img = cv2.imdecode(byte_input, cv2.IMREAD_COLOR)

        img = self.transform(img)
        if self.network == "CNN":
            img = torch.unsqueeze(img, 0)  # added later
            steering = self.policy(img)
            # squeeze from (1,1) to (1)
            steering = torch.squeeze(steering)
        else:  # for RNN
            img = torch.unsqueeze(img, 0)
            img = torch.unsqueeze(img, 0)
            steering, self.hidden = \
                self.policy.evaluate_on_single_sequence(img, self.hidden)
            # squeeze from (1,1,1) to (1)
            steering = torch.squeeze(steering)
        return steering

    def publish_callback(self):
        """Publish actions."""
        if self.img is None:
            return
        # give the latest predicted_action to the message
        header = Header()
        header.stamp = convert_timestamp(self.inference_time).to_msg()

        control = VehicleControlData()
        control.header = header
        control.target_wheel_angle = self.steering

        # publish the message
        self.control_pub.publish(control)
        self.get_logger().info(f'Publishing: '
                               f'{"Send steering to Vehicle Control"}')
        self.get_logger().info(f'[{time.time():.3f}] steering:'
                               f'{control.target_wheel_angle}')
        """ 
        self.get_logger().info('[{:.3f}] throttle: "{}"'.
                               format(time.time(), control.acceleration_pct))
        self.get_logger().info('[{:.3f}] brake: "{}"'.
                               format(time.time(), control.braking_pct))
        """

    def get_fps(self):
        """Get fps."""
        self.frames += 1
        now = time.time()
        if now >= self.last_time + 1.0:
            delta = now - self.last_time
            self.last_time = now
            self.fps = self.frames/delta
            self.frames = 0

    def check_camera_topic(self):
        """Check the topic of camera."""
        if self.img is None:
            self.log.info('Waiting for a camera image')
        else:
            self.log.info('Received a camera image')
            self.destroy_timer(self.timer)
            self.create_timer(self.publish_period, self.publish_callback)


def main(args=None):
    """Define the main function."""
    rclpy.init(args=args)
    simulation = Drive(network='NCP')
    rclpy.spin(simulation)


if __name__ == '__main__':
    main()
