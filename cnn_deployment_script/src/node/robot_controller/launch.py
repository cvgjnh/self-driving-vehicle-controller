#! /usr/bin/env python3

import os
from geometry_msgs.msg import Twist
import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time

import plate_detector_SIFT
import run
from subprocess import call

"""Main method."""
def main(args):
  # os.system('source ~/ros_ws/devel/setup.bash')
  # os.system('cd ~/ros_ws/src/2022_competition/enph353/enph353_utils/scripts')
  # os.system('./run_sim.sh -vpg')
  
  # call(["python3", "run.py"])
  # call(["python3", "plate_detector_SIFT.py"])

  os.system("/home/fizzer/ros_ws/src/controller_pkg/src/node/robot_controller/run.py")
  os.system("/home/fizzer/ros_ws/src/controller_pkg/src/node/robot_controller/plate_detector_SIFT.py")



if __name__ == '__main__':
    main(sys.argv)