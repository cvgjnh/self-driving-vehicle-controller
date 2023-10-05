
#! /usr/bin/env python3

"""@package enph353_ros_lab
Python executable for line following.
"""

from __future__ import print_function
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
from skimage.metrics import structural_similarity
import os

from tensorflow.keras import models




class runner:
  """Object that processes ROS camera images and publishes robot velocity commands."""

  def __init__(self, argv):
    runner.Y_CROP = 300
    scale_percent = 12.5 # percent of original size
    width = int(1280 * scale_percent / 100)
    height = int((720-runner.Y_CROP) * scale_percent / 100)
    runner.dim = (width, height)

    runner.cooldown = 0
    
    self.dir_path = str(os.path.dirname(os.path.realpath(__file__)))
    print(self.dir_path)
    runner.imgLast = cv2.imread(self.dir_path + '/sample.png', 0)[int(720/2):,int(1280*5/12):int(1280*7/12)]

    runner.scoreLast = 0
    runner.waitCount = 0
    runner.wait = False
    runner.pedestrianCount = 0
    runner.countTillPID = float('inf')

    runner.conv_model = models.load_model(self.dir_path + '/NN_8RALL.h5')

    #call the script with an argument if starting from the beginning
    if len(argv) > 1:
      runner.firstLoop = True
    else:
      runner.firstLoop = False
    

    """The constructor."""
    #initialize a publisher to publish velocity commands to robot
    self.image_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

    #initialize a subcriber to take in ROS camera images
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)

    self.state = 0
    self.entered_inner = False
    self.race_begin_time = time.time()

    time.sleep(1)



  def callback(self,data):
    """Feedback loop."""
    start_time = time.time()

    #convert the raw image to cv2 format
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    move = Twist()

    if self.state == 0:
      #if starting in the starting position (need to call script with an argument to activate)
      if runner.firstLoop == True:
        time.sleep(1)
        
        move.linear.x = .25
        move.angular.z = 0
        self.image_pub.publish(move)
        time.sleep(1.5)
        
        move = Twist()
        move.angular.z = 1
        move.linear.x = 0
        self.image_pub.publish(move)
        time.sleep(1.5)

        runner.firstLoop = False

      #if waiting at red line
      elif runner.wait == True:
        
        # Load images as grayscale
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)[int(720/2):,int(1280*5/12):int(1280*7/12)]
        # Compute SSIM between the two images
        (score, diff) = structural_similarity(cv_image, runner.imgLast, full=True)
        print("Image Similarity: {:.4f}%".format(score * 100))
        
        #if passenger just moved across the middle
        #can tell when the difference between similarity scores jumps
        if runner.waitCount > 5 and np.abs(score - runner.scoreLast) > 0.02:
          runner.wait = False
          runner.cooldown = 100
          if runner.pedestrianCount == 4:
            runner.countTillPID = time.time()
          # move.linear.x = .15
          # move.angular.z = 0
          # self.image_pub.publish(move)
          # time.sleep(2.2 )
          # move = Twist()
          # move.linear.x = 0
          # move.angular.z = 0
          # self.image_pub.publish(move)

        #update the last image and last similarity score
        runner.imgLast = cv_image
        runner.scoreLast = score
        runner.waitCount += 1

      #check if in front of red line
      elif cv_image[650][640][0] <20 and runner.cooldown == 0:
          move.linear.x = 0
          move.angular.z = 0
          runner.wait = True
          runner.waitCount = 0
          runner.pedestrianCount += 1

      #regular movement
      else:
        img_aug = cv_image[300:]
        img_aug = cv2.resize(img_aug, runner.dim, interpolation = cv2.INTER_AREA)

        img_aug = img_aug.astype(np.float32)/255
        img_aug = np.expand_dims(img_aug, axis=0)
        output_state = runner.conv_model.predict(img_aug)[0]
        maxIndex = output_state.tolist().index(max(output_state))

        #change stop to go straight for now
        if maxIndex == 0 or maxIndex == 2:
          move.linear.x = .18
          move.angular.z = 0

        elif maxIndex == 1:
          move.linear.x = 0
          move.angular.z = 0.5

        elif maxIndex == 3:
          move.linear.x = 0
          move.angular.z = -0.5

      self.image_pub.publish(move)
      if runner.cooldown != 0:
        runner.cooldown -= 1 
      
      if runner.pedestrianCount == 4 and time.time() - runner.countTillPID > 12:
        self.state = 1
      
    elif self.state == 1:
      row, column, ch = cv_image.shape
      input = cv_image[int(row*2/3):]
      mask = self.mask_road(input)

      cv2.imshow('asdf', mask)
      cv2.waitKey(1)

      cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      biggest_contour, ind = self.find_max_area_contour(cnts, hierarchy)

      area = cv2.contourArea(biggest_contour)
      print(area)
      M = cv2.moments(biggest_contour)
      cm = [int(M['m10']/M['m00']), int(M['m01']/M['m00'])]

      if area < 260000:
        move.linear.x = 0.3
        move.angular.z = (column/2 - cm[0])/100
        self.image_pub.publish(move)
      else:
        if self.entered_inner:
          move.linear.x = 0.3
          move.angular.z = (column/2 - cm[0])/100

        else:
          self.entered_inner = True
          self.enter_inner_loop()


    # print(time.time() - start_time)

  def mask_road(self, image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0,0,75])
    upper_black = np.array([0,0,90])
    mask = cv2.inRange(hsv, lower_black, upper_black)

    return mask.astype(np.uint8)

  def find_max_area_contour(self, cnt, hierarchy):
    max_area = 0
    ind = 0

    for i in range(len(cnt)):
      area = cv2.contourArea(cnt[i])
      if area > max_area and hierarchy[0, i, -1] == -1:
        max_area = area
        ind = i
    
    return cnt[ind], ind

  def enter_inner_loop(self):
    move = Twist()
    move.linear.x = 0.3
    move.angular.z = 0
    self.image_pub.publish(move)
    time.sleep(0.5)

    move.linear.x = 0
    move.angular.z = 1.2
    self.image_pub.publish(move)
    time.sleep(1.5)
    print("turned left")

    move = Twist()
    move.linear.x = 0.3
    move.angular.z = 0
    self.image_pub.publish(move)
    time.sleep(1.8)

    move.linear.x = 0
    move.angular.z = 1.2
    self.image_pub.publish(move)
    time.sleep(1.5)
    print("turned left")

    move = Twist()
    move.linear.x = 0.3
    move.angular.z = 0
    self.image_pub.publish(move)
    time.sleep(0.1)
  
  def turn_right(self):
    move = Twist()
    move.linear.x = 0.3
    move.angular.z = 0
    self.image_pub.publish(move)
    time.sleep(1.5)

    move.linear.x = 0
    move.angular.z = -1.2
    self.image_pub.publish(move)
    time.sleep(2)
    print("turned right")

    move.linear.x = 0.3
    move.angular.z = 0
    self.image_pub.publish(move)
    time.sleep(0.2)

"""Main method."""
def main(args):
  rospy.init_node('runner', anonymous=True)
  ic = runner(args)

  #loop through calback() forever
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)