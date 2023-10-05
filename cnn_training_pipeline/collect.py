#! /usr/bin/env python3

"""@package enph353_ros_lab
Python executable for collecting training data for imitation learning
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
from datetime import datetime
import os

"""some code for the saving of data is adapted from https://www.youtube.com/watch?v=_wi2L-KrGqk"""

#100 frames per 7 seconds

class collector:


  def __init__(self):
    """The constructor."""

    self.vel_sub = rospy.Subscriber('/R1/cmd_vel', Twist, self.getTwist)
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    self.bridge = CvBridge()

    collector.linVel = 0
    collector.angVel = 0
    collector.masterArr = np.eye(4)

    myDirectory = "/home/fizzer/Pictures/"
    countFolder = 0
    while os.path.exists(os.path.join(myDirectory,f'IMG{str(countFolder)}')):
      countFolder += 1

    collector.newPath = myDirectory + "/IMG"+str(countFolder)
    os.makedirs(collector.newPath)
    collector.count = 0
    global imgList, stateList
    imgList = []
    stateList = []

    




  def getTwist(self,data):
    collector.linVel = data.linear.x
    collector.angVel = data.angular.z

  def callback(self,data):
    """Feedback loop."""
    collector.count += 1
    print(collector.count)
    #convert the raw image to cv2 format
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    #don't count the first 14 seconds of data to account for the user getting ready
    #this doesn't remove the last 14 seconds of data so terminate the program quickly to prevent getting too much junk data when you aren't controlling the robot!
    if collector.count > 200:
      if collector.linVel != 0:
        state = collector.masterArr[2]
      elif collector.angVel > 0:
        state = collector.masterArr[1]
      elif collector.angVel < 0:
        state = collector.masterArr[3]
      else:
        state = collector.masterArr[0]
      
      now = datetime.now()
      timestamp = str(datetime.timestamp(now)).replace('.','')


      fileName = os.path.join(collector.newPath,f'Image_{timestamp}.jpg')
      cv2.imwrite(fileName,cv_image)
      imgList.append(fileName)
      stateList.append(state)
    cv2.imshow("Image window", cv_image)
    cv2.waitKey(1)

"""Main method."""
def main(args):
  rospy.init_node('collector', anonymous=True)
  ic = collector()



  #loop through calback() forever
  try:
    rospy.spin()
  finally:
    print("Shutting down")
    np.save(collector.newPath + "/imgList.npy",imgList)
    np.save(collector.newPath + "/stateList.npy",stateList)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main(sys.argv)