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
from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
import os
import string
import random
import time
from skimage.metrics import structural_similarity


from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
import tensorflow.image as tfimg

class controller:
  """Object that processes ROS camera images and publishes robot velocity commands."""

  def __init__(self, argv):
    """The constructor."""
    self.dir_path = str(os.path.dirname(os.path.realpath(__file__)))
    print(self.dir_path)

    controller.count = 0

    controller.img = cv2.imread(self.dir_path + "/P.png")
    controller.img  = cv2.cvtColor(controller.img, cv2.COLOR_BGR2GRAY)
    controller.sift = cv2.SIFT_create()
    controller.kp_image, controller.desc_image = controller.sift.detectAndCompute(controller.img, None)

    # Feature matching
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    controller.flann = cv2.FlannBasedMatcher(index_params, search_params)


    #initialize a publisher to publish velocity commands to robot
    self.image_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

    #initialize a subcriber to take in ROS camera images
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)

    self.license_pub = rospy.Publisher("/license_plate", String, queue_size=1)
    
    self.conv_model = models.load_model(self.dir_path + '/num_detect.h5')
    self.char_model = models.load_model(self.dir_path + '/char_detect.h5')


    self.last_predicted_ID = -1

    self.frequency = [[{},{},{},{}],[{},{},{},{}],[{},{},{},{}],[{},{},{},{}],[{},{},{},{}],[{},{},{},{}],[{},{},{},{}],[{},{},{},{}]]
    self.guess_last_license_counter = 0

    time.sleep(1)




  def callback(self,data):
    start_time = time.time()
    try:
      frame = self.bridge.imgmsg_to_cv2(data, "bgr8")

    except CvBridgeError as e:
      print(e)
    

    #can only sift gray
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp_grayframe, desc_grayframe = controller.sift.detectAndCompute(grayframe, None)

    #k=2 means the two closest matches to each keypoint are found
    matches = controller.flann.knnMatch(controller.desc_image, desc_grayframe, k=2)
    good_points = []

    matchesMask = [[0,0] for i in range(len(matches))]

    #lower distance means closer match, a point is only good if the #1 point is a lot better than the #2 point
    for i,(m,n) in enumerate(matches):
      if m.distance < 0.6*n.distance:
        matchesMask[i]=[1,0]
        good_points.append(m)


    dst_pt = [ kp_grayframe[m.trainIdx].pt for m in good_points ]
    #print(dst_pt)

    #img3 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)
    

    if len(dst_pt) >1:
      xvalues = []
      yvalues = []
      for pt in dst_pt:
        xvalues.append(pt[0])
        yvalues.append(pt[1])
      xavg = int(np.mean(xvalues))
      yavg = int(np.mean(yvalues))
      plateImg = frame[yavg-80:yavg+80, xavg-70:xavg+140]
      cv2.imshow("Plate", plateImg)
      cv2.waitKey(1)


      try:
        license_plate, ID = self.predict(plateImg)

        print("Before last predict: " + str(self.last_predicted_ID) + " ID: " + str(ID))
        if int(self.last_predicted_ID) == 7 and int(ID) == 5:
          ID = 8
          print("last predict: " + str(self.last_predicted_ID) + " ID: " + str(ID))

        ID = int(ID) - 1
        print(self.frequency)
        lic = list(license_plate)
        for i in range(len(lic)):
          if lic[i] in self.frequency[ID][i]:
            self.frequency[ID][i][lic[i]] += 1
          else:
            add = {lic[i]: 1}
            self.frequency[ID][i].update(add)

        result = []
        for i in range(4):
          max = 0
          key_max = ''
          for key in self.frequency[ID][i]:
            if self.frequency[ID][i][key] > max:
              max = self.frequency[ID][i][key]
              key_max = key
          result.append(key_max)
          
        ID += 1

        if self.guess_last_license_counter < 13:
          self.license_pub.publish(str('Sad_Asian_Kid,password,'+ str(ID) + ','+''.join(result)))

        if ID == 8 :
          self.last_predicted_ID = 7
          self.guess_last_license_counter += 1
        
        else:
          self.last_predicted_ID = ID
        
        if self.guess_last_license_counter > 12:
          self.license_pub.publish(str('Sad_Asian_Kid,password,-1,aa00'))

        print("Adjusted last ID: " + str(self.last_predicted_ID))

      except:
        print("error")
    

    # print(time.time()-start_time)

 




  def predict(self, image):
    IMG_SIZE = (30, 30, 1)
    row, column, ch = image.shape
    mask = self.banner_mask(image)
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
    cnts = list(cnts)
    cnt1, ind = self.find_max_area_contour(cnts, hierarchy)
    del cnts[ind]
    hierarchy = np.array([np.delete(hierarchy[0], ind, axis = 0)])
    cnt2, ind = self.find_max_area_contour(cnts, hierarchy)
    
    banner_contour = [cv2.boundingRect(cnt1), cv2.boundingRect(cnt2)]
    if banner_contour[0][1] > banner_contour[1][1]:
      banner_contour = [banner_contour[1], banner_contour[0]]
    
    crop = image[banner_contour[0][1]:banner_contour[1][1]+banner_contour[1][3], 
                  banner_contour[1][0] + 3:banner_contour[1][0]+banner_contour[1][2] - 3]
    mask_crop = self.mask_from_img(crop)
    
    mask_parking_ID = self.find_parking_ID(crop)

    # Find letters in Parking ID
    cnts_parking_id, hierarchy = cv2.findContours(mask_parking_ID, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
    cnts_parking_id = list(cnts_parking_id)

    cnt_parking_id1, ind = self.find_max_area_contour(cnts_parking_id, hierarchy)
    del cnts_parking_id[ind]
    hierarchy = np.array([np.delete(hierarchy[0], ind, axis = 0)])
    cnt_parking_id2, ind = self.find_max_area_contour(cnts_parking_id, hierarchy)
    box_parking_ID = [cv2.boundingRect(cnt_parking_id1), 
                      cv2.boundingRect(cnt_parking_id2)]

    if box_parking_ID[0][0] > box_parking_ID[1][0]:
      box_parking_ID = [box_parking_ID[1], box_parking_ID[0]]
    img_parking_ID = []

    for box in box_parking_ID:
      img_parking_ID.append(crop[box[1]:box[1]+box[3], box[0]:box[0]+box[2]])

    # Use kmeans to find the letters
    arr = []
    for i in range(mask_crop.shape[0]):
      for j in range(mask_crop.shape[1]):
        if mask_crop[i][j] == 255:
          arr.append([i, j])

    arr = np.float32(np.array(arr))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    if len(arr) < 4:
      return
    ret,label,center=cv2.kmeans(arr,4,None,criteria,20,cv2.KMEANS_RANDOM_CENTERS)

    datapoints = []
    boxes = []

    for i in range(4):
      temp = arr[label.ravel() == i]
      datapoints.append(temp)

    for points in datapoints:
      (xa,ya,wa,ha) = cv2.boundingRect(points)
      boxes.append(np.array((xa,ya,wa,ha)))
    
    # Sorting the letters from left to right
    for i in range(len(boxes)):
      low_val = column
      low_index = 0
      for j in range(i, len(boxes)):
        cent = boxes[j][1] + boxes[j][3]/2
        if cent < low_val:
          low_val = cent
          low_index = j
      
      temp = boxes[i]
      boxes[i] = boxes[low_index]
      boxes[low_index] = temp

    result = []
    confidence_pred = []
    imges = []
    pred = []
      
    for box in boxes:
      imges.append(crop[box[0]:box[0]+box[2], box[1]:box[1]+box[3]])
    
    # Predict Parking ID
    pred_img = self.find_parking_ID(img_parking_ID[1])
    pred_img = cv2.resize(pred_img, (IMG_SIZE[1], IMG_SIZE[0]))
    pred_img = np.around(pred_img/255)
    pred_img = np.expand_dims(pred_img, axis=0)
    predict = self.conv_model.predict(pred_img)[0]
    print(predict)
    result_parking_ID = np.argmax(predict)

    # Predict character
    for i in range(2):
      pred_img = self.mask_from_img(imges[i])

      b = []
      cnts, hierarchy = cv2.findContours(pred_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      max_area = 0
      ind = 0
      for i in range(len(cnts)):
        area = cv2.contourArea(cnts[i])
        if area > max_area:
          max_area = area
          ind = i
      b = cv2.boundingRect(cnts[ind])
        
      pred_img = pred_img[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]
      pred_img = cv2.resize(pred_img, (IMG_SIZE[1], IMG_SIZE[0]))
      pred_img = np.around(pred_img/255)
      pred_img = np.array(pred_img*255)

      pred.append(pred_img)
      pred_img = np.expand_dims(pred_img/255., axis=0)
      predict = self.char_model.predict(pred_img)[0]

      print(predict)
      highest_ind = np.argmax(predict)
      result.append(highest_ind)
    
    # Predict number
    for i in range(2,4):
      pred_img = self.mask_from_img(imges[i])

      b = []
      cnts, hierarchy = cv2.findContours(pred_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      max_area = 0
      ind = 0
      for i in range(len(cnts)):
        area = cv2.contourArea(cnts[i])
        if area > max_area:
          max_area = area
          ind = i
      b = cv2.boundingRect(cnts[ind])
        
      pred_img = pred_img[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]
      pred_img = cv2.resize(pred_img, (IMG_SIZE[1], IMG_SIZE[0]))
      pred_img = np.around(pred_img/255)
      pred_img = np.array(pred_img*255)

      pred.append(pred_img)
      pred_img = np.expand_dims(pred_img/255., axis=0)
      predict = self.conv_model.predict(pred_img)[0]
      print(predict)
      highest_ind = np.argmax(predict)

      result.append(highest_ind)

    for i in range(2):
      result[i] = chr(result[i] + 65)
    for i in range(2,4):
      result[i] = chr(result[i] + 48)
    result_parking_ID = chr(result_parking_ID + 48)

    print(result)
    print(result_parking_ID)
    return "".join(result), result_parking_ID, confidence_pred
  
  def mask_from_img(self, image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100,100,20])
    upper_blue = np.array([140,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = mask.astype(np.uint8)
    return mask
  
  def find_parking_ID(self, image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0,0,0])
    upper_black = np.array([0,0,60])
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

  def banner_mask(self, image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,90])
    upper = np.array([0,0,220])
    mask = cv2.inRange(hsv, lower, upper)

    mask = mask.astype(np.uint8)
    return mask

"""Main method."""
def main(args):
  print(args)
  rospy.init_node('controller', anonymous=True)
  ic = controller(args)
  
  license_pub = rospy.Publisher("/license_plate", String, queue_size=1)
  license_pub.publish(str('Sad_Asian_Kid,password,0,aa00'))
  #loop through calback() forever
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
