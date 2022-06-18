#!/usr/bin/env python3

import sys
import os
import rospy
import math
import cv2
import glob
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
 
def callback(data):
  br = CvBridge()
  rospy.loginfo("receiving video frame")
   
  current_frame = br.imgmsg_to_cv2(data)

#   image = cv2.imread("/home/mai/catkin_ws/src/detect_face/data/face.jpg")

  ## loads the required XML classifiers
  file_path_1 = '/home/mai/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_default.xml'  #trọng số có sẵn trong thư viện của opencv
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  face_cascade.load(file_path_1)

  gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
  ## detects faces & returns positions of faces as Rect(x,y,w,h)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  
  #ve khung tron detect mat nguoi
  for (x,y,w,h) in faces:
    square =(w/3)**2 +(h/3)**2
    radius =int(math.sqrt(square))
    cv2.circle(current_frame,(int(x+w/2),int(y+h/2)),radius,(0,0,255),2)

#   cv2.imshow("my image", current_frame)
  cv2.imshow("my image", current_frame)

   
  cv2.waitKey(1)
      
def receive_message():
 
  rospy.init_node('video_sub_py', anonymous=True)
   
  rospy.Subscriber('video_frames', Image, callback)
 
  rospy.spin()
  cv2.destroyAllWindows()
  
if __name__ == '__main__':
  receive_message()