""" import cv2 
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

def rec_sign_logic():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    offset = 20
    imgSize = 300
    counter = 0

    folder = "Dataset/V"

    while True:
       success,img = cap.read()
       img = cv2.flip(img, 1)
       hands,img = detector.findHands(img)

       if hands: 
            hand = hands[0] #because only one hand
            x,y,w,h = hand['bbox']

            imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255 #to create matrix
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset] #dimensions

            imgCropShape = imgCrop.shape
            aspectratio = h/w 

            if aspectratio >1:
              k = imgSize/h #stretching height k is constant
              wCal = math.ceil(k*w)
              imgResize = cv2.resize(imgCrop,(wCal,imgSize))
              imgResizeShape = imgResize.shape
              wGap = math.ceil((imgSize-wCal)/2)
              imgWhite[:,wGap:wCal+wGap]= imgResize

            #imgWhite[0:imgCropShape[0],[0]:imgCropShape[1]] = imgCrop
            cv2.imshow("ImageCrop",imgCrop)
            cv2.imshow("ImageWhite",imgCrop)


       cv2.imshow("Image", img)
       key = cv2.waitKey(1)
       if key == ord('s'):
           counter +=1
           cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
           print(counter)


rec_sign_logic()"""





#Data collection -- create new file to show modularity


import cv2 
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

def rec_sign_logic():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    offset = 20
    imgSize = 300
    counter = 0

    folder = "data/"
    

    while True:
       success,img = cap.read()
       img = cv2.flip(img, 1)
       hands,img = detector.findHands(img)

       if hands: 
            hand = hands[0] #because only one hand
            x,y,w,h = hand['bbox']

            imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255 #to create matrix
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset] #dimensions

            imgCropShape = imgCrop.shape
            aspectratio = h/w 

            if aspectratio >1:
              k = imgSize/h #stretching height k is constant
              wCal = math.ceil(k*w)
              imgResize = cv2.resize(imgCrop,(wCal,imgSize))
              imgResizeShape = imgResize.shape
              wGap = math.ceil((imgSize-wCal)/2)
              imgWhite[:,wGap:wCal+wGap]= imgResize

            #imgWhite[0:imgCropShape[0],[0]:imgCropShape[1]] = imgCrop
            cv2.imshow("ImageCrop",imgCrop)
            cv2.imshow("ImageWhite",imgCrop)


       cv2.imshow("Image", img)
       key = cv2.waitKey(1)
       if key == ord('s'):
           counter +=1
           cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
           print(counter)


rec_sign_logic() 
