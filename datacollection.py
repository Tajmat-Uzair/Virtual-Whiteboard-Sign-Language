#script for data collection --- for hand sign language 

import cv2 
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20 #to give extra space while cropping
imgSize = 300 #to solve the dimension pblm while cropping
folder = "data/C"
counter = 0

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
            imgWhite[:,wGap:wCal+wGap]= imgResize #overlay img on white
       
        else :
            k = imgSize/w #stretching height k is constant
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:]= imgResize #overlay img on white




             

        cv2.imshow("Cropped Image", imgCrop)
        cv2.imshow("Image White", imgWhite)


    cv2.imshow("Image", img)

    #to save images
    key = cv2.waitKey(1) 
    if key == ord("s"):
        counter +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)

