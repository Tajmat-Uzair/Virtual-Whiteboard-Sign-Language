import cv2 
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

#HAND SIGN RECOGNITION

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20 #to give extra space while cropping
imgSize = 300 #to solve the dimension pblm while cropping
folder = "data/C"
counter = 0

labels = ["A","B","C"]

while True:
    success,img = cap.read()
    img = cv2.flip(img, 1)
    imgOutput = img.copy()
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
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction,index)
        
        else :
            k = imgSize/w #stretching height k is constant
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:]= imgResize #overlay img on white
            prediction, index = classifier.getPrediction(imgWhite,draw=False)

        cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+90,y-offset-50+50),(255,0,255),cv2.FILLED)
        cv2.putText(imgOutput,labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)



             

        cv2.imshow("Cropped Image", imgCrop)
        cv2.imshow("Image White", imgWhite)


    cv2.imshow("Image", imgOutput)

    cv2.waitKey(1) 
 

