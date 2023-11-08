import cv2 
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

class HandRecognitionModule:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier("Model/keras_model.h5","Model/labels.txt")
        self.offset = 20 #to give extra space while cropping
        self.imgSize = 300 #to solve the dimension pblm while cropping
        self.folder = "data/C"
        self.counter = 0
        self.labels = ["A","B","C"]
    
    def recognize_signs(self):
        while True:
          success,img = self.cap.read()
          img = cv2.flip(img, 1)
          imgOutput = img.copy()
          hands,img = self.detector.findHands(img)

          if hands: 
            hand = hands[0] #because only one hand
            x,y,w,h = hand['bbox']

            imgWhite = np.ones((self.imgSize,self.imgSize,3),np.uint8)*255 #to create matrix

            imgCrop = img[y-self.offset:y+h+self.offset, x-self.offset:x+w+self.offset] #dimensions
            imgCropShape = imgCrop.shape
            aspectratio = h/w 
            if aspectratio>1:
                k = self.imgSize/h #stretching height k is constant
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop,(wCal,self.imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((self.imgSize-wCal)/2)
                imgWhite[:,wGap:wCal+wGap]= imgResize #overlay img on white
                prediction, index = self.classifier.getPrediction(imgWhite)
                print(prediction,index)                 
        
            else :
                 k = self.imgSize/w #stretching height k is constant
                 hCal = math.ceil(k*h)
                 imgResize = cv2.resize(imgCrop,(self.imgSize,hCal))
                 imgResizeShape = imgResize.shape
                 hGap = math.ceil((self.imgSize-hCal)/2)
                 imgWhite[hGap:hCal+hGap,:]= imgResize #overlay img on white
                 prediction, index = self.classifier.getPrediction(imgWhite,draw=False)

          cv2.rectangle(imgOutput,(x-self.offset,y-self.offset-50),(x-self.offset+90,y-self.offset-50+50),(255,0,255),cv2.FILLED)
          cv2.putText(imgOutput,self.labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)
          cv2.rectangle(imgOutput,(x-self.offset,y-self.offset),(x+w+self.offset,y+h+self.offset),(255,0,255),4)



             

          cv2.imshow("Cropped Image", imgCrop)
          cv2.imshow("Image White", imgWhite)


          cv2.imshow("Image", imgOutput)

          cv2.waitKey(1) 


        #add the main method yet