import cv2 
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

class SignRecognizer:
    def __init__(self):
        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier("Model/keras_model.h5","Model/labels.txt")
        self.offset = 20
        self.imgSize = 300
        self.folder = "data/C"
        self.labels = ["A","B","C"]

    def recognize_sign(self, img):
        imgOutput = img.copy()
        hands, img = self.detector.findHands(img)

        if hands: 
            hand = hands[0] 
            x,y,w,h = hand['bbox']

            imgWhite = np.ones((self.imgSize,self.imgSize,3),np.uint8)*255 

            imgCrop = img[y-self.offset:y+h+self.offset, x-self.offset:x+w+self.offset] 
            imgCropShape = imgCrop.shape
            
            aspectratio = h/w 

            if aspectratio > 1:
                k = self.imgSize/h 
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop,(wCal,self.imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((self.imgSize-wCal)/2)
                imgWhite[:,wGap:wCal+wGap]= imgResize 
                prediction, index = self.classifier.getPrediction(imgWhite)
                print(prediction,index)
            
            else :
                k = self.imgSize/w 
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop,(self.imgSize,hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((self.imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap,:]= imgResize 
                prediction, index = self.classifier.getPrediction(imgWhite,draw=False)

            cv2.rectangle(imgOutput,(x-self.offset,y-self.offset-50),(x-self.offset+90,y-self.offset-50+50),(255,0,255),cv2.FILLED)
            cv2.putText(imgOutput,self.labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)
            cv2.rectangle(imgOutput,(x-self.offset,y-self.offset),(x+w+self.offset,y+h+self.offset),(255,0,255),4)

            cv2.imshow("Cropped Image", imgCrop)
            cv2.imshow("Image White", imgWhite)

        return imgOutput