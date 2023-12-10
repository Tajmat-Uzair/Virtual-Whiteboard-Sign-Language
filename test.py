""" import tkinter as tk
from hand_recognition_module import HandRecognitionModule

def recognize_signs():
    mod = HandRecognitionModule()
    mod.recognize_signs()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sign Language Recognition")

    button = tk.Button(root,text="Signs",command=recognize_signs)
    button.pack(pady=20)

    root.mainloop()

#gui for sign language """

import cv2 
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow
import time

def rec_sign_logic():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    offset = 20
    imgSize = 300
    counter = 0

    folder = "Dataset/V"
    classifier = Classifier("Model2/keras_model.h5","Model2/labels.txt")

    labels = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o",
              "p","q","r","s","t","u","v","w","x","y","z","End","0","1",
              "2","3","4","5","6","7","8","9"]



    while True:
       success,img = cap.read()
       imgOutput = img.copy()
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
              prediction, index = classifier.getPrediction(imgWhite)
              print(prediction,index)



            else :
              k = imgSize/w #stretching height k is constant
              hCal = math.ceil(k*h)
              imgResize = cv2.resize(imgCrop,(imgSize,hCal))
              imgResizeShape = imgResize.shape
              hGap = math.ceil((imgSize-hCal)/2)
              imgWhite[hGap:hCal+hGap,:]= imgResize #overlay img on white

            #imgWhite[0:imgCropShape[0],[0]:imgCropShape[1]] = imgCrop
            cv2.imshow("ImageCrop",imgCrop)
            cv2.imshow("ImageWhite",imgWhite)
            cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)



       cv2.imshow("Image", img)
       cv2.waitKey(1)

rec_sign_logic()