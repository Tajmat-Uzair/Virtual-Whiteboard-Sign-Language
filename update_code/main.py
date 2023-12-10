from hand_detection_module import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import cv2 

class MainApp:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.classifier = Classifier("Model/keras_model.h5","Model/labels.txt")
        self.imgSize = 300
        self.offset = 20
        self.folder = "data/C"
        self.labels = ["A", "B"]

    def rec_signs(self, img):
        hands = self.hand_detector.findhands(img)

        if hands:
            hand = hands[0]  # assuming only one hand is detected
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255

            imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

            aspect_ratio = h / w

            if aspect_ratio > 1:
                k = self.imgSize / h
                w_cal = math.ceil(k * w)
                img_resize = cv2.resize(imgCrop, (w_cal, self.imgSize))
                w_gap = math.ceil((self.imgSize - w_cal) / 2)
                imgWhite[:, w_gap:w_cal + w_gap] = img_resize
                prediction, index = self.classifier.getPrediction(imgWhite)
                print(prediction, index)
            else:
                k = self.imgSize / w
                h_cal = math.ceil(k * h)
                img_resize = cv2.resize(imgCrop, (self.imgSize, h_cal))
                h_gap = math.ceil((self.imgSize - h_cal) / 2)
                imgWhite[h_gap:h_cal + h_gap, :] = img_resize
                prediction, index = self.classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(img, (x - self.offset, y - self.offset - 50), (x - self.offset + 90, y - self.offset - 50 + 50),
                          (255, 0, 255), cv2.FILLED)
            cv2.putText(img, self.labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(img, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset),
                          (255, 0, 255), 4)

            cv2.imshow("Cropped Image", imgCrop)
            cv2.imshow("Image White", imgWhite)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
       
def hands():
    hand_detector = HandDetector()
    hand_detector.main()


if __name__=="__main__":
   ob1 =  hands()
   ob2 = MainApp()

