#import modules here
import cv2 
import mediapipe
import handtracking_module as ht
import time

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)

detector = ht.handDetector()

while True:
      success, img = cap.read()
      img = detector.findHands(img)
      lmList = detector.findPosition(img)
      if lmList is not None and len(lmList)!=0:
         print(lmList[4])


      cTime = time.time()
      fps = 1 / (cTime - pTime)
      pTime = cTime

      cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3,
                (255, 0, 255), 3)

      cv2.imshow("Image", img)
      cv2.waitKey(1)