###modified virtual whiteboard

import cv2
import numpy as np
import os
import mod_handtrackingmodule as htm
from hand_recognition_module import HandRecognitionModule
import hand_recognition_module as hrm


# locating headers for ui
folderPath = "Headers"
myList = os.listdir(folderPath)
print(myList)
overLayList = []


# 1. import those images
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overLayList.append(image)  # store images

print(len(overLayList))  # re-check

header = overLayList[0]  # default header for webcam
drawColor = (255, 255, 255)
brushthickness = 15
eraserthickness = 60   

detector = htm.handDetector(detectionCon=0.85)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)  # to draw on the canvas
ob = hrm.HandRecognitionModule()

def process_frame(img):
    global imgCanvas

    # 2. Find Hand Landmarks
    img = detector.findHands(img)

    lmList = detector.findPosition(img, draw=False)  # get landmarks

    if lmList is not None:
        if len(lmList) != 0:
            print(lmList)

        # tip of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Checking which fingers are up
        fingers = detector.fingersUp()

        # 4. Selection Mode- 2 fingers up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")

            # condition check for click
            if y1 < 125:  # in header?
                if 180 < x1 < 400:
                    header = overLayList[0]  # pink ig
                    drawColor = (147, 20, 255)
                elif 405 < x1 < 580:
                    header = overLayList[1]  # blue ig
                    drawColor = (255, 0, 0)
                elif 590 < x1 < 820:
                    header = overLayList[3]  # green
                    drawColor = (0, 255, 0)
                elif 850 < x1 < 1000:
                    header = overLayList[4]  # sign
                    ob.recognize_signs()
                    # sign.recognize_signs()
                elif 1030 < x1 < 1069:
                    header = overLayList[2]  # eraser
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, 2)

        # 5. Drawing Mode - Index finger up
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):  # erasing
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserthickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserthickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushthickness)  # for drawing
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushthickness)  # on canvas

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)  # will draw in black
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)  # to binary image
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = header  # overlay the default header

    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0) #to blend and draw on original image
    return img

# Assuming you have a list of frames named hand_tracking_frames
for frame in hand_tracking_frames:
    processed_frame = process_frame(frame)
    cv2.imshow("Virtual Whiteboard", processed_frame)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
