import cv2
import numpy as np

#Importamos el código que usamos en cp1 para usar la webcam
cap = cv2.VideoCapture(0)  # Con el 0 indicamos que queremos la webcam
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100) # For brightness

myColors = [[0, 179, 103, 255, 255, 255], 
            [0, 131, 105, 255, 231, 255],
            [0,179,0,0,231,255]]    # Red, blue, white

def findColor(img, myColors):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(myColors[0][0:3])
    upper = np.array(myColors[0][3:6])
    mask = cv2.inRange(img, lower, upper) 
    cv2.imshow("mask", mask)  # This is for testing

while (cap.isOpened()):
    sucess, img = cap.read()
    findColor(img, myColors)
    if sucess:
        cv2.imshow('Cam',img)
    key = cv2.waitKey(25)
    if key == ord('n') or key == ord('p'):
        break

