#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 12:40:05 2020

@author: nitolapio
"""
from cv2 import cv2
import numpy as np

img = cv2.imread("/home/nitolapio/Escritorio/Programación/Machine Learning/OpenCV/Resources/lena.jpeg")
kernel = np.ones((2,2), np.uint8)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Para convertir la imagen a blanco y negro
imgBlur = cv2.GaussianBlur(imgGray, (7,7),0)    # BLur Gaussiano
imgCanny = cv2.Canny(img, 100, 100) # Para detectar los bordes
imgDialation = cv2.dilate(imgCanny, kernel, iterations=None)  # Para poder ver todos los bordes que antes no se podían ver
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

# Aquí ejecutamos
cv2.imshow("Blur Image", imgBlur)
cv2.imshow("Canny Image", imgCanny)
cv2.imshow("Dialation Image", imgDialation)
cv2.imshow("Eroded Image", imgEroded)
cv2.waitKey(0)







