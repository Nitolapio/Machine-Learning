import cv2
import numpy as np


img = np.zeros((512,512, 3), np.uint8)   # Le ponemos 3 para darle la dimensionalidad del color
# img.shape to check the dimension of the img
# img[:] = 255,0,0  # De esta manera recorremos todo el array de los píxeles y la pintamos de esta manera

cv2.line(img, (0,0), (300,300), img.shape[1], 3) # Para dibujar una línea
cv2.rectangle(img, (0,0), (250,350), (0,0,255), cv2.FILLED) # Rectángulo
cv2.circle(img, (400,50), 30, (255,255,0), 5) #Circle
cv2.putText(img, "  OPENCV  ", (300, 200), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0,150,0), 1) # Para poner texto


cv2.imshow("Image", img)

cv2.waitKey(0)