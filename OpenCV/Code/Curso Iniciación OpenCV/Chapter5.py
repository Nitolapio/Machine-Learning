import cv2
import numpy as np

img = cv2.imread("/home/nitolapio/Escritorio/Programación/Machine Learning/OpenCV/Resources/cards.jpg")


# Con esto convertimos un elemento torcido en un elemento plano

width, height = 250, 350
pts1 = np.float32([[204, 215], [317, 217], [193, 368], [309, 369]])  # Los puntos donde tenemos el cortante de la carta
pts2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])  #Definimos el punto que es cada coordenada anterior
matrix = cv2.getPerspectiveTransform(pts1, pts2) # Transformation matrix
imgOutput = cv2.warpPerspective(img, matrix, (width, height))


cv2.imshow("Cards", img)
cv2.imshow("Warp Card", imgOutput)
cv2.waitKey(0)