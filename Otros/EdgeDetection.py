from cv2 import cv2
import numpy as np

img = cv2.imread(r"C:\Users\alexa\Desktop\Programación\Machine-Learning\OpenCV\Resources\lena.jpeg")

img2 = np.array(img)

cv2.utils.dumpInputArray(img2)

cv2.imshow("Lena image", img2)

cv2.waitKey(0)
