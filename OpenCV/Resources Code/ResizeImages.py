import cv2

### This is just an example

src1 = cv2.imread('data/src/lena.jpg')
src2 = cv2.imread('data/src/rocket.jpg')

src2 = cv2.resize(src2, src1.shape[1::-1])