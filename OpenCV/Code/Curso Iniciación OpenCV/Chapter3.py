import cv2

img = cv2.imread("/home/nitolapio/Escritorio/Programación/Machine Learning/OpenCV/Resources/lambo.jpeg")
print(img.shape)

# In this way we resize the size of the image. Also we can increase it
imgResize = cv2.resize(img, (100,200))

# We can crop the image this way, as an array of numbers
imgCropped = img[0:100, 20: 150]

cv2.imshow("Lambo", img)
cv2.imshow("Lambo resized", imgResize)
cv2.imshow("Cropped Image", imgCropped)
cv2.waitKey(0)