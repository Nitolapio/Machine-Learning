import cv2

faceCascade = cv2.CascadeClassifier("/home/nitolapio/Escritorio/Programación/Machine Learning/OpenCV/Resources/opencv/data/haarcascades/haarcascade_frontalface_default.xml")
img = cv2.imread("/home/nitolapio/Escritorio/Programación/Machine Learning/OpenCV/Resources/peopleFaces.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgSmall = cv2.resize(img, (1080,720))

faces = faceCascade.detectMultiScale(imgSmall, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(imgSmall, (x,y), (x+w, y+h), (255,0,0), 2)

cv2.imshow("Deteccion de Caras", imgSmall)
cv2.waitKey(0)