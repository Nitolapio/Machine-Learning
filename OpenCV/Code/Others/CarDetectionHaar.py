import cv2

### This method is not quite good; it's old

# cap = cv2.VideoCapture("/home/nitolapio/Escritorio/Programación/Machine Learning/OpenCV/Resources/Vehicle-And-Pedestrian-Detection-Using-Haar-Cascades/Main Project/Main Project/Car Detection/video1.avi")
carCascade = cv2.CascadeClassifier("/home/nitolapio/Desktop/Programación/Machine Learning/OpenCV/Resources/vehicle_detection_haarcascades/cars.xml")
cap = cv2.VideoCapture("/home/nitolapio/Desktop/Programación/Machine-Learning/OpenCV/Resources/test_video.mp4")


while True:
    ret, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = carCascade.detectMultiScale(imgGray, 1.1, 1)

    for(x,y,w,h) in cars:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)


    if ret:
        cv2.imshow("Video", img)
    key = cv2.waitKey(23)
    if key == ord("q"):
        break