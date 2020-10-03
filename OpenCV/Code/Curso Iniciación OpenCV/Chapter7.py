import cv2
import numpy as np

# Creamos una función vacía porque necesitamos pasarla a los trackbars
def empty():
    pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


img = cv2.imread("/home/nitolapio/Escritorio/Programación/Machine Learning/OpenCV/Resources/lambo.jpeg")

# Aquí creamos todos los trackbars y les damos los valores
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 15, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 47, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 84, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True: 

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    # Obtenemos los valores seleccionados en los trackbars
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    sat_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    sat_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    val_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    val_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    
    print(h_min, h_max, sat_min, sat_max, val_min, val_max)

    # Con esto creamos una nueva máscara de la foto
    lower = np.array([h_min, sat_min, val_min])
    upper = np.array([h_max, sat_max, val_max])
    mask = cv2.inRange(imgHSV, lower, upper)  # Nos va a dar la imagen con el filtro de los colores que nosotros elijamos

    imgResult = cv2.bitwise_and(img, img, mask = mask)  #Unimos dos imágenes para crear una imágen

    # Usamos la función para poner todas las imágenes creadas
    imgStacked = stackImages(1, [[img, imgHSV,], [mask, imgResult]])

    
    #cv2.imshow("Lambo", img)
    #cv2.imshow("HSV", imgHSV)
    #cv2.imshow("Masked Image", mask)
    #cv2.imshow("Result", imgResult)
    
    cv2.imshow("All Images", imgStacked)

    cv2.waitKey(1)
    