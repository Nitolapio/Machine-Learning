import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture("/home/nitolapio/Desktop/Programación/Machine-Learning/OpenCV/Resources/test2.mp4")

# Con los siguientes dos métodos vamos a optimizar la detección de lineas (Crear un average de las líneas detectadas)
def makeCoordinates(img, lineParameters):
    slope, intercept = lineParameters   # y = mx + b     x = (y-b)/m
    y1 = img.shape[0]
    y2 = int(y1*(3/5))   # The lines are starting from the bottom (y1 = 700), and go up to 3/5 of the coordinates (y2 =~ 400)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1, x2, y2])

def averageSlopeIntercept(img, lines):
    leftFit = []
    rightFit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)  # No lo he entendido muy bien
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            leftFit.append((slope, intercept))
        else:
            rightFit.append((slope, intercept))
    leftFitAverage = np.average(leftFit, axis = 0)
    rightFitAverage = np.average(rightFit, axis = 0)
    leftLine = makeCoordinates(img, leftFitAverage)
    rightLine = makeCoordinates(img, rightFitAverage) 
    return np.array([leftLine, rightLine])

# Función para convertir una imagen a canny
def canny(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Procesar 1 dimensión de una imagen en GrayScale es más rápido que procesar las 3 dimensiones de color
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 0)        #Reduce noise and smoothen image because noise can make the canny to detect false edges. Anyways, the Canny does this automatically
    imgCanny = cv2.Canny(imgBlur, 50,150)   # Esto detecta la diferencia de brightness. Las diferencias más notables, las detecta como un edge
    return imgCanny

def regionOfInterest(img):
    height = img.shape[0]  # La primera propiedad de la imagen: el bottom del eje y (hacia abajo)  (approx. 700)
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)] # Tiene que ser más que un polígono
        ])  #creamos un triángulo dentro del que queremos detectar el carril
    mask = np.zeros_like(img)  #Arrays con el mismo número de píxeles
    cv2.fillPoly(mask, polygons, (255,255,255))  # Llenamos una imagen con el triángulo blanco (dentro del mask). Sólo puede llenar una imagen con varios polígonos; si lo hacemos solo con uno, nos dará un error
    maskedImage = cv2.bitwise_and(img, mask)  # Cogemos los bits de ambas imágenes. Blanco = 1, negro = 0 . Mask tiene un 1, y va a hacer un AND con los bits blancos de la imagen en sí. De esta manera, sólo se muestran los bits blancos de la región de interés
    return maskedImage

# Para unir todas las imágenes
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


def display_lines(img, lines):
    line_image = np.zeros_like(img)  #Displays a black image
    if lines is not None:   # If the lines image is not empty we loop through it
         for line in lines:
            x1, y1, x2, y2 = line.reshape(4)  # each line is a 2D array containing our line coordinates [[x1, y, x2, y2]]. # We are gonna reshape each 2D line in a 1D shape line
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)  # We drew the lines into the blank image
    return line_image                                  




while (cap.isOpened()):
    ret, frame = cap.read()
    imgCanny = canny(frame)
    croppedImage = regionOfInterest(imgCanny) #Hemos usad la función de region of interest
    lines = cv2.HoughLinesP(croppedImage, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)   # Usamos el método de Hough. 2 pixels, 1 degree precision (pi/180). El threshhold indica el número de intersecciones que necesitamos para considerar un píxel una línea
    averagedLines = averageSlopeIntercept(frame, lines)   # this is for optimizing line detection
    lineImageOptimized = display_lines(frame, averagedLines)  
    lineImage = display_lines(frame, lines)  
    comboImage = cv2.addWeighted(frame, 0.8, lineImage, 1, 1)  # Con esto unimos las líneas a la imagen
    comboImageOptimized = cv2.addWeighted(frame, 0.8, lineImageOptimized, 1, 1)
    stack = stackImages(0.28, [[imgCanny, regionOfInterest(imgCanny), lineImage, comboImage], 
                            [imgCanny, regionOfInterest(imgCanny), lineImageOptimized, comboImageOptimized]])

    if ret:
        cv2.imshow('frame',stack)
    key = cv2.waitKey(25)  #it can also be another waitkey. It returns a 32 bit integer value, which we can compare to a numeric coding that we get with 'ord()'
    if key == ord('n') or key == ord('p'):
        break

