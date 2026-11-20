import cv2
import os
import numpy as np
from typing import Optional, List

def gaussian_blur(img: np.array, sigma: float, filter_shape: Optional[List] = None, verbose: bool = False) -> np.array:
    if filter_shape == None:
        n = 2 * int(np.ceil(3 * sigma)) + 1
        filter_l = [n,n]
    else:
        filter_l = filter_shape

    k = filter_l[0] // 2
    y, x = np.mgrid[-k:k + 1, -k:k + 1]
    
    formula = (1/(2*np.pi*sigma**2))*np.exp(-((x**2+y**2)/(2*sigma**2)))
    gaussian_filter = formula/formula.sum()
    gb_img = cv2.filter2D(img,ddepth=-1,kernel=gaussian_filter)
    
    return gaussian_filter, gb_img.astype(np.uint8)

def custom_dilate(img):
    padded = np.pad(img,1)
    width = img.shape[0]
    height = img.shape[1]
    dilated = np.zeros(padded.shape,dtype="uint8")
    
    
    for j in range(1,height+1):
        for i in range(1,width+1):
            dilated[i][j]=max([padded[i][j],padded[i][j-1],padded[i][j+1],padded[i+1][j],padded[i-1][j]])
            
    dilated = dilated[1:width+1, 1:height+1]
    
    return dilated

def custom_erode(img):
        padded = np.pad(img,1)
        width = img.shape[0]
        height = img.shape[1]
        eroded = np.zeros(padded.shape,dtype="uint8")
        
        for j in range(1,height+1):
            for i in range(1,width+1):
                eroded[i][j]=min([padded[i][j],padded[i-1][j],padded[i][j+1],padded[i+1][j],padded[i][j-1]])
                
        eroded = eroded[1:width+1, 1:height+1]
        return eroded

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) 
if not cap.isOpened():
    print('Error: Could not open the video file')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  #Hay menos frames
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolución: {frame_width}x{frame_height}")

# Rojo: dos rangos de tono
lower1 = np.array([0, 120, 70])
upper1 = np.array([10, 255, 255])
lower2 = np.array([170, 120, 70])
upper2 = np.array([180, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    _, blurred = gaussian_blur(frame, sigma=2, filter_shape=[11, 11])
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)
    mask = custom_dilate(custom_erode(mask))   # apertura
    mask = custom_erode(custom_dilate(mask))   # cierre
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        if r > 8:
            cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)

    cv2.imshow("Camara", frame)
    cv2.imshow("Mascara", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):   # q para salir
        break

cap.release()
cv2.destroyAllWindows()