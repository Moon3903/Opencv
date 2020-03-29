import cv2
import numpy as np

def nothing(x):
    pass

img = cv2.imread('tes.png')

cv2.imshow('img',img)
cv2.namedWindow('image')

fontFace = cv2.FONT_HERSHEY_TRIPLEX
fontScale = 1

hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
lower=np.array([10,140,224])
upper=np.array([26,255,255])
     
mask = cv2.inRange(hsv, lower, upper)
res = cv2.bitwise_and(img,img, mask= mask)

kernel = np.ones((5,5), np.uint8) 

gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
_,gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)

rho = 1
theta = np.pi/180

edge = cv2.Canny(gray,50,150)

linep = cv2.HoughLinesP(edge, rho, theta, 15, np.array([]),50, 20)

hasilp = img.copy()

for linepp in linep:
    for x1,y1,x2,y2 in linepp:
        cv2.line(hasilp,(x1,y1),(x2,y2),(255,0,0),5)

cv2.imshow("lines",img)
cv2.imshow("lines",hasilp)

cv2.imshow("img",img)
cv2.imshow("sip",gray)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break