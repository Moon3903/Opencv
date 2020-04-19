import cv2
import numpy as np
import math

def nothing(x):
    pass


#mengambil gambar
img = cv2.imread('tes.png')

cv2.imshow('img',img)
cv2.namedWindow('image')

#trackbar untuk masking
cv2.createTrackbar('LH', 'image',0,255,nothing)
cv2.createTrackbar('LS', 'image',0,255,nothing)
cv2.createTrackbar('LV', 'image',0,255,nothing)
cv2.createTrackbar('UH', 'image',255,255,nothing)
cv2.createTrackbar('US', 'image',255,255,nothing)
cv2.createTrackbar('UV', 'image',255,255,nothing)
cv2.createTrackbar('DIL', 'image',0,5,nothing)
cv2.createTrackbar('ERO', 'image',0,5,nothing)

fontFace = cv2.FONT_HERSHEY_TRIPLEX
fontScale = 1

#menset masking untuk mendapatkan garis
#garis yang di dapat di tandai dengan warna biru
#garis di dapat dengan fungsi opencv houghlinep
while True:
    #masking
    lh=cv2.getTrackbarPos('LH','image')
    ls=cv2.getTrackbarPos('LS','image')
    lv=cv2.getTrackbarPos('LV','image')
    uh=cv2.getTrackbarPos('UH','image')
    us=cv2.getTrackbarPos('US','image')
    uv=cv2.getTrackbarPos('UV','image')
    dil=cv2.getTrackbarPos('DIL','image')
    ero=cv2.getTrackbarPos('ERO','image')

    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower=np.array([lh,ls,lv])
    upper=np.array([uh,us,uv])
     
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(img,img, mask= mask)

    #mengubah ke gray image dan di blur.
    kernel = np.ones((5,5), np.uint8) 

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    _,gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)

    #erotion dan dilation
    gray = cv2.dilate(gray, kernel, iterations=dil) 
    gray = cv2.erode(gray, kernel, iterations=ero)
    
    #mencari garis yang ada di gambar dengan houghlinep
    rho = 1
    theta = np.pi/180

    edge = cv2.Canny(gray,50,150)

    linep = cv2.HoughLinesP(edge, rho, theta, 15, np.array([]),50, 20)

    #menampilkan garis vertikal yang terdeteksi
    #garis yang di dapat di tandai dengan warna biru
    hasilp = img.copy()

    if not linep is None:
        for linepp in linep:
            for x1,y1,x2,y2 in linepp:
                #garis vertikal di tentukan dari sudut nya
                #garis dengan sudut antara 100 dan 70 
                #atau -70 dan -100 di anggap vertikal
                angle=np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
                if angle > -100:
                    if angle < -70:
                        cv2.line(hasilp,(x1,y1),(x2,y2),(255,0,0),5)
                if angle > 70:
                    if angle < 100:
                        cv2.line(hasilp,(x1,y1),(x2,y2),(255,0,0),5)

    #menampilkan hasil
    cv2.imshow("lines",img)
    cv2.imshow("lines",hasilp)

    cv2.imshow("img",img)
    cv2.imshow("sip",gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
