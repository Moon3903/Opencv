import numpy as np
import cv2

def nothing(x):
    pass

#mengambil frame dari kamera
cap = cv2.VideoCapture(0)

cv2.namedWindow('image')
cv2.resizeWindow('image', 600,350)

#trackbar untuk masking
cv2.createTrackbar('LH', 'image',0,255,nothing)
cv2.createTrackbar('LS', 'image',0,255,nothing)
cv2.createTrackbar('LV', 'image',0,255,nothing)
cv2.createTrackbar('UH', 'image',255,255,nothing)
cv2.createTrackbar('US', 'image',255,255,nothing)
cv2.createTrackbar('UV', 'image',255,255,nothing)
cv2.createTrackbar('DIL', 'image',1,5,nothing)
cv2.createTrackbar('ERO', 'image',1,5,nothing)
cv2.createTrackbar('APPROX', 'image',1,10,nothing)

fontFace = cv2.FONT_HERSHEY_TRIPLEX
fontScale = 1

#menset masking untuk mendapat bentuk
#bentuk ditentukan dari banyak sisi
#dengan approxpolydp
#bentuk dengan sudut lebih dari 6 di anggap bulat
while(True):
    # mengabil frame
    ret, img = cap.read()

    # Masking
    lh=cv2.getTrackbarPos('LH','image')
    ls=cv2.getTrackbarPos('LS','image')
    lv=cv2.getTrackbarPos('LV','image')
    uh=cv2.getTrackbarPos('UH','image')
    us=cv2.getTrackbarPos('US','image')
    uv=cv2.getTrackbarPos('UV','image')
    dil=cv2.getTrackbarPos('DIL','image')
    ero=cv2.getTrackbarPos('ERO','image')
    ap=cv2.getTrackbarPos('APPROX','image')
    
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower=np.array([lh,ls,lv])
    upper=np.array([uh,us,uv])

    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(img,img, mask= mask)

    #mengubah ke grayscale image
    edited = img.copy()
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray,(5,5),0)
    edited = cv2.GaussianBlur(edited,(5,5),0)
    
    _,gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)

    #dilation dan erotion
    kernel = np.ones((5,5), np.uint8) 
   
    gray = cv2.dilate(gray, kernel, iterations=dil) 
    gray = cv2.erode(gray, kernel, iterations=ero)

    #mengambil contour dari gambar yang sudah di mask dan di grayscale
    count,_=cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    hasil = img.copy()
    kotak = gray.copy()

    #menentukan bentuk dengan approxpolydp dengan menghitung sisi
    bulat = 0
    tiga = 0
    empat = 0
    for i in count :
        x,y,w,h = cv2.boundingRect(i)

        approx = cv2.approxPolyDP(i, (ap/100)*cv2.arcLength(i,True), True)

        cv2.drawContours(gray, [approx], 0, (122), 5)

        if(len(approx)>6):
            bulat = bulat+1
            cv2.rectangle(hasil,(x,y),(x+w,y+h),(0,0,255),2)
        elif(len(approx)==3):
            tiga = tiga+1
            cv2.rectangle(hasil,(x,y),(x+w,y+h),(0,255,0),2)
        elif(len(approx)==4):
            cv2.rectangle(hasil,(x,y),(x+w,y+h),(255,0,0),2)
            empat = empat+1
            
    #menampilkan hasil
    cv2.putText(hasil, "segi > 6 : "+str(bulat),(30,50),fontFace,fontScale,(0,0,255),1)
    cv2.putText(hasil, "segi 3 : "+str(tiga),(30,80),fontFace,fontScale,(0,255,0),1)
    cv2.putText(hasil, "segi 4 : "+str(empat),(30,120),fontFace,fontScale,(255,0,0),1)
    cv2.putText(hasil, "total : "+str(empat+tiga+bulat),(30,150),fontFace,fontScale,(0,0,0),1)
    cv2.imshow("edited",edited)
    cv2.imshow("img",hasil)
    cv2.imshow("sip",gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
