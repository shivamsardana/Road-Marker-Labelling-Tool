import cv2
import numpy as np

DELAY_CAPTION = 1500;
DELAY_BLUR = 500;

img = cv2.imread('ts4.jpg')
i=19;
# Remember, bilateral is a bit slow, so as value go higher, it takes long time
bilateral_blur = cv2.bilateralFilter(img,i, i*2,i/2)
string = ''
cv2.putText(bilateral_blur,string,(20,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
cv2.imshow('Blur',bilateral_blur)
cv2.waitKey(DELAY_BLUR)


cv2.waitKey(0)
cv2.imwrite('blur4(k=19).jpg',bilateral_blur)


