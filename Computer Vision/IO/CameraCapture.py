#! /usr/bin/python

import cv2
import numpy as np;

#Open Camera
cap = cv2.VideoCapture(0)

#Video Writer
wrt = cv2.VideoWriter()
ret,frame = cap.read()
fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
s = np.shape(frame)
wrt.open("./video.avi",fourcc,20.0,(640,480))

#Capture Image Frame From Camera
while(True):
    ret,frame = cap.read()
    
    #Write Video
    #frame = cv2.flip(frame,0)
    wrt.write(frame)
    
    #Show The Frame
    cv2.imshow("video",frame)
    
    #Key 'q' to Quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
wrt.release()
cap.release()
cv2.destroyAllWindows()
