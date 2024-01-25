import cv2
import numpy as np
cap=cv2.VideoCapture(0)
'''  A Haar Cascade is basically a classifier which is used to
 detect the object for which it has been trained for, from the source. 
'''
face=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eye=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")
while True:
    ret,frame=cap.read()
    #covert image into gray
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
   
    
    faces=face.detectMultiScale(gray,1.3,5)
    #r=cv2.rectangle(image, start_point, end_point, color, thickness)
    #top left and bottom right coordinates of the rectangle
    for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
            #We pass slice instead of index like this: [start:end].
            roi_gray=gray[y:y+h,x:x+w]
            roi_color=frame[y:y+h,x:x+w]

            eyes=eye.detectMultiScale(roi_gray,1.3,5)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
         
         
    cv2.imshow('frame',frame)
    if cv2.waitKey(1)==ord('q'):
           break
cap.release()
cv2.destroyAllWindows()