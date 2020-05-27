import cv2
import numpy as np
import matplotlib.pyplot as plt



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("dgsg")



cap = cv2.VideoCapture(0)
while True:
    
    
    ret , frame = cap.read(0)   

    face_rects = face_cascade.detectMultiScale(frame,scaleFactor=1.2,minNeighbors= 5)
    
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),10)
    
    cv2.imshow('video face detect',frame)    
    
    k = cv2.waitKey(1)
    if k == 27:
        break
        
cap.release()
cv2.destroyAllWindows()