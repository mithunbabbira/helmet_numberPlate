import cv2
import numpy as np
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("start")


crop = []
crop = np.array(crop)




cam = cv2.VideoCapture(0)
while True:
    
    
    ret , img = cam.read(0)   

    face_rects = face_cascade.detectMultiScale(img,scaleFactor=1.2,minNeighbors= 10)

    detected = len(face_rects)# total no of face detect




    def extract_plate(face):
    	plate_img = face.copy()

    	plate_cascade = cv2.CascadeClassifier('./indian_license_plate.xml')
    	plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.3, minNeighbors = 7)

    	print("number plate = "+str(len(plate_rect)))

    	for (x,y,w,h) in plate_rect:
    		a,b = (int(0.02*face.shape[0]), int(0.025*face.shape[1]))
    		plate = plate_img[y+a:y+h-a, x+b:x+w-b, :]
    		cv2.rectangle(plate_img, (x,y), (x+w, y+h), (51,51,255), 3)
    		cv2.imshow("Number plate",plate_img)



    
    for (x,y,w,h) in face_rects:
    	cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),10)
    	a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1])) #parameter tuning


    	j = 70


    	p = y + a-j     # top - = up  , + = down
    	q = y+h-a+j + 300 # bottom (- = up , + = down)
    	r = x+b-j  # left (- = right , + = left)
    	s = x+w-b+j  # right  (- = left , + = right )


    	crop = img[p:q,r:s]
    
    cv2.imshow('original',img)

    if ( detected is not 0):
    	cv2.imshow('video face detect',crop)
    	print ("total face = " + str(detected))
    	extract_plate(crop)
    else:
    	print("face not  detected")
   
    
    k = cv2.waitKey(1)
    if k == 27:
        break
        
cam.release()
cv2.destroyAllWindows()