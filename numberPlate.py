import cv2
import numpy as np
import matplotlib.pyplot as plt



#img = cv2.imread('car.JPEG')

plate_img1 = cv2.VideoCapture(0)
#Loads the data required for detecting the license plates from cascade classifier.
plate_cascade = cv2.CascadeClassifier('indian_license_plate.xml')


print("dg")

while True:

	ret ,img = plate_img1.read(0) 

	plate_img = img.copy()
	
	plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.3, minNeighbors = 7)

	for (x,y,w,h) in plate_rect:
		a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1])) #parameter tuning
		plate = plate_img[y+a:y+h-a, x+b:x+w-b, :]
		# finally representing the detected contours by drawing rectangles around the edges.
		cv2.rectangle(plate_img, (x,y), (x+w, y+h), (51,51,255), 3)
       
	cv2.imshow('video face detect',plate_img) 
	k = cv2.waitKey(1)
	if k == 27:
		break


	


cv2.destroyAllWindows()