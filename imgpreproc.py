import cv2
import numpy as np 
import matplotlib.pyplot as plt 

cap = cv2.VideoCapture(1)
fgbg  = cv2.createBackgroundSubtractorMOG2()

while True:
	ret, frame = cap.read()

	#converting to gray scale first to make image processing easier
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = fgbg.apply(gray)

	width = 100
	height = 100
	dim = (width, height)
	# resize image using fixed width and height number of pixels
	#this is to ensure uniformity in the dataset later on
	resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)	

	#to make it easier to detect edges we first apply blur
	#5,5 is the dimensions of the filter convolving
	#0 is standard deviation in each direction (unnecessary for most apps.)
	#basically smoothening the image
	blur = cv2.GaussianBlur(gray,(5,5),0)


	#now to use thresholding to seperate light and dark parts of the image
	#using adaptive threshold which calculates different thresholds
	#for different regions(better than a single global threshold)
	#the threshold value is a gaussian-weighted sum of the 
	#neighbourhood values minus the constant C.
	th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	#parameters : 255 is max value of pixel, 11 is the number of surrounding pixels to be consulted
	#2 is constant to be subtracted from gaussian weighted sum


	edges = cv2.Canny(frame, 50,50)

	cv2.imshow('frame', frame)
	cv2.imshow('gray', edges)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()