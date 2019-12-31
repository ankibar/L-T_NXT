# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import os
import numpy as np
from PIL import Image

def producefg(allfiles):
	# Access all PNG files in directory
	#allfiles=r'/home/ankit/Desktop/Internship/SegNet-Tutorial-master/CamVid/train_bw_resized/'
	

	allfiles=r'D:\Internship\trialim'
	

	images = list()
	for filename in os.listdir(allfiles):
	    img = cv2.imread(os.path.join(allfiles,filename), 1)
	    if img is not None:
	        images.append(img)    
	#print(images[1].shape)


	# Assuming all images are the same size, get dimensions of first image
	w=512
	h=424
	N=len(images)
	print(N)
	# Create a numpy array of floats to store the average (assume RGB images)
	arr=np.zeros((h,w,3),np.float)

	# Build up average pixel intensities, casting each image as an array of floats
	for im in images:
	    #print(im)
	    im=cv2.cvtColor(im, cv2. COLOR_BGR2RGB)
	    #imarr=numpy.array(Image.open(im),dtype=numpy.float)
	    imarr=np.array(Image.fromarray(im),dtype=np.float)
	    #print(imarr)
	    #print(len(imarr))
	    arr=arr+imarr/N

	# Round values in array and cast as 8-bit integer
	arr=np.array(np.round(arr),dtype=np.uint8)
	#arr = Image.fromarray(arr)
	#print(arr)
	#Generate, save and preview final image
	'''
	out=Image.fromarray(arr,mode="RGB")
	out.save("Average.png")
	out.show()
	'''

	#print(type(arr))
	#print(type(images[1]))
	p = []
	for im in images:
	    input_image=np.asarray(im, np.uint8)
	    #print(ty)
	    #imarr_subracted = cv2.subtract(input_image,arr)
	    p.append(input_image)
	#len(p)	
	return p
	#out2=Image.fromarray(p[55],mode="RGB")
	#out2.show()

k = list()
k = producefg(r'D:\Internship\trialim')

size = (512,424)
out = cv2.VideoWriter('project1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(k)):
	out.write(k[i])

out.release()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1]
	text = "Unoccupied"

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Occupied"

	# draw the text and timestamp on the frame
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()