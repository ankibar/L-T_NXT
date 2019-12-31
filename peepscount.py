# import the necessary packages
import numpy as np 
import cv2
import glob
import os
from PIL import Image

from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import FPS
import argparse
import imutils
import time
#logic : use the background subtracted image
#count contours 
#number of contours = number of people
#tracking same code as people_counter.py

def producefg(allfiles):
	# Access all PNG files in directory
	#allfiles=r'/home/ankit/Desktop/Internship/SegNet-Tutorial-master/CamVid/train_bw_resized/'
	

	#allfiles=r'/home/ankit/Desktop/Internship/trialim/'
	

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
	    imarr_subracted = cv2.subtract(input_image,arr)
	    p.append(imarr_subracted)
	#len(p)	
	return p
	#out2=Image.fromarray(p[55],mode="RGB")
	#out2.show()

k = list()
k = producefg(r'D:\Internship\trialim')

size = (512,424)
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(k)):
	out.write(k[i])

out.release()

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
#src='D:\Internship\final-20191224T050105Z-001\final\project.avi'
stream = cv2.VideoCapture('project.avi')
#vs = VideoStream(src='project.avi').start()
fps = FPS().start()

#time.sleep(2.0)
#print(vs.isOpened())
print("is it reaching")
# loop over the frames from the video stream
while True:
	# read the next frame from the video stream and resize it
	#frame = vs.read()
	(grabbed, frame) = stream.read()
	if not grabbed:
		break
	frame = imutils.resize(frame, width=400)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = np.dstack([frame, frame, frame])

	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
	blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
		(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	rects = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# filter out weak detections by ensuring the predicted
		# probability is greater than a minimum threshold
		if detections[0, 0, i, 2] > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
			(startX, startY, endX, endY) = box.astype("int")
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)

	# update our centroid tracker using the computed set of bounding
	# box rectangles
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# show the output frame
	#print(frame)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(190) & 0xFF
	fps.update()

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
fps.stop()
stream.release()
cv2.destroyAllWindows()
#vs.stop()




'''
python peepscount.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel
'''