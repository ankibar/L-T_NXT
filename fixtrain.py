#Utility function

import os
import cv2
from os import listdir,makedirs
from os.path import isfile,join

path = r'/home/ankit/Desktop/Internship/SegNet-Tutorial-master/CamVid/train_bw_resized' # Source Folder
dstpath = r'/home/ankit/Desktop/Internship/SegNet-Tutorial-master/CamVid/train_bw' # Destination Folder

try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in same folder")

# Folder won't used
files = [f for f in listdir(path) if isfile(join(path,f))] 

for image in files:
    img = cv2.imread(os.path.join(path,image))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    width = 200
    height = 200
    dim = (width, height)
	# resize image
    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    dstPath = join(dstpath,image)
    cv2.imwrite(dstPath,resized,0)	
