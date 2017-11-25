# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
from __future__ import print_function
import argparse
import datetime
import imutils
import time
import cv2
import keyboard

import io
import os.path
from google.cloud import vision
from google.cloud.vision import types
import cv2
import os

from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="blah.json"
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


#Number of frames to throw away
ramp_frames = 180

print ("start")

camera = cv2.VideoCapture(0)
time.sleep(0.25)

def opencv_boxit(imagePath):
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	print ("opencv_boxit: imagePath:", imagePath)
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=min(400, image.shape[1]))
	#orig = image.copy()

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)
	#print ("type(weights)", type(weights))

	# draw the original bounding boxes
	#for (x, y, w, h) in rects:
	#	cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	#print ("type(pick)", type(pick))

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# show some information on the number of bounding boxes
	filename = imagePath[imagePath.rfind("/") + 1:]
	#print("[INFO] {}: {} original boxes, {} after suppression".format(filename, len(rects), len(pick)))

	# show the output images
	#cv2.imshow("Before NMS", orig)
	#cv2.imshow("After NMS", image)
	cv2.imshow('filename:'+imagePath,image)
	print("cv2.imshow done, wait for keypress")
	cv2.waitKey(0)#units = milliseconds
	#print("cv2.waitKey key pressed")
	#time.sleep(1)#units = seconds
	#print("sleep done")
	cv2.destroyAllWindows()

def detect_face(face_file, max_results=4):
    """
    Uses the Vision API to detect faces in the given file.
    Args:
        face_file: A file-like object containing an image with faces.
    Returns:
        An array of Face objects with information about the picture.
    """
    client = vision.ImageAnnotatorClient()

    content = face_file.read()
    image = types.Image(content=content)

    return client.face_detection(image=image).face_annotations

def getInfo(file_name):
    #print ("in method getInfo, file_name:", file_name)
    vision_client = vision.Client()
    #file_name = 'detection1.jpg'

    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()
        image = vision_client.image(
            content=content, )

    labels = image.detect_labels()
    descriptions = []
    for label in labels:
        descriptions.append(label.description)
        #print("label.description:", label.description)
        #print("label.bounds:", label.bounds)
        #print("label.locale:", label.locale)
        #print("label.locations:", label.locations)
        #print("label.mid:", label.mid)
        #print("label.score:", label.score)
        #print("label.from_pb:", label.from_pb)
        #print("label._score:", label._score)
        #print("label._mid:", label._mid)
        #print()

    return descriptions


# Captures a single image from the camera and returns it in PIL format
def get_image():
	# read is the easiest way to get a full image out of a VideoCapture object.
	retval, im = camera.read()
	return im

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break

	#for i in range(0,ramp_frames):
		#temp = get_image()

	print('press q when ready for next image.')
	while True:#making a loop
	    try: #used try so that if user pressed other than the given key error will not be shown
	    	temp = get_image()
	    	if keyboard.is_pressed('q'):#if key 'q' is pressed
	    		print('You Pressed A Key!')
	    		break#finishing the loop
	    	else:
	    		pass
	    except:
	    	print("other key pressed.")
	    	break #if user pressed other than the given key the loop will break


	print("Taking image...")
	# Take the actual image we want to keep
	camera_capture = get_image()
	print ("type(camera_capture):", type(camera_capture))

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	# draw the text and timestamp on the frame
	#cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# show the frame and record if the user presses a key
	#cv2.imshow("Security Feed", frame)
	cv2.imshow("frame captured", camera_capture)
	#save camera_capture as file for ease
	file = "tempFile.jpg"
	cv2.imwrite(file, camera_capture)
	#now get info on image from cloud functions.
	print (getInfo(file))
    #print("detect_face")
	with open(file, 'rb') as image:
		faces = detect_face(image, 4)
		print('Found {} face{}'.format(len(faces), '' if len(faces) == 1 else 's'))
	#opencv_boxit(file)

	#cv2.imshow("Thresh", thresh)
	#cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
