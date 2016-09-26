import cv2
import argparse
import datetime
import imutils
import time
import sys
import numpy as np

# Detect new object entering current frame
def detect_change():
	# cv2.accumulateWeighted(gray, prev_frame, 0.5)
	delta_frame = cv2.absdiff(gray, cv2.convertScaleAbs(prev_frame))
	thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if len(cnts) == 0:
		return []
	else:
		return cnts[0]

ap = argparse.ArgumentParser()
ap.add_argument('-a', '--min-area', type=int, default=500)
args = vars(ap.parse_args())

cap = cv2.VideoCapture(0)

# Make sure the camera is working
if (cap.isOpened()):
	print("[INFO] Camera OK")
else:
	print("[INFO] Opening camera...")
	cap.open()
	time.sleep(0.25)
	if (cap.isOpened()):
		print("[INFO] Camera OK")
	else:
		sys.exit("[ERROR] Camera Problem Detected!")

# Initialize previous frame to Null
prev_frame = None

# Initialize detected flag to zero
detected_flag = 0

# Initialize object's bounding rect to zero 
x_detect = 0
y_detect = 0
w_detect = 0
h_detect = 0

# Variables to save temporarily detected object's (x, y)
prev_x = 0
prev_y = 0

text = 'Nothing detected'
while True:
	ret, cur_frame = cap.read()

	if not ret:
		break

	# Resize read frame to width 500, then convert to gray scale
	cur_frame = imutils.resize(cur_frame, width=500)
	gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

	# Apply Gaussian blur and median blur for noise reduction
	gray = cv2.GaussianBlur(gray, (7, 7), 0)
	gray = cv2.medianBlur(gray, 7)

	# Make a copy of current frame for display output from detect_change method
	delta_frame = cur_frame.copy()

	# Initialize previous frame with first frame from camera, using it as background
	if prev_frame is None:
		prev_frame = gray.copy().astype('float')
		continue

	# Threshold then dilate grayscaled frame for better contour result
	cur_thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
	cur_thresh = cv2.dilate(cur_thresh, None, iterations=4)
	cv2.imshow('Thresh', cur_thresh) # Show the threshold image for debugging purpose

	# If new object is not detected, execute detect_change method
	if detected_flag == 0:
		cnt_change = detect_change()
		prev_x = x_detect
		prev_y = y_detect

	# If new object is detected, save its bounding rectangle
	if (len(cnt_change) != 0):
		x_detect, y_detect, w_detect, h_detect = cv2.boundingRect(cnt_change)
		ratio_detect = w_detect / float(h_detect)
	
	# Draw a rectangle around new detected object
	cv2.rectangle(delta_frame, (x_detect, y_detect), (x_detect + w_detect, y_detect + h_detect), (0, 255, 0), 0)

	# If object stopped moving, and area criteria as well as ratio criteria are satisfied
	# Stop the detecting process by setting detected flag to 1
	if (abs(x_detect - prev_x) < 1 and abs(y_detect - prev_y) < 1 and w_detect * h_detect > 1200 
		and w_detect * h_detect < 5000 and ratio_detect > 0.7 and ratio_detect < 1.3):
		detected_flag = 1
		cnt_change = []

	# If object is now locked (detected flag equals 1)
	if detected_flag == 1:
		min_distance = 1000
		min_index = 0

		# Find contours of all objects in current frame
		(cnts, _) = cv2.findContours(cur_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		rect = []
		x_track = 0
		y_track = 0
		w_track = 0
		h_track = 0

		# Loop through found contours and find the object whose distance to detected object is smallest
		# (as well as area and ratio criterias)
		# That is the object detected before whose position changed by the moving of the camera
		for (i, cnt) in enumerate(cnts):
			rect = cv2.boundingRect(cnt)
			distance = np.sqrt((rect[0] - x_detect)**2 + (rect[1] - y_detect)**2)
			ratio = rect[2] / float(rect[3])
			area = rect[2] * rect[3]
			if (distance < 20 and distance < min_distance and ratio > 0.7 and ratio < 1.3 and area < 10000 and area > 1200):
				text = 'Occupied!'
				min_distance = distance
				min_index = i
				x_track = rect[0]
				y_track = rect[1]
				w_track = rect[2]
				h_track = rect[3]
			
		# Update detected object's new position only if the criteria above was satisfied at least once
		if (x_track != 0 and y_track != 0):
			x_detect = x_track
			y_detect = y_track
			w_detect = w_track
			h_detect = h_track
		else:
			# Otherwise, we lost the object
			text = 'Lost!'

		# When object is lost, (x_tract, y_track) is (0, 0) so we can draw a line to know the last position of the detected object
		cv2.line(cur_frame, (x_detect, y_detect), (x_track, y_track), (0, 0, 255))
		cv2.rectangle(cur_frame, (x_track, y_track), (x_track + w_track, y_track + h_track), (0, 255, 0), 0)
		cv2.putText(cur_frame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

	key = cv2.waitKey(1) & 0xFF
	if (key == ord('q')):
		break

	cv2.imshow('Frame', cur_frame)
	cv2.imshow('Tracking', delta_frame)

cap.release()
cv2.destroyAllWindows()