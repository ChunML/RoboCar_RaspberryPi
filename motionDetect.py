import cv2
import argparse
import datetime
import imutils
import time
import sys

def detect_change():
	delta_frame = cv2.absdiff(prev_frame, gray)
	thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=4)
	(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if len(cnts) == 0:
		return []
	else:
		return cnts[0]

ap = argparse.ArgumentParser()
ap.add_argument('-a', '--min-area', type=int, default=500)
args = vars(ap.parse_args())

cap = cv2.VideoCapture(0)
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

prev_frame = None

first_flag = 1
x_detect = 0
y_detect = 0
w_detect = 0
h_detect = 0

prev_x = 0
prev_y = 0

while True:
	ret, cur_frame = cap.read()
	text = 'Nothing detected'

	if not ret:
		break

	cur_frame = imutils.resize(cur_frame, width=500)
	gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)
	gray = cv2.medianBlur(gray, 7)

	delta_frame = cur_frame.copy()

	if prev_frame is None:
		prev_frame = gray
		continue

	cur_thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
	cur_thresh = cv2.dilate(cur_thresh, None, iterations=4)
	cv2.imshow('Thresh', cur_thresh)

	(cnts, _) = cv2.findContours(cur_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if first_flag:
		cnt_change = detect_change()
		prev_x = x_detect
		prev_y = y_detect

	if (len(cnt_change) != 0):
		x_detect, y_detect, w_detect, h_detect = cv2.boundingRect(cnt_change)
	
	cv2.rectangle(delta_frame, (x_detect, y_detect), (x_detect + w_detect, y_detect + h_detect), (0, 255, 0), 0)

	if (abs(x_detect - prev_x) < 1 and abs(y_detect - prev_y) < 1 and w_detect * h_detect > 1500 and w_detect * h_detect < 5000):
		first_flag = 0
		cnt_change = []
	
	for c in cnts:
		if (cv2.contourArea(c) < 500 or cv2.contourArea(c) > 3000):
			continue

		x, y, w, h = cv2.boundingRect(c)
		cv2.rectangle(cur_frame, (x, y), (x + w, y + h), (0, 255, 0), 0)
		text = 'Detected!'

	key = cv2.waitKey(1) & 0xFF
	if (key == ord('q')):
		break

	cv2.imshow('Frame', cur_frame)
	cv2.imshow('Tracking', delta_frame)

cap.release()
cv2.destroyAllWindows()