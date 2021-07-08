# OBJECT DETECTION USING OPENCV PYTHON
# TASK 1
# The Sparks Foundation Network
# CHERRY MYO MYINT

import cv2  # pip install opencv-python

config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # algorithms
weight_path = 'frozen_inference_graph.pb'

classNames = []  # empty list of python
classFile = 'coco.names'  # Dataset
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

net = cv2.dnn_DetectionModel(weight_path, config_path)

# Configuration
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)  # 255/2 =127.5
net.setInputMean((127.5, 127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# VIDEO DEMO

cap = cv2.VideoCapture('ny.mp4')  # load the video

# Check if ht video is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cannot open video')

# Resolution of the frame
w = 860
h = 480
dim = (w, h)

font_scale = 1
font = cv2.FONT_HERSHEY_SIMPLEX

while True:  # play the video by reading frame by frame
    success, video = cap.read()
    frame = cv2.resize(video, dim, interpolation=cv2.INTER_AREA) # resize of the frame resolution0
    classIds, conf, bbox = net.detect(frame, confThreshold=0.59)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), conf.flatten(), bbox):
            if classId <= 80:
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, classNames[classId - 1], (box[0] + 10, box[1] + 40), font, font_scale,
                            (0, 255, 0), 2)

    cv2.imshow('Object Detection Tutorial', frame)

    # Press 'q' on keyboard to exit
    if cv2.waitKey(35) & 0xFF == ord('q'):
        break
# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
