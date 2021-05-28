import math
import threading
import time
import cv2
import imutils
import numpy
import numpy as np
import time
import cv2
import dlib
import time
import os

from imutils.video import FileVideoStream

# Importing the libraries
import os
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import YOLOModelDetector as md
from PIL import Image, ImageDraw
import copy

from threading import Thread, Lock
from queue import Queue
import multiprocessing

from scipy.spatial import distance as dist
from collections import OrderedDict


class CentroidTracker():
    def __init__(self, maxDisappeared=100):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # Impending deletion
        # Cant have size of 0
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (_, startX, startY, w, h)) in enumerate(rects):
            cX = int(startX + (w / 2.0))
            cY = int(startY + (h / 2.0))
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            # Get vrv centroid
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # get distance between all
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # sorting
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                # skip uswd
                if row in usedRows or col in usedCols:
                    continue
                # assign new centroid coord
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                # Mark not disappear
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # mark pending disappear
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # increase counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # remove
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                # case have more o
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects


pos_face = Lock()
pos_feature = Lock()
rect_lock = Lock()
face_lock = Lock()

vid_path = 'abcc.mp4'
result_path = 'over_f.mp4'

detector = md.ModelDetector()
detector.prepare()

path = "shape_predictor_68_face_landmarks_GTX.dat"
predictor_of_landmark = dlib.shape_predictor(path)

v = FileVideoStream(vid_path)
vs = v.start()

default_size = (int(v.stream.get(3)), int(v.stream.get(4)))
#
size = (560, 560)
#
result = cv2.VideoWriter(os.path.splitext(result_path)[0] + '.avi',
                         cv2.VideoWriter_fourcc(*'MPEG'),
                         60, size)
#
time.sleep(2.0)
#
frame_rate = 30
prev = time.time()

i = 0
min_i = 20

faceRects = []
rect_raw = []

face_in = Queue()
face_out = Queue()
feature_out = Queue()

ct = CentroidTracker()


def predict_face(q, q_out_face, q_out_feature):
    while True:
        face = None
        if q.empty() is False:
            pos_face.acquire()
            try:
                while q.empty() is False:
                    face = q.get()
            finally:
                pos_face.release()

        if face is not None:
            x = detector.detect(Image.fromarray(face))
            pos_feature.acquire()
            try:
                while q_out_face.empty() is False:
                    q_out_face.get()
                q_out_face.put(x)
            finally:
                pos_feature.release()


threads = [
    Thread(target=predict_face,
           args=(face_in, face_out, feature_out),
           name='face_detect_thread' + str(i),
           daemon=True)
    for i in range(multiprocessing.cpu_count())
]

for t in threads:
    t.start()

extense_face = 10
inner_face_counter = extense_face

while vs.more():
    i += 1
    # grab the frame from the video stream, resize it, and convert it
    # to grayscale
    time_elapsed = time.time() - prev
    if time_elapsed < 1. / frame_rate:

        continue
    else:
        prev = time.time()

    frame = vs.read()
    if frame is None:
        break

    if i < min_i:
        pass
    else:
        pos_face.acquire()
        try:
            face_in.put(copy.deepcopy(frame))
        finally:
            pos_face.release()
        # predict_face(face_in, face_out, feature_out)
        # predict_face()
    # faceRects = detector.detect(Image.fromarray(frame))

    # Replace this with your detect

    # print('k')
    # # print(len(face_out))
    # print('k1')

    pos_feature.acquire()
    while face_out.empty() is False:
        faceRects = face_out.get()
    pos_feature.release()

    k = ct.update(faceRects)

    for (_, fX, fY, fW, fH) in faceRects:
        # extract the face ROI
        cv2.rectangle(frame, (fX.astype(int), fY.astype(int)), ((fX + fW).astype(int), (fY + fH).astype(int)),
                      (0, 255, 0), 1)
        rect = dlib.rectangle(fX.astype(int), fY.astype(int), (fX + fW).astype(int), (fY + fH).astype(int))
        #
        # if i < min_i:
        #     pass
        # else:
        if inner_face_counter > extense_face:
            rect_raw = numpy.matrix([[p.x, p.y] for p in predictor_of_landmark(frame, rect).parts()])
        else:
            inner_face_counter += 1

        for idx, point in enumerate(rect_raw):
            cv2.circle(frame, (point[0, 0], point[0, 1]), 0, (255, 0, 0), -1)

    for (id, coord) in k.items():
        cv2.putText(frame, str(id), (coord[0] - 10, coord[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if i < min_i:
        pass
    else:
        i = 0

    frame = cv2.resize(frame, size)

    result.write(frame)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

result.release()
cv2.destroyAllWindows()
vs.stop()
