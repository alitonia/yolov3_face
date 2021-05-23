# Importing the libraries
import os
import cv2
from matplotlib import pyplot as plt
import YOLOModelDetector as md
from PIL import Image
import numpy as np

#Prepare model
detector = md.ModelDetector()
detector.prepare()

#Use model to predict 
test_image_path = './samples/test.jpeg'
img = Image.open(test_image_path)

face_config = {"model_path": 'wider.h5',
               "anchors_path": 'wider_anchors.txt',
               "classes_path": 'wider_classes.txt',
               }

detected = detector.detect(img)
# print('Detected: ')
# print(detected)
# print("foo")
#Write to new file named "resultbb.txt"

def xyxy_to_xywh(boxes):
         # inverse function of the above function
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

file_path = "resultbb.txt"
with open(file_path, 'w') as file:
    print(xyxy_to_xywh(detected))
#         file.write(str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')
        

        