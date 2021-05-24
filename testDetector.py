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
test_image_path = 'anh_hoi_dong.jpg'
img = Image.open(test_image_path)
detected = detector.detect(img)

for conf,x,y,w,h in detected:
    print(conf,x,y,w,h)
        

        