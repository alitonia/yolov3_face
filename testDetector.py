# Importing the libraries
import os
import cv2
from matplotlib import pyplot as plt
# import YOLOModelDetector as md

#Prepare model
# detector = md.ModelDetector()
# detector.prepare()

#Use model to predict 
test_image_path = 'anh_hoi_dong.jpg'
img = cv2.imread(test_image_path)
cv2.imshow("window",img)
cv2.waitKey(0)

# detected = detector.detect(img)

# #Write to new file named "resultbb.txt"
# file_path = "resultbb.txt"
# with open(file_path, 'w') as file:
#     for box in detected:
#         for (x,y,w,h) in box: 
#             file.write(str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')
        