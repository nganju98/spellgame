import matplotlib.pyplot as plt
import os
import keras_ocr
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline(max_size=1000)
import scipy.misc

#image = keras_ocr.tools.read("./pet_black2.jpg")
image = keras_ocr.tools.read("https://upload.wikimedia.org/wikipedia/commons/b/bd/Army_Reserves_Recruitment_Banner_MOD_45156284.jpg")
# Get a set of three example images
#images = [
##    keras_ocr.tools.read(url) for url in [
#         'https://upload.wikimedia.org/wikipedia/commons/b/bd/Army_Reserves_Recruitment_Banner_MOD_45156284.jpg',
#         #'https://upload.wikimedia.org/wikipedia/commons/e/e8/FseeG2QeLXo.jpg',
#         'https://upload.wikimedia.org/wikipedia/commons/b/b4/EUBanana-500x112.jpg'
#     ]
# ]

#rgb = Image.fromarray(images[0])
#cv2.imshow("test", images[0])
#cv2.waitKey(0)
# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
x = 1 # displays the frame rate every 1 second
counter = 0
start_time = time.time()
while True:
    counter+=1
    if (time.time() - start_time) > x :
        print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()
    prediction_groups = pipeline.recognize([image])
    keras_ocr.tools.drawAnnotations(image=image, predictions=prediction_groups[0])
    # Blue color in BGR
    color = (255, 0, 0)
  
# Line thickness of 2 px
    thickness = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
    org = (50, 50)
  
# fontScale
    fontScale = 1
    groups = prediction_groups[0]
#groups[image][textgroup][0]
    for group in groups:

        point = np.int32( group[1][0])
        text = group[0]
        cv2.putText(image, text, (point[0], point[1] - 80), font, fontScale, color, thickness)
        rect = np.int32([group[1]])
        #print(rect)
        cv2.polylines(image, rect, isClosed=True, color=color, thickness=thickness)
        
    #cv2.putText(image, "Pet32", org, font, fontScale, color, thickness)
    #cv2.polylines(image, np.int32([prediction_groups[0][0][1 ]]), isClosed=True, color=color, thickness=thickness)
    cv2.imshow("im",image)
    cv2.waitKey(0)
    #print(prediction_groups)
# Plot the predictions
#fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
#for ax, image, predictions in zip(axs, images, prediction_groups):
##    keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
#    cv2.imshow('img', image)
#    cv2.waitKey(0)
    #plt.imshow(image,aspect="auto")
    #plt.show()  
    
