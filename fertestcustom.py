# load json and create model
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import numpy as np
import cv2

#loading the model
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("fer.h5")
print("Loaded model from disk")

#setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x=None
y=None
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#loading image
full_size_image = cv2.imread("test1.jpg") 
'''
    my happy (happy.jpg): angry
    online happy (happy2.jpeg): happy 
    my sad (sad.jpg): fear
    online sad (sad2.jpeg): sad
    my surprise: angry
    my fear: fear
    my discust: angry
    my angry: angry
    my neutral: angry
'''
print("Image Loaded")
gray=cv2.cvtColor(full_size_image,cv2.COLOR_RGB2GRAY)
face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face.detectMultiScale(gray, 1.3  , 10)

#detecting faces
for (x, y, w, h) in faces:
    roi_gray = gray[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
    cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    #predicting the emotion
    yhat= loaded_model.predict(cropped_img)[0]
    print(yhat)
    cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    print("Emotion: "+labels[int(np.argmax(yhat))])
    confidence = {labels[0]:yhat[0], labels[1]:yhat[1], labels[2]:yhat[2], labels[3]:yhat[3], labels[4]:yhat[4], labels[5]:yhat[5], labels[6]:yhat[6]}
    print(confidence)
    

cv2.imshow('Emotion', full_size_image)
cv2.waitKey(3000)