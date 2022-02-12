from __future__ import division
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import mediapipe as mp
import os

# FACE EXPRESSION MODEL

# loading the model
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
faceClass = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#loading image
full_size_image = cv2.imread("myPics/sad.jpg") # disgust, surprise, neutral
print("Image Loaded")
gray=cv2.cvtColor(full_size_image,cv2.COLOR_RGB2GRAY)
print(np.shape(full_size_image))
face = faceClass.detectMultiScale(gray, 1.3  , 10)[0]
x, y, w, h = face[0], face[1], face[2], face[3]

roi_gray = gray[y:y + h, x:x + w]
cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

#predicting the emotion
yhat= loaded_model.predict(cropped_img)[0]

# HAND JOINT DETECTION
mp_hands = mp.solutions.hands

def hand_keypoints(img_file): 
  IMAGE_FILES = [img_file]
  landmarks_list = []

  with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=2,
      min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):

      image = cv2.flip(cv2.imread(file), 1)
      results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      if not results.multi_hand_landmarks:
        continue
      image_height, image_width, _ = image.shape
      annotated_image = image.copy()
      for hand_landmarks in results.multi_hand_landmarks:
        landmarks_list.append(np.array(hand_landmarks.landmark))

  return landmarks_list

landmarks = hand_keypoints("hands.jpeg")

for hand in range(2):
  for point in range(21):
    landmarks[hand][point] = [landmarks[hand][point].x, landmarks[hand][point].y, landmarks[hand][point].z]

landmarks = np.array(landmarks)
print(landmarks)

joined = 

''' 
define neural network model (91 stuff)
- fully connected
- activation layer
- fully connected
- softmax --> probabilities --> output
'''

'''
define network model (keras / pytorch/ tensormodel)
x = face/ hand datapoints
  y = class
model.fit(csv) or model.train(csv) (for keras) --> use x and y
'''


