from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import mediapipe as mp
import os
import cv2
import numpy as np
from hand import hand_keypoints

# FACE EXPRESSION MODEL

#loading the model
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
# loaded_model.load_weights("fer.h5")
print("Loaded model from disk")

#setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x=None
y=None
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#loading image
full_size_image = cv2.imread("myPics/sad.jpg") # disgust, surprise, neutral
print("Image Loaded")
gray=cv2.cvtColor(full_size_image,cv2.COLOR_RGB2GRAY)
faceClass = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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

landmarks = hand_keypoints("hands.jpeg")

for hand in range(2):
  for point in range(21):
    landmarks[hand][point] = [landmarks[hand][point].x, landmarks[hand][point].y, landmarks[hand][point].z]

landmarks = np.array(landmarks)






# # bimodal = np.concatenate((yhat, landmarks), axis=0)
# # print(bimodal)
# # model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
# # model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# # model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# # model.add(BatchNormalization())
# # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# # model.add(Dropout(0.5))

# # model.add(Flatten())

# # model.add(Dense(2*num_features, activation='relu'))
# # model.add(Dropout(0.5))

# # model.add(Dense(num_labels, activation='softmax'))

# # model.add(Flatten())
# # model.add(Dense(2*2*2*num_features, activation='relu'))
# # model.add(Dropout(0.4))


# # maybe 3 sets of FCN + activation), and make sure the final layer output size is again N. Then, you can use a softmax layer to turn these 
# # values into 'probabilities', and let each probability be the probability of that particular emotion.