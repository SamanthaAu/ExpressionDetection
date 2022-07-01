# load json and create model
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import numpy as np
import cv2
import mediapipe as mp
from enum import Enum
from google.protobuf.json_format import MessageToDict

count = 0
totalCount = 0
acc = 0.0
mp_hands = mp.solutions.hands
class HandState(Enum):
	LEFT = 1
	RIGHT = 2
	BOTH = 3
	NONE = 4
hand_state = HandState.NONE

#loading the model
emotion_json_file = open('fer.json', 'r')
emotion_loaded_model_json = emotion_json_file.read()
emotion_json_file.close()
emotion_loaded_model = model_from_json(emotion_loaded_model_json)
emotion_loaded_model.load_weights("fer.h5")
print("Loaded emotion model from disk")

combined_json_file = open('combinedfer.json', 'r')
combined_loaded_model_json = combined_json_file.read()
combined_json_file.close()
combined_loaded_model = model_from_json(combined_loaded_model_json)
combined_loaded_model.load_weights("combinedfer.h5")
print("Loaded combined model from disk")

#setting image resizing parameters
WIDTH = 19
HEIGHT = 7
x=None
y=None
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
txtFile = open('testFilepaths.txt', 'r')
filenames = txtFile.read().split(',')[:-1]
j=0

for imgFile in filenames:
    full_size_image = cv2.imread(imgFile) 
    emotion = imgFile[11:imgFile.find('/', 11)]	

    # print(imgFile + " Loaded")
    gray=cv2.cvtColor(full_size_image,cv2.COLOR_RGB2GRAY)
    faceClass = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face = faceClass.detectMultiScale(gray, 1.3  , 10)[0]
    x, y, w, h = face[0], face[1], face[2], face[3]

    roi_gray = gray[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (72, 72)), -1), 0)
    cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
    cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    #predicting the emotion
    yhat= emotion_loaded_model.predict(cropped_img)[0].tolist()
    # print(yhat)
    # print("Emotion: "+labels[int(np.argmax(yhat))])
        
    landmarks_list = []
    landmarks_list_formatted = []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        image = cv2.flip(cv2.imread(imgFile), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            if len(results.multi_handedness) == 2:
                hand_state = HandState.BOTH
            else:
                handedness_dict = MessageToDict(results.multi_handedness[0])
                hand = handedness_dict['classification'][0]['label']

                if hand == 'Left':
                    hand_state = HandState.LEFT
                else:
                    hand_state = HandState.RIGHT

            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_list.append(np.array(hand_landmarks.landmark))
            
            if hand_state == HandState.LEFT:
                for i in range(21*3):
                    landmarks_list_formatted.append(0.0)
            
            for hand in range(len(results.multi_handedness)):
                for point in range(21):
                    landmarks_list_formatted.append(landmarks_list[hand][point].x)
                    landmarks_list_formatted.append(landmarks_list[hand][point].y)
                    landmarks_list_formatted.append(landmarks_list[hand][point].z)

            if hand_state == HandState.RIGHT:
                for i in range(21*3):
                    landmarks_list_formatted.append(0.0)

        else:
            hand_state = HandState.NONE
            for i in range(2*21*3):
                landmarks_list_formatted.append(0.0)

    # print(landmarks_list_formatted )
    combined_list = yhat + landmarks_list_formatted 
    combined_list = np.expand_dims(combined_list, 0)
    combinedyhat= combined_loaded_model.predict(combined_list)[0].tolist()

    # print("Emotion: "+labels[int(np.argmax(combinedyhat))])
    confidence = {labels[0]:combinedyhat[0], labels[1]:combinedyhat[1], labels[2]:combinedyhat[2], labels[3]:combinedyhat[3], labels[4]:combinedyhat[4], labels[5]:combinedyhat[5], labels[6]:combinedyhat[6]}
    # print(confidence)

    # print(emotion, labels[int(np.argmax(combinedyhat))])

    if(emotion == labels[int(np.argmax(combinedyhat))]):
        count+=1
        totalCount+=1
        
    # print()

    if (j+1) % 20 == 0:
        print(emotion + ":" + str(count/20))
        count = 0

    j+=1

acc = (totalCount/len(filenames))*100
print(acc)


