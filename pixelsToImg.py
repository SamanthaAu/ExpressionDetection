import cv2
import csv
from skimage import img_as_float, img_as_ubyte
import pandas as pd
import numpy as np
import os
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
from enum import Enum

mp_hands = mp.solutions.hands
data = pd.read_csv('./data3.csv')

width, height = 72, 72

datapoints = data['pixels'].tolist()[0:3]

# getting features for training
X = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(72, 72)
    # img = img_as_ubyte(xx)
    # img = cv2.resize(img, (1080, 1080))
    xx = xx.astype(np.uint8)
    cv2.imshow("pic", xx)
    cv2.waitKey(300) 
    

    landmarks_list = []
    landmarks_list_formatted = []
    class HandState(Enum):
        LEFT = 1
        RIGHT = 2
        BOTH = 3
        NONE = 4
    hand_state = HandState.NONE

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        # image = cv2.flip(cv2.imread("./data/angry/angry1.jpg"), 1)
        # image = cv2.resize(image, (72, 72), interpolation = cv2.INTER_AREA)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print(gray.astype(np.uint8))
        # cv2.imshow("pic", gray)
        # cv2.waitKey(300) 

        results = hands.process(cv2.cvtColor(xx, cv2.COLOR_GRAY2RGB))

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
        else:
            print("hello")
    print(landmarks_list_formatted)
    




	