from keras.models import model_from_json
import numpy as np
import cv2
import mediapipe as mp
from enum import Enum
from google.protobuf.json_format import MessageToDict
import csv
from os.path import exists
import face_recognition

json_file = open('unimodalModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("unimodalModel.h5")
print("Loaded model from disk")

mp_hands = mp.solutions.hands
class HandState(Enum):
	LEFT = 1
	RIGHT = 2
	BOTH = 3
	NONE = 4
hand_state = HandState.NONE

labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
txtFile = open('filepaths.txt', 'r')
filenames = txtFile.read().split(',')[0:]
txtFile.close()

# process each image in dataset
for imgFile in filenames:

	print(imgFile)
	emotion = imgFile[7:imgFile.find('/', 7)]	

	# crop image to detected face

	# img = face_recognition.load_image_file(imgFile)
	# loc = face_recognition.face_locations(img)[0]
	# img = img[loc[0]:loc[2], loc[3]:loc[1]]
	# gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

	# try:
	# 	faceClass = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	# 	face = faceClass.detectMultiScale(gray, 1.3  , 10)[0]
	# 	x, y, w, h = face[0], face[1], face[2], face[3]
	# except IndexError:
	# 	continue
	
	# roi_gray = gray[y:y + h, x:x + w]

	# obtain emotion confidences using unimodal model
	# cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray, (72, 72)), -1), 0)
	# cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
	# yhat = loaded_model.predict(cropped_img)[0].tolist()

	# hands keypoint
	landmarks_list = []
	landmarks_list_formatted = []
	
	# detect hand keypoints
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

		# pad data with 0's if no hand detected
		else:
			hand_state = HandState.NONE
			for i in range(2*21*3):
				landmarks_list_formatted.append(0.0)
		
	# combined_list = yhat + landmarks_list_formatted 
	combined_list = landmarks_list_formatted
	combined_list.append(labels.index(emotion))
	data_list = [str(elmt) for elmt in combined_list][0 : len(combined_list) - 1]
	
	file_exists = exists('bimodal2.csv')

	# write datapoints to file
	with open('bimodal2.csv', 'a', newline='') as csvfile:
		fieldnames = ['data_values', 'emotion']
		writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
		
		if not file_exists:
			writer.writeheader()

		writer.writerow({"data_values": " ".join(data_list), "emotion": combined_list[len(combined_list) - 1]})