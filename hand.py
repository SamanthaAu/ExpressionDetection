import cv2
import numpy as np
import mediapipe as mp
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
landmarks_list_formatted = []

for hand in range(2):
  for point in range(21):
    landmarks_list_formatted.append(landmarks[hand][point].x)
    landmarks_list_formatted.append(landmarks[hand][point].y)
    landmarks_list_formatted.append(landmarks[hand][point].z)

landmarks_list_formatted = np.array(landmarks_list_formatted)
print(landmarks_list_formatted)