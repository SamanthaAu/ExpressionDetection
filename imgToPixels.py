import cv2
import csv
from os.path import exists
import face_recognition

labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
txtFile = open('filepaths.txt', 'r')
filenames = txtFile.read().split(',')[:-1]

for filename in filenames:

    print(filename)

    # crop image to detected face
    img = face_recognition.load_image_file(filename)
    loc = face_recognition.face_locations(img)[0]
    crop_img = img[loc[0]:loc[2], loc[3]:loc[1]]

    cv2.imshow("pic", crop_img)
    cv2.waitKey(1)

    emotion = filename[7:filename.find('/', 7)]
    emotionInd = labels.index(emotion)

    resized = cv2.resize(crop_img, (72, 72), interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).flatten()

    data_list = [str(elmt) for elmt in gray]
    file_exists = exists('unimodal.csv')
    
    # write pixel values to file
    with open('unimodal.csv', 'a', newline='') as csvfile:
        fieldnames = ['emotion', 'pixels']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
		
        if not file_exists:
            writer.writeheader()

        writer.writerow({"emotion": emotionInd, "pixels": " ".join(data_list)})


	