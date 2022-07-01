import cv2
import csv
from os.path import exists

labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
txtFile = open('filepaths.txt', 'r')
filenames = txtFile.read().split(',')[:-1]

for filename in filenames:

    print(filename)

    img = cv2.imread(filename)
    crop_img = img[580:1660, 0:1080]

    emotion = filename[7:filename.find('/', 7)]
    emotionInd = labels.index(emotion)

    resized = cv2.resize(crop_img, (72, 72), interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).flatten()

    data_list = [str(elmt) for elmt in gray]
    file_exists = exists('data.csv')
    
    with open('data.csv', 'a', newline='') as csvfile:
        fieldnames = ['emotion', 'pixels']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
		
        if not file_exists:
            writer.writeheader()

        writer.writerow({"emotion": emotionInd, "pixels": " ".join(data_list)})


	