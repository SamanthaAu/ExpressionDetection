emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

with open('filepaths.txt', 'w') as f:
    for emotion in emotions:
        for i in range(300):
            filePath = "./Data/" + emotion + "/" + emotion  + str(i) + ".jpg" 
            f.write(filePath + ',')

txtFile = open('filepaths.txt', 'r')
filenames = txtFile.read().split(',')
txtFile.close()






