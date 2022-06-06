emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

with open('testFilepaths.txt', 'w') as f:
    for emotion in emotions:
        for i in range(20):
            filePath = "./TestData/" + emotion + "/" + emotion  + str(i) + ".jpg" 
            f.write(filePath + ',')

txtFile = open('testFilepaths.txt', 'r')
filenames = txtFile.read().split(',')
txtFile.close()






