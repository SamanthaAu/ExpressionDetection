emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# write 4500 filepaths per emotion
with open('filepaths.txt', 'w') as f:
    for emotion in emotions:
        for i in range(4500):
            filePath = "./Data/" + emotion + "/" + emotion  + str(i) + ".jpg" 
            f.write(filePath + ',')

txtFile = open('filepaths.txt', 'r')
filenames = txtFile.read().split(',')
txtFile.close()






