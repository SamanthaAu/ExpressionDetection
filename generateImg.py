import cv2 

# generate images from video
cap= cv2.VideoCapture('surprise.mp4')
i = 0
while(cap.isOpened() and i < 4500):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('surprise'+str(int(i))+'.jpg',frame)
    i = i + 1
 
cap.release()
cv2.destroyAllWindows()