import cv2 

cap= cv2.VideoCapture('fear.mp4')
i = 0
while(cap.isOpened() and i < 240):
    ret, frame = cap.read()
    if ret == False:
        break
    if i%9 == 0:
        cv2.imwrite('fear'+str(int(i/9))+'.jpg',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()