import cv2 

cap= cv2.VideoCapture('happy1.mp4')
i = 0
while(cap.isOpened() and i < 300):
    ret, frame = cap.read()
    if ret == False:
        break
    if i%2 == 0:
        cv2.imwrite('happy'+str(int(i/3) + 0)+'.jpg',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()