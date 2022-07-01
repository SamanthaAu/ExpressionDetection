import numpy as np
import cv2

x = np.load('./modXtest.npy')
y = np.load('./modytest.npy')
x_all = np.load('./fdataX.npy')
# y = np.load('./flabels.npy')

std = np.std(x_all, axis = 0)
mean = np.mean(x_all, axis = 0)

x *= std
x += mean

for line in x:
    xx = np.asarray(line).reshape(72, 72)
    xx = xx.astype(np.uint8)
    cv2.imshow("pic", xx)
    cv2.waitKey(3000) 
    #SCALE?
    break

