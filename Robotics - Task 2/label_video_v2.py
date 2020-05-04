# guidance to use this program
# 1. click the target point in the image (image will be saved automatically in the current folder)
# 2. press enter key to proceed to another image

#Note:
#Adjust the value of skipframes to skip some frames, we want the dataset to be diverse
#Adjust the nameprefix for each video to avoid having a same name for different images

import numpy as np
import cv2

skipframes=5
nameprefix='a'

cap = cv2.VideoCapture('output.mp4')
i=1
def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY,clickflag,i
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.imwrite(nameprefix+str(i)+'_xy_'+str(x)+'_'+str(y)+'.jpg',frame)
        cv2.circle(frame,(x,y),2,(0,0,255),-1)
        cv2.imshow('frame',frame)
        mouseX,mouseY = x,y
        i=i+1
        
cv2.namedWindow('frame')
cv2.setMouseCallback('frame',draw_circle)
while(cap.isOpened()):
    for x in range(skipframes):
        ret, frame = cap.read()
    if ret==True:
        cv2.imshow('frame',frame)
        key=cv2.waitKey(0)
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
