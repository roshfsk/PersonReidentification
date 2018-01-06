import cv2
import numpy as np

#s_methods = ['cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']            

img_rgb = cv2.imread('temp_roi/2/0.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('rz.png',0)#'test_roi/2_roi.png',0)
w, h = template.shape[::-1]

# Apply template Matching
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)

#for pt in zip(*loc[::-1]):
pt = zip(*loc[::-1])[0]
cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv2.putText(img_rgb,"p1_l", (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
cv2.imshow('Detected',img_rgb)
cv2.waitKey()
#import time
#time.sleep(5)