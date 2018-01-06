import cv2
import numpy as np
from matplotlib import pyplot as plt
 
img_path = "temp_roi/57/3.png"
template_path = "test_roi/56/2.png"

def templateCheck(imagetocheckpath, templatepath):
    
    print "templateCheck"

    imagetocheck = cv2.imread(imagetocheckpath)
    cv2.imshow("check", imagetocheck)
    imagetocheck2 = cv2.imread(imagetocheckpath,0)
    iw, ih = imagetocheck2.shape[::-1]
    img_gray = cv2.cvtColor(imagetocheck, cv2.COLOR_BGR2GRAY)
    img2 = imagetocheck.copy()
    disp_img = cv2.imread(img_path)
    template = cv2.imread(templatepath,0)
    ctemp = cv2.imread(templatepath)
    w, h = template.shape[::-1]

    newwidth = w
    newheight = h

    wratio = float(w)/float(iw)
    hratio = float(h)/float(ih)
    resizeratio = hratio

    if wratio > hratio:
        resizeratio = wratio

    if resizeratio > 0:
        newheight = float(h)/resizeratio
        newwidth = float(w)/resizeratio

    resized_t = cv2.resize(ctemp,(int(newwidth), int(newheight)), interpolation = cv2.INTER_CUBIC)
    cv2.imshow("Resized", resized_t)
    cv2.imwrite("rz.png", resized_t)
    print "templateCheck saving rz"

    nt = cv2.imread("rz.png",0)
    wt,ht = nt.shape[::-1]
    print "templateCheck loading rz"

    # Apply template Matching
    res = cv2.matchTemplate(img_gray,nt,cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    threshold = 0.4
    loc = np.where(res >= threshold)
    zip_loc = zip(*loc[::-1])
    if len(zip_loc) > 0:
        print "templateCheck Found"
        pt = zip_loc[int(len(zip_loc)/2)]
        cv2.rectangle(disp_img, pt, (pt[0] + wt, pt[1] + ht), (0,0,255), 2)
        cv2.putText(disp_img,"p1_l", (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
    else:
        print "templateCheck Not found"
        cv2.putText(disp_img,"Not Found", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)

    cv2.imshow('Detected',disp_img)
    cv2.waitKey()


templateCheck(img_path, template_path)