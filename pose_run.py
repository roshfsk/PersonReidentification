import numpy as np
import cv2
import os
import glob
import subprocess

video = ""
folder = "/home/roshan/Workspace/RESEARCH/Realtime_Multi-Person_Pose_Estimation/testing/python/test/*.jpg"

#create frames from video
#frameSplitCmd = "ffmpeg -i test.mp4 -vf fps=1/30 test/%06d.jpg"
#subprocess.call(frameSplitCmd,True)
os.system('GREPDB="ffmpeg -i test.mp4 -vf fps=1/30 test/%06d.jpg"; /bin/bash -c "$GREPDB"') 

#Read from frames dir
filenames = [img for img in glob.glob(folder)]
filenames.sort()

# for filename in os.listdir(folder):
for filename in filenames:
  img = cv2.imread(os.path.join(folder,filename))
  if img is not None:
    # img = cv2.imread('/home/roshan/Pictures/RoshanF.png',0)
    cv2.imshow('image',img)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('messigray.png',img)
        cv2.destroyAllWindows()