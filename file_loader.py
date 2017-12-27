import os, glob, cv2, numpy
path = 'test/'
files = glob.glob( os.path.join(path, '*.*'))
files.sort()
for infile in files:
    print("current file is: " + infile)
    img_rgb = cv2.imread(infile)
    cv2.imshow(str(infile),img_rgb)
    cv2.waitKey()


