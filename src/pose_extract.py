#!/usr/bin/python

import cv2 as cv 
import numpy as np
import scipy
import PIL.Image
import math
import caffe
import time
from config_reader import config_reader
import util
import copy
import matplotlib
import pylab as plt
import os, glob
import sys

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

index_pos = 1
fileIndex = 6


if len(sys.argv) > 1:
    fileIndex = int(sys.argv[1])

threshold1 = 0.1
threshold2 = 0.4
roiMatchMethod1=cv.TM_SQDIFF_NORMED#cv.TM_CCOEFF
roiMatchMethod2=cv.TM_CCOEFF_NORMED
outMatchMethod=cv.TM_SQDIFF_NORMED
inpath = 't7/'
posepath = 't7_pose/'
temp_roi_dir = 't7_temp_roi/'
roipath = 't7_roi/'
outpath = 't7_out/'
test_image = inpath + str(fileIndex) + '.jpg'

oriImg = cv.imread(test_image) # B,G,R order
f = plt.imshow(oriImg[:,:,[2,1,0]]) # reorder it before displaying

param, model = config_reader()
multiplier = [x * model['boxsize'] / oriImg.shape[0] for x in param['scale_search']]

if param['use_gpu']: 
    caffe.set_mode_gpu()
    caffe.set_device(param['GPUdeviceNumber']) # set to your device!
else:
    caffe.set_mode_cpu()
net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
# first figure shows padded images
f, axarr = plt.subplots(1, len(multiplier))
f.set_size_inches((20, 5))
# second figure shows heatmaps
f2, axarr2 = plt.subplots(1, len(multiplier))
f2.set_size_inches((20, 5))
# third figure shows PAFs
f3, axarr3 = plt.subplots(2, len(multiplier))
f3.set_size_inches((20, 10))

for m in range(len(multiplier)):
    scale = multiplier[m]
    imageToTest = cv.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model['stride'], model['padValue'])
    print imageToTest_padded.shape
    
    axarr[m].imshow(imageToTest_padded[:,:,[2,1,0]])
    axarr[m].set_title('Input image: scale %d' % m)

    net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
    #net.forward() # dry run
    net.blobs['data'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
    start_time = time.time()
    output_blobs = net.forward()
    print('At scale %d, The CNN took %.2f ms.' % (m, 1000 * (time.time() - start_time)))

    # extract outputs, resize, and remove padding
    heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1,2,0)) # output 1 is heatmaps
    heatmap = cv.resize(heatmap, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
    heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
    
    paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1,2,0)) # output 0 is PAFs
    paf = cv.resize(paf, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
    paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
    paf = cv.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
    
    # visualization
    axarr2[m].imshow(oriImg[:,:,[2,1,0]])
    ax2 = axarr2[m].imshow(heatmap[:,:,3], alpha=.5) # right wrist
    axarr2[m].set_title('Heatmaps (Rwri): scale %d' % m)
    
    axarr3.flat[m].imshow(oriImg[:,:,[2,1,0]])
    ax3x = axarr3.flat[m].imshow(paf[:,:,16], alpha=.5) # right elbow
    axarr3.flat[m].set_title('PAFs (x comp. of Rwri to Relb): scale %d' % m)
    axarr3.flat[len(multiplier) + m].imshow(oriImg[:,:,[2,1,0]])
    ax3y = axarr3.flat[len(multiplier) + m].imshow(paf[:,:,17], alpha=.5) # right wrist
    axarr3.flat[len(multiplier) + m].set_title('PAFs (y comp. of Relb to Rwri): scale %d' % m)
    
    heatmap_avg = heatmap_avg + heatmap / len(multiplier)
    paf_avg = paf_avg + paf / len(multiplier)

f2.subplots_adjust(right=0.93)
cbar_ax = f2.add_axes([0.95, 0.15, 0.01, 0.7])
_ = f2.colorbar(ax2, cax=cbar_ax)

f3.subplots_adjust(right=0.93)
cbar_axx = f3.add_axes([0.95, 0.57, 0.01, 0.3])
_ = f3.colorbar(ax3x, cax=cbar_axx)
cbar_axy = f3.add_axes([0.95, 0.15, 0.01, 0.3])
_ = f3.colorbar(ax3y, cax=cbar_axy)

#Let's have a closer look on those averaged heatmaps and PAFs!

plt.imshow(oriImg[:,:,[2,1,0]])
plt.imshow(heatmap_avg[:,:,2], alpha=.5)
fig = matplotlib.pyplot.gcf()
cax = matplotlib.pyplot.gca()
fig.set_size_inches(20, 20)
fig.subplots_adjust(right=0.93)
cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
_ = fig.colorbar(ax2, cax=cbar_ax)

from numpy import ma
U = paf_avg[:,:,16] * -1
V = paf_avg[:,:,17]
X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
M = np.zeros(U.shape, dtype='bool')
M[U**2 + V**2 < 0.5 * 0.5] = True
U = ma.masked_array(U, mask=M)
V = ma.masked_array(V, mask=M)

# 1
plt.figure()
plt.imshow(oriImg[:,:,[2,1,0]], alpha = .5)
s = 5
Q = plt.quiver(X[::s,::s], Y[::s,::s], U[::s,::s], V[::s,::s], 
            scale=50, headaxislength=4, alpha=.5, width=0.001, color='r')

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(20, 20)

import scipy
#print heatmap_avg.shape

#plt.imshow(heatmap_avg[:,:,2])
from scipy.ndimage.filters import gaussian_filter
all_peaks = []
peak_counter = 0

for part in range(19-1):
    x_list = []
    y_list = []
    map_ori = heatmap_avg[:,:,part]
    map = gaussian_filter(map_ori, sigma=3)
    
    map_left = np.zeros(map.shape)
    map_left[1:,:] = map[:-1,:]
    map_right = np.zeros(map.shape)
    map_right[:-1,:] = map[1:,:]
    map_up = np.zeros(map.shape)
    map_up[:,1:] = map[:,:-1]
    map_down = np.zeros(map.shape)
    map_down[:,:-1] = map[:,1:]
    
    peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
    peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
    peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
    id = range(peak_counter, peak_counter + len(peaks))
    peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

    all_peaks.append(peaks_with_score_and_id)
    peak_counter += len(peaks)

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
        [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
        [1,16], [16,18], [3,17], [6,18]]
# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
        [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
        [55,56], [37,38], [45,46]]

connection_all = []
special_k = []
mid_num = 10

for k in range(len(mapIdx)):
    score_mid = paf_avg[:,:,[x-19 for x in mapIdx[k]]]
    candA = all_peaks[limbSeq[k][0]-1]
    candB = all_peaks[limbSeq[k][1]-1]
    nA = len(candA)
    nB = len(candB)
    indexA, indexB = limbSeq[k]
    if(nA != 0 and nB != 0):
        connection_candidate = []
        for i in range(nA):
            for j in range(nB):
                vec = np.subtract(candB[j][:2], candA[i][:2])
                norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                vec = np.divide(vec, norm)
                
                startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                            np.linspace(candA[i][1], candB[j][1], num=mid_num))
                
                vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                for I in range(len(startend))])
                vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                for I in range(len(startend))])

                score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                try:
                    score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                except Exception:
                    pass
                criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                criterion2 = score_with_dist_prior > 0
                if criterion1 and criterion2:
                    connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

        connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
        connection = np.zeros((0,5))
        for c in range(len(connection_candidate)):
            i,j,s = connection_candidate[c][0:3]
            if(i not in connection[:,3] and j not in connection[:,4]):
                connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                if(len(connection) >= min(nA, nB)):
                    break

        connection_all.append(connection)
    else:
        special_k.append(k)
        connection_all.append([])

subset = -1 * np.ones((0, 20))
candidate = np.array([item for sublist in all_peaks for item in sublist])

for k in range(len(mapIdx)):
    if k not in special_k:
        partAs = connection_all[k][:,0]
        partBs = connection_all[k][:,1]
        indexA, indexB = np.array(limbSeq[k]) - 1

        for i in range(len(connection_all[k])): #= 1:size(temp,1)
            found = 0
            subset_idx = [-1, -1]
            for j in range(len(subset)): #1:size(subset,1):
                if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                    subset_idx[found] = j
                    found += 1
            
            if found == 1:
                j = subset_idx[0]
                if(subset[j][indexB] != partBs[i]):
                    subset[j][indexB] = partBs[i]
                    subset[j][-1] += 1
                    subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
            elif found == 2: # if found 2 and disjoint, merge them
                j1, j2 = subset_idx
                print "found = 2"
                membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                if len(np.nonzero(membership == 2)[0]) == 0: #merge
                    subset[j1][:-2] += (subset[j2][:-2] + 1)
                    subset[j1][-2:] += subset[j2][-2:]
                    subset[j1][-2] += connection_all[k][i][2]
                    subset = np.delete(subset, j2, 0)
                else: # as like found == 1
                    subset[j1][indexB] = partBs[i]
                    subset[j1][-1] += 1
                    subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

            # if find no partA in the subset, create a new subset
            elif not found and k < 17:
                row = -1 * np.ones(20)
                row[indexA] = partAs[i]
                row[indexB] = partBs[i]
                row[-1] = 2
                row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                subset = np.vstack([subset, row])

# delete some rows of subset which has few parts occur
deleteIdx = [];
for i in range(len(subset)):
    if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
        deleteIdx.append(i)
subset = np.delete(subset, deleteIdx, axis=0)

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
        [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
cmap = matplotlib.cm.get_cmap('hsv')

canvas = cv.imread(test_image) # B,G,R order

clothingArea=[2,5,8,11,10,13]
for i in clothingArea:
    for j in range(len(all_peaks[i])):
        cv.circle(canvas, all_peaks[i][j][0:2], 4, [255, 0, 0], thickness=-1)
    # print "Peak : " + str(i) + " People : " + str(len(all_peaks[i]))       

to_plot = cv.addWeighted(oriImg, 0.3, canvas, 0.7, 0)
plt.imshow(to_plot[:,:,[2,1,0]])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12, 12)

# visualize 2
stickwidth = 4

for i in range(17):
    for n in range(len(subset)):
        index = subset[n][np.array(limbSeq[i])-1]
        if -1 in index:
            continue
        cur_canvas = canvas.copy()
        Y = candidate[index.astype(int), 0]
        X = candidate[index.astype(int), 1]
        mX = np.mean(X)
        mY = np.mean(Y)
        # print "index : " + str(i) + " Y : " + str(Y) + " X : " + str(X) + " mX : " + str(mX) + " mY : " + str(mY)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
        cv.fillConvexPoly(cur_canvas, polygon, colors[i])
        canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
plt.imshow(canvas[:,:,[2,1,0]])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12, 12)

# fig.show()
# time.sleep(5)
if not os.path.exists(posepath):
        os.makedirs(posepath)
fig.savefig(posepath + str(fileIndex) + "_pose.png")

from timeit import default_timer as timer

rec_start = timer()

personCount = len(all_peaks[2])
print "Number of detected people : " + str(len(subset))
pepsList=[]
pIndex=0

def getMeanCoordinates(n,i):
    if len(subset) < n:
        return [0,0]
    try:
        index = subset[n][np.array(limbSeq[i])-1]
    except Exception:
        return [0,0]
    if -1 in index:
        return [0,0]
    Y = candidate[index.astype(int), 0]
    X = candidate[index.astype(int), 1]
    mX = np.mean(X)
    mY = np.mean(Y)
    return [int(X[1]),int(Y[1])]

#store all x, y coordinates into person arrays
#get min max for x and y to draw the bounding box

for positionIndex in range(personCount):
    
    print "Person" + str(positionIndex) + " : " + str(pIndex)        
    individual={}
    individual["id"]=pIndex
    for l_index in range(17):
        individual[str(l_index)]=getMeanCoordinates(positionIndex,l_index)
    pepsList.append(individual)
    pIndex=pIndex+1

#ROI extraction and template matching

img_rgb = cv.imread(test_image,0)
out_img_rgb = cv.imread(test_image)
op_img_rgb = cv.imread(test_image)
foundintemplatefiles=[]

for person in pepsList:
    pid = person["id"]
    xc=[]
    yc=[]
    for pli in range(17):
        xc.append(person[str(pli)][0])
        yc.append(person[str(pli)][1])

    nzyc = np.array(yc)
    yc_list = []
    for yc_val in nzyc[np.nonzero(yc)]:
        yc_list.append(yc_val)

    nzxc = np.array(xc)
    xc_list = []
    for xc_val in nzxc[np.nonzero(xc)]:
        xc_list.append(xc_val)

    if (len(xc_list) <= 3) and (len(yc_list) <= 3):
        break  

    right_x = min(xc_list)
    left_x = max(xc_list)
    up_y = min(yc_list)
    low_y = max(yc_list)

    if (left_x-right_x < 20) or (low_y-up_y < 20):
        break

    print "pid: " + str(pid) +" right_x: " + str(right_x) + " left_x: " + str(left_x) + " up_y: " + str(up_y) + " low_y: " + str(low_y)

    #roi = op_img_rgb[up_y:low_y,right_x:left_x]
    roi = op_img_rgb[right_x:left_x,up_y:low_y]
    temproipath = temp_roi_dir + str(fileIndex) + "/"
    if not os.path.exists(temproipath):
        os.makedirs(temproipath)
    cv.imwrite(temproipath + str(pid) + ".png", roi)
    
#--------------------  

    if fileIndex > index_pos:
        matcherroipath = temproipath + str(pid) + ".png"
        matcherroi = cv.imread(matcherroipath)
        # wm, hm = matcherroi.shape[::-1]

        templateloadpath = r'' + roipath + str(fileIndex-1) +'/'
        files = glob.glob( os.path.join(templateloadpath, '*.*'))
        files.sort()

        intemplatefileindex = 0
        for intemplatefile in files:
            print "template matcher inpath : " + intemplatefile
            foundlen = len(foundintemplatefiles)
            print "Found paths count : " + str(len(files))

            if foundlen > 0:
                try:
                    findex = foundintemplatefiles.index(intemplatefile)
                    if findex >= 0:
                        if intemplatefileindex == len(files)-1:
                            if len(files) < len(pepsList):
                                print "templateCheck adding new entry"
                                newsavepath = r'' + roipath + str(fileIndex) +'/' 
                                if not os.path.exists(newsavepath):
                                    os.makedirs(newsavepath)
                                cv.imwrite(newsavepath + str(pid) + ".png", roi)
                                # intemplatefileindex += 1
                                break
                        print intemplatefile + " Already found. Skipping...."        
                        continue
                except ValueError:
                    print intemplatefile + " Not used before. passing on...."
                    pass
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            imagetocheck = cv.imread(matcherroipath)
            imagetocheck2 = cv.imread(matcherroipath,0)
            iw, ih = imagetocheck2.shape[::-1]
            img_gray = cv.cvtColor(imagetocheck, cv.COLOR_BGR2GRAY)

            template = cv.imread(intemplatefile,0)
            ctemp = cv.imread(intemplatefile)
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

            resized_t = cv.resize(ctemp,(int(newwidth), int(newheight)), interpolation = cv.INTER_CUBIC)
            # cv2.imshow("Resized", resized_t)
            cv.imwrite(str(pid)+"rz.png", resized_t)

            nt = cv.imread(str(pid)+"rz.png",0)
            wt,ht = nt.shape[::-1]

            # Apply template Matching 1
            res1 = cv.matchTemplate(img_gray,nt,roiMatchMethod1)
            min_val1, max_val1, min_loc1, max_loc1 = cv.minMaxLoc(res1)
            print "min_val1 : " + str(min_val1) + " max_val1 : " + str(max_val1) + " min_loc1 : " + str(min_loc1) + " max_loc1 : " + str(max_loc1)
            loc1 = np.where(res1 >= threshold1)
            zip_loc1 = zip(*loc1[::-1])

            # Apply template Matching 1
            res2 = cv.matchTemplate(img_gray,nt,roiMatchMethod2)
            min_val2, max_val2, min_loc2, max_loc2 = cv.minMaxLoc(res2)
            print "min_val2 : " + str(min_val2) + " max_val2 : " + str(max_val2) + " min_loc2 : " + str(min_loc2) + " max_loc2 : " + str(max_loc2)

            if max_val2 >= threshold2 : 
                print "Match found! Saving..."
                splittedpath = intemplatefile.split('/')
                person_id_png = splittedpath[len(splittedpath)-1]
                newsavepath = r'' + roipath + str(fileIndex) +'/' 
                if not os.path.exists(newsavepath):
                    os.makedirs(newsavepath)
                cv.imwrite(newsavepath + person_id_png, matcherroi)
                foundintemplatefiles.append(intemplatefile)
                break
            elif min_val1 <= threshold1:
                print "Match found! Saving..."
                splittedpath = intemplatefile.split('/')
                person_id_png = splittedpath[len(splittedpath)-1]
                newsavepath = r'' + roipath + str(fileIndex) +'/' 
                if not os.path.exists(newsavepath):
                    os.makedirs(newsavepath)
                cv.imwrite(newsavepath + person_id_png, matcherroi)
                foundintemplatefiles.append(intemplatefile)
                break
            else:
                print "templateCheck Not found : " + str(intemplatefileindex)
            intemplatefileindex += 1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---------------------
    else:
        newsavepath = r'' + roipath + str(fileIndex) +'/' 
        if not os.path.exists(newsavepath):
            os.makedirs(newsavepath)
        cv.imwrite(newsavepath + str(pid) + ".png", roi)


if fileIndex > index_pos:
    templateloadpath = r'' + roipath + str(fileIndex-1) +'/'
else:
    templateloadpath = r'' + roipath + str(fileIndex) +'/'
files = glob.glob( os.path.join(templateloadpath, '*.*'))
#files.sort()
for intemplatefile in files:
    
    print "intemplatefilePath : " + intemplatefile
    template = cv.imread(intemplatefile,0)
    w, h = template.shape[::-1]

    #Template Matching
    res = cv.matchTemplate(img_rgb,template,outMatchMethod)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(out_img_rgb,top_left, bottom_right, [0,0,255], 2)
    splittedpath = intemplatefile.split('/')
    person_id_file = splittedpath[len(splittedpath)-1]
    splitted_person_id_file = person_id_file.split('.')
    person_id = splitted_person_id_file[0]
    cv.putText(out_img_rgb,"P_" + person_id, (int(top_left[0]) + 10, int(top_left[1]) + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),1)

#cv.imshow('Detected',out_img_rgb)
#cv.waitKey()
if not os.path.exists(outpath):
    os.makedirs(outpath)
cv.imwrite(outpath + str(fileIndex) + ".png", out_img_rgb)
rec_end = timer()
print(rec_end - rec_start) 