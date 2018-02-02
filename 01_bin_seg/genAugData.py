import shutil
import os
import cv2
import numpy as np
from scipy import ndimage as ndi # for filling operation only

numSubDirs = 50; # total number of subdirectories
stepSize = 50; # step size to create subimages
imgSize = 224; # image size
basePath = '../../data/01_bin_seg/train'
inRgbPath = 'rgb'
inGtPath = 'gt'
outRgbPath = 'rgb_aug'
outGtPath = 'gt_aug'

inRgbPath = os.path.join(basePath, inRgbPath)
inGtPath = os.path.join(basePath, inGtPath)
outRgbPath = os.path.join(basePath, outRgbPath)
outGtPath = os.path.join(basePath, outGtPath)
imgList = [];
numImgPerDir = [0]*(numSubDirs+1); # number of images per subdirectory

def checkExistence() :
    global imgList, inGtPath;
    imgList = os.listdir(inGtPath)

    if len(imgList) == 0 : # check if input directory is empty
        print 'Input directory is empty.'
        exit()
# -----------------------------------------------------------------------

def processDirs() :
    global outRgbPath, outGtPath, numSubDirs;
    # remove old paths
    if os.path.isdir(outRgbPath) :
        shutil.rmtree(outRgbPath)
    if os.path.isdir(outGtPath) :
        shutil.rmtree(outGtPath)
    # create fresh paths
    os.mkdir(outRgbPath)
    os.mkdir(outGtPath)

    # create all subdirectories
    for i in range(1, numSubDirs+1) :
        print 'creating subdirectories = {}'.format(i)
        os.mkdir(os.path.join(outRgbPath, str(i) ) )
        os.mkdir(os.path.join(outGtPath, str(i) ) )
# -----------------------------------------------------------------------

def saveSingleImage(im, gt) :
    global numImgPerDir, outRgbPath, outGtPath;
    tmp = np.random.randint(1, numSubDirs+1, 4); # for 4 images
    # original image
    numImgPerDir[tmp[0]] += 1
    tmpFileName = str(numImgPerDir[tmp[0]]) + '.png'
    cv2.imwrite(os.path.join(outRgbPath, str(tmp[0]), tmpFileName), im)
    cv2.imwrite(os.path.join(outGtPath, str(tmp[0]), tmpFileName), gt)
    # x-axis flipped image
    numImgPerDir[tmp[1]] += 1
    tmpFileName = str(numImgPerDir[tmp[1]]) + '.png'
    cv2.imwrite(os.path.join(outRgbPath, str(tmp[1]), tmpFileName), cv2.flip(im, 0))
    cv2.imwrite(os.path.join(outGtPath, str(tmp[1]), tmpFileName), cv2.flip(gt, 0))
    # y-axis flipped image
    numImgPerDir[tmp[2]] += 1
    tmpFileName = str(numImgPerDir[tmp[2]]) + '.png'
    cv2.imwrite(os.path.join(outRgbPath, str(tmp[2]), tmpFileName), cv2.flip(im, 1))
    cv2.imwrite(os.path.join(outGtPath, str(tmp[2]), tmpFileName), cv2.flip(gt, 1))
    # 180 degree rotated image
    numImgPerDir[tmp[3]] += 1
    tmpFileName = str(numImgPerDir[tmp[3]]) + '.png'
    cv2.imwrite(os.path.join(outRgbPath, str(tmp[3]), tmpFileName), cv2.flip(im, -1))
    cv2.imwrite(os.path.join(outGtPath, str(tmp[3]), tmpFileName), cv2.flip(gt, -1))
# -----------------------------------------------------------------------

def extractSubImages() :
    global imgList, inRgbPath, inGtPath;
    global numSubDirs, stepSize, imgSize;
    countTotalImg = 0;
    for i in range(len(imgList)) :
        im = cv2.imread(os.path.join(basePath, inRgbPath, imgList[i]),
                                        cv2.IMREAD_COLOR)
        gt = cv2.imread(os.path.join(basePath, inGtPath, imgList[i]),
                                        cv2.IMREAD_GRAYSCALE)
        rowSize, colSize = gt.shape
#        print rowSize, colSize;
        rowSize -= 1 # 0-indexing
        colSize -= 1 # 0-indexing
        i, j = 0, 0;
        rlim = rowSize - imgSize + 1;
        clim = colSize - imgSize + 1;
        while (i <= rlim) :
            j = 0;
            while (j <= clim) :
                countTotalImg += 1
                print countTotalImg
                im_sub = im[i:i+imgSize, j:j+imgSize, :]
                gt_sub = gt[i:i+imgSize, j:j+imgSize]
                saveSingleImage(im_sub, gt_sub)
                j += stepSize
            j-= stepSize
            if (j < clim) :
                countTotalImg += 1
                print countTotalImg
                j = clim;
                im_sub = im[i:i+imgSize, j:j+imgSize, :]
                gt_sub = gt[i:i+imgSize, j:j+imgSize]
                saveSingleImage(im_sub, gt_sub)
            i += stepSize
        i -= stepSize
        if (i < rlim) :
            i = rlim;
            j = 0;
            while (j <= clim) :
                countTotalImg += 1
                print countTotalImg
                im_sub = im[i:i+imgSize, j:j+imgSize, :]
                gt_sub = gt[i:i+imgSize, j:j+imgSize]
                saveSingleImage(im_sub, gt_sub)
                j += stepSize
            j-= stepSize
            if (j < clim) :
                countTotalImg += 1
                print countTotalImg
                j = clim;
                im_sub = im[i:i+imgSize, j:j+imgSize, :]
                gt_sub = gt[i:i+imgSize, j:j+imgSize]
                saveSingleImage(im_sub, gt_sub)

# -----------------------------------------------------------------------
if __name__ == "__main__" :
    checkExistence()
    processDirs()
    extractSubImages()
