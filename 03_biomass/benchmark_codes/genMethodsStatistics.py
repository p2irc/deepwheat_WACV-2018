import numpy as np
import scipy.io
import os

def printStats(fileName) :
    out = scipy.io.loadmat(fileName)['predictions'];
    gt = scipy.io.loadmat(fileName)['counts'];
    gt = np.squeeze(gt);
    out = np.squeeze(out).astype(gt.dtype);
#    print out, gt;
    dif_ = out - gt;
    overest = dif_[dif_>0].sum()
    underest = abs(dif_[dif_<0]).sum()
    total_count = gt.sum();

    print 'Number of test samples = {}'.format(len(gt));
    print 'Total test count = {}'.format(total_count);
    print 'Mean absolute difference = {}'.format(abs(dif_).mean());
    print 'Std absolute difference = {}'.format(abs(dif_).std());
    print 'Overestimate (%) = {} %'.format(float(overest*100)/total_count);
    print 'Underestimate (%) = {} %'.format(float(underest*100)/total_count);
    print 'Deviation (%) = {} %'.format(float((overest + underest)*100)/total_count);
    print '---------------------------------------------------'

printStats('method_01.mat');
printStats('method_02.mat');
printStats('method_03.mat');
