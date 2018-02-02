import numpy as np
import scipy.io
import os

basePath = '../../data/02_count_emergence/test';
gtFileName = 'count_gt.mat';
outFileName = 'count_test_output.mat';

gtFileName = os.path.join(basePath, gtFileName);
outFileName = os.path.join(basePath, outFileName);

gt = scipy.io.loadmat(gtFileName)['counts'][0];
out = scipy.io.loadmat(outFileName)['counts'];
out = np.round(np.squeeze(out)).astype(gt.dtype);
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


