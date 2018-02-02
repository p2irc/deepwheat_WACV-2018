import numpy as np
import scipy.io
import os

basePath = '../../data/03_biomass/test';

#typeInput = 'all';
#typeInput = 'rgb';
#typeInput = 'osavi';
typeInput = 'dem';

if typeInput == 'all' :
	resPath = 'result_full';
elif typeInput == 'rgb' :
	resPath = 'result_rgb';
elif typeInput == 'osavi' :
	resPath = 'result_osavi';
elif typeInput == 'dem' :
	resPath = 'result_dem';
else :
	print 'Invalid input type specification.'
	exit()

resFilePrefix = 'countTest_epoch_';
gtFileName = 'countTest.mat';

resPath = os.path.join(basePath, resPath, resFilePrefix);
resPath = os.path.join(basePath, resPath, resFilePrefix);
gtFileName = os.path.join(basePath, gtFileName);

gt = scipy.io.loadmat(gtFileName)['counts'][0];

if typeInput == 'dem' :
# print dem results first
	for epoch in range(50, 60, 10) :
	    tmpResPath = resPath + str(epoch) + '.mat';
	    out = scipy.io.loadmat(tmpResPath)['counts'];
	    out = np.squeeze(out).astype(gt.dtype);
	    dif_ = out - gt;
	    overest = dif_[dif_>0].sum()
	    underest = abs(dif_[dif_<0]).sum()
	    total_count = gt.sum();

	    print 'Dem Epoch = {}'.format(epoch);
	    print 'Number of test samples = {}'.format(len(gt));
	    print 'Total test count = {}'.format(total_count);
	    print 'Mean absolute difference = {}'.format(abs(dif_).mean());
	    print 'Std absolute difference = {}'.format(abs(dif_).std());
	    print 'Overestimate (%) = {} %'.format(float(overest*100)/total_count);
	    print 'Underestimate (%) = {} %'.format(float(underest*100)/total_count);
	    print 'Deviation (%) = {} %'.format(float((overest + underest)*100)/total_count);
	    print '---------------------------------------------------'

else :
	# print full results second
	for epoch in range(50, 60, 10) :
	    tmpResPath = resPath + str(epoch) + '.mat';
	    out = scipy.io.loadmat(tmpResPath)['counts'];
	    out = np.squeeze(out).astype(gt.dtype);
	    dif_ = out - gt;
	    overest = dif_[dif_>0].sum()
	    underest = abs(dif_[dif_<0]).sum()
	    total_count = gt.sum();

	    print 'Full Epoch = {}'.format(epoch);
	    print 'Number of test samples = {}'.format(len(gt));
	    print 'Total test count = {}'.format(total_count);
	    print 'Mean absolute difference = {}'.format(abs(dif_).mean());
	    print 'Std absolute difference = {}'.format(abs(dif_).std());
	    print 'Overestimate (%) = {} %'.format(float(overest*100)/total_count);
	    print 'Underestimate (%) = {} %'.format(float(underest*100)/total_count);
	    print 'Deviation (%) = {} %'.format(float((overest + underest)*100)/total_count);
	    print '---------------------------------------------------'
