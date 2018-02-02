rm(list=ls());
setwd('~/Desktop/my_work/codes/898_phenowheat/03_biomass/benchmark_codes/');

library(R.matlab);
library(pls);

numMethod = 2;

dataFileName = paste('data_method_0', as.character(numMethod), '.mat', sep="");
outFileName = paste('method_0', as.character(numMethod), '.mat', sep="");

tmp = readMat(dataFileName);
# get data
trainData = tmp$trainData;
trainCounts = tmp$trainCounts;
testData = tmp$testData;
testCounts = tmp$testCounts;
rm(tmp);

plsFit = plsr(trainCounts ~ trainData);
predictions = drop(predict(plsFit, testData, ncomp=1));

writeMat(outFileName, predictions=predictions, counts=drop(testCounts))