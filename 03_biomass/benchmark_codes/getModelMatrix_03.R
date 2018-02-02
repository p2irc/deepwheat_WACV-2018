rm(list=ls());
setwd('./');

library(R.matlab);

numMethod = 3;

dataFileName = paste('data_method_0', as.character(numMethod), '.mat', sep="");
tmp = readMat(dataFileName);
# get data
trainData = tmp$trainData;
trainCounts = tmp$trainCounts;
testData = tmp$testData;
testCounts = tmp$testCounts;
rm(tmp);

trainMatrix = model.matrix( ~ poly(trainData, degree=2, raw=TRUE) );
testMatrix = model.matrix( ~ poly(testData, degree=2, raw=TRUE) )
writeMat(dataFileName, trainCounts=trainCounts, testCounts=testCounts, 
         trainData=trainMatrix, testData=testMatrix)