clear all; close all; clc;

% source
% @article{bendig2015,
% title = "Combining UAV-based plant height from crop surface models, visible, and near infrared vegetation indices for biomass monitoring in barley",
% journal = "International Journal of Applied Earth Observation and Geoinformation",
% volume = "39",
% number = "",
% pages = "79 - 87",
% year = "2015",
% note = "",
% issn = "0303-2434",
% doi = "http://dx.doi.org/10.1016/j.jag.2015.02.012",
% url = "http://www.sciencedirect.com/science/article/pii/S0303243415000446",
% author = "Juliane Bendig and Kang Yu and Helge Aasen and Andreas Bolten and Simon Bennertz and Janis Broscheit and Martin L. Gnyp and Georg Bareth",
% }

basePath = '../../data/03_biomass';
trainPath = 'train';
testPath = 'test';
trainOrthoPath = 'ortho';
testOrthoPath = 'ortho';
trainDemPath = 'dem';
testDemPath = 'dem';

trainPath = fullfile(basePath, trainPath);
testPath = fullfile(basePath, testPath);
trainOrthoPath = fullfile(trainPath, trainOrthoPath);
testOrthoPath = fullfile(testPath, testOrthoPath);
trainDemPath = fullfile(trainPath, trainDemPath);
testDemPath = fullfile(testPath, testDemPath);

numTrainFiles = length(dir(fullfile(trainOrthoPath, '*.mat')));
numTestFiles = length(dir(fullfile(testOrthoPath, '*.mat')));

% load pre-computed features
load features.mat;
%features_list = {};
features_list = {'osavi'};%, 'mgrvi', 'rgbvi'};
%features_list = {'ndvi', 'savi', 'osavi', 'msavi', 'grvi', 'mgrvi', 'rgbvi'};

trainData = single(zeros(numTrainFiles, length(features_list)+1));
testData = single(zeros(numTestFiles, length(features_list)+1));

%count = 0;
for i = 1:length(features_list)
    fprintf('train feature = %s\n', features_list{i});
    v = matlab.lang.makeValidName(features_list{i});
    eval(['trainData(:,i) = features.train.', v, ';']);
%    count = count + 1;
%    eval(['trainData(:,count) = features.train.', v, ';']);
%    count = count + 1;
%    trainData(:,count) = bsxfun(@power, trainData(:,count-1), 2);
end
count = 0;
for i = 1:length(features_list)
    fprintf('test feature = %s\n', features_list{i});
    v = matlab.lang.makeValidName(features_list{i});
    eval(['testData(:,i) = features.test.', v, ';']);    
%    count = count + 1;
%    eval(['testData(:,count) = features.test.', v, ';']);
%    count = count + 1;
%    testData(:,count) = bsxfun(@power, testData(:,count-1), 2);
end
clear features;

% compute new features
for i = 1:numTrainFiles
    load(fullfile(trainDemPath, [num2str(i), '.mat']));
    dem = image;
    clear image;
    dem = bsxfun(@minus, dem, min(dem(:)));
    if ~isempty(features_list)
        trainData(i, end) = sum(dem(:));
    else
        trainData(i, 1) = sum(dem(:));
    end
end
%trainData(:,end) = bsxfun(@power, trainData(:,end-1), 2);

for i = 1:numTestFiles
    load(fullfile(testDemPath, [num2str(i), '.mat']));
    dem = image;
    clear image;
    dem = bsxfun(@minus, dem, min(dem(:)));
    if ~isempty(features_list)
        testData(i, end) = sum(dem(:));
    else
        testData(i, 1) = sum(dem(:));
    end
end
%testData(:,end) = bsxfun(@power, testData(:,end-1), 2);

% normalize features
valMax = max(trainData);
trainData = bsxfun(@rdivide, trainData, valMax);
testData = bsxfun(@rdivide, testData, valMax);
% add intercept
%trainData = [ones(size(trainData,1),1), trainData];
%testData = [ones(size(testData,1),1), testData];

load(fullfile(trainPath, 'countTrain.mat'));
trainCounts = single(counts');
load(fullfile(testPath, 'countTest.mat'));
testCounts = single(counts');
save('data_method_03.mat', 'trainData', 'trainCounts', 'testData', 'testCounts');

clear all; close all; clc;
system('Rscript getModelMatrix_03.R');
load('data_method_03.mat');
%trainData = [trainData; testData];
%trainCounts = [trainCounts; testCounts];
beta = mvregress(trainData, trainCounts, 'algorithm', 'cwls');
predictions = testData * beta;
counts = testCounts;
save('method_03.mat', 'predictions', 'counts');
