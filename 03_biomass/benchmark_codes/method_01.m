clear all; close all; clc;

% source
% @article{laurin2016,
% title = "Above ground biomass and tree species richness estimation with airborne lidar in tropical Ghana forests",
% journal = "International Journal of Applied Earth Observation and Geoinformation",
% volume = "52",
% number = "",
% pages = "371 - 379",
% year = "2016",
% note = "",
% issn = "0303-2434",
% doi = "http://dx.doi.org/10.1016/j.jag.2016.07.008",
% url = "http://www.sciencedirect.com/science/article/pii/S0303243416301155",
% author = "Gaia Vaglio Laurin and Nicola Puletti and Qi Chen and Piermaria Corona and Dario Papale and Riccardo Valentini",
% }

addpath(genpath('ARESLab'));
numRows = 96;
numCols = 384;
sizePatch = 96;
percentileVal = 10; % 10% for this method
basePath = '../../data/03_biomass';
trainPath = 'train';
testPath = 'test';
trainDemPath = 'dem';
testDemPath = 'dem';

trainPath = fullfile(basePath, trainPath);
testPath = fullfile(basePath, testPath);
trainDemPath = fullfile(trainPath, trainDemPath);
testDemPath = fullfile(testPath, testDemPath);

numTrainFiles = length(dir(fullfile(trainDemPath, '*.mat')));
numTestFiles = length(dir(fullfile(testDemPath, '*.mat')));

numBlockRows = numRows/sizePatch;
numBlockCols = numCols/sizePatch;
%dimFeatures = 5;
dimFeatures = 5 + numBlockRows * numBlockCols;
trainData = single(zeros(numTrainFiles, dimFeatures));
testData = single(zeros(numTestFiles, dimFeatures));

% compute new features
for i = 1:numTrainFiles
    fprintf('train file = %d\n', i);
	count = 0;
    load(fullfile(trainDemPath, [num2str(i), '.mat']));
    dem = image;
    clear image;
    dem = bsxfun(@minus, dem, min(dem(:)));
	for r = 1:sizePatch:numRows
		for c = 1:sizePatch:numCols
			count = count + 1;
			blk = dem(r:r+sizePatch-1, c:c+sizePatch-1);
			trainData(i,count) = sum(blk(:));
		end
	end
	dem = dem(:);
	count = count + 1;
	trainData(i, count) = mean(dem);
	count = count + 1;
	trainData(i, count) = std(dem);
	count = count + 1;
	trainData(i, count) = skewness(dem);
	count = count + 1;
	trainData(i, count) = kurtosis(dem);
	count = count + 1;
	trainData(i, count) = prctile(dem, percentileVal);	
end

for i = 1:numTestFiles
    fprintf('test file = %d\n', i);
	count = 0;
    load(fullfile(testDemPath, [num2str(i), '.mat']));
    dem = image;
    clear image;
    dem = bsxfun(@minus, dem, min(dem(:)));
	for r = 1:sizePatch:numBlockRows
		for c = 1:sizePatch:numBlockCols
			count = count + 1;
			blk = dem(r:r+sizePatch-1, c:c+sizePatch-1);
			testData(i,count) = sum(blk(:));
		end
	end
	dem = dem(:);
	count = count + 1;
	testData(i, count) = mean(dem);
	count = count + 1;
	testData(i, count) = std(dem);
	count = count + 1;
	testData(i, count) = skewness(dem);
	count = count + 1;
	testData(i, count) = kurtosis(dem);
	count = count + 1;
	testData(i, count) = prctile(dem, percentileVal);	
end

% normalize features
valMax = max(trainData);
trainData = bsxfun(@rdivide, trainData, valMax);
testData = bsxfun(@rdivide, testData, valMax);

load(fullfile(trainPath, 'countTrain.mat'));
trainCounts = single(counts');
load(fullfile(testPath, 'countTest.mat'));
testCounts = single(counts');
%save('data_method_01.mat', 'trainData', 'trainCounts', 'testData', 'testCounts');
params = aresparams2('maxFuncs', 15, 'maxInteractions', 1, 'useMinSpan', 3);
[model, ~, ~] = aresbuild(trainData, trainCounts, params);
[predictions, ~] = arespredict(model, testData);
counts = testCounts;
save('method_01.mat', 'predictions', 'counts');
restoredefaultpath;
