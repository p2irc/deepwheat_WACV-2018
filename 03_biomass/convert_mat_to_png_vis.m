clear all; close all; clc;

offset_ = 10; % 10 pixels along both row and column
row_images = 4; % images in a single row
basePath = '../../data/03_biomass';
trainPath = 'train';
testPath = 'test';
inPathTrain = 'ortho';
inPathTest = 'ortho';
inPathTrainDem = 'dem';
inPathTestDem = 'dem';
outPathTrain = 'rgb_vis';
outPathTest = 'rgb_vis';

inPathTrain = fullfile(basePath, trainPath, inPathTrain);
inPathTest = fullfile(basePath, testPath, inPathTest);
inPathTrainDem = fullfile(basePath, trainPath, inPathTrainDem);
inPathTestDem = fullfile(basePath, testPath, inPathTestDem);
outPathTrain = fullfile(basePath, trainPath, outPathTrain);
outPathTest = fullfile(basePath, testPath, outPathTest);

if isdir(outPathTrain)
    assert(rmdir(outPathTrain, 's'), ...
        'Cannot remove old train RGB-PNG directory\n %s', outPathTrain);
end
assert(mkdir(outPathTrain), 'Cannot create new train RGB-PNG directory\n %s', outPathTrain);
if isdir(outPathTest)
    assert(rmdir(outPathTest, 's'), ...
        'Cannot remove old test RGB-PNG directory\n %s', outPathTest);
end
assert(mkdir(outPathTest), 'Cannot create new test RGB-PNG directory\n %s', outPathTest);

numFiles = length(dir(fullfile(inPathTrain, '*.mat')));
val_max = intmax('uint16');
im_stack = [];
dem_stack = [];
for i = 1:numFiles
    fprintf('train plot = %d\n', i);
    load(fullfile(inPathTrain, [num2str(i), '.mat']));
    im = cat(3, image(:,:,3), image(:,:,2), image(:,:,1));
    load(fullfile(inPathTrainDem, [num2str(i), '.mat']));
    dem = image; 
    dem = bsxfun(@minus, dem, min(dem(:)));
    dem = mat2gray(dem);
    im_stack = [im_stack; [im, val_max * uint16(ones(size(im,1), offset_, 3))]];
    dem_stack = [dem_stack; [dem, single(ones(size(im,1), offset_))]];
    im_stack = [im_stack; val_max * uint16(ones(offset_, size(im_stack, 2), 3))];
    dem_stack = [dem_stack; single(ones(offset_, size(im_stack, 2)))];
    imwrite(im, fullfile(outPathTrain, [num2str(i), '.png']));
end

col_new = row_images * size(dem_stack, 2);
row_new = size(im_stack, 1) * size(im_stack, 2) / col_new;
im_big = [];
dem_big = [];
for i = 1:row_new:size(im_stack,1)
    im_big = [im_big, im_stack(i:i+row_new-1, :,:)];
    dem_big = [dem_big, dem_stack(i:i+row_new-1, :)];
end
imwrite(im_big, fullfile(basePath, trainPath, 'rgb_all.png'));
imwrite(dem_big, fullfile(basePath, trainPath, 'dem_all.png'));

numFiles = length(dir(fullfile(inPathTest, '*.mat')));
im_stack = [];
dem_stack = [];
for i = 1:numFiles
    fprintf('test plot = %d\n', i);
    load(fullfile(inPathTest, [num2str(i), '.mat']));
    im = cat(3, image(:,:,3), image(:,:,2), image(:,:,1));
    load(fullfile(inPathTestDem, [num2str(i), '.mat']));
    dem = image; clear image;  
    dem = bsxfun(@minus, dem, min(dem(:)));
    dem = mat2gray(dem);
    im_stack = [im_stack; [im, val_max * uint16(ones(size(im,1), offset_, 3))]];
    dem_stack = [dem_stack; [dem, single(ones(size(im,1), offset_))]];
    im_stack = [im_stack; val_max * uint16(ones(offset_, size(im_stack, 2), 3))];
    dem_stack = [dem_stack; single(ones(offset_, size(im_stack, 2)))];
    imwrite(im, fullfile(outPathTrain, [num2str(i), '.png']));
end

col_new = row_images * size(dem_stack, 2);
row_new = size(im_stack, 1) * size(im_stack, 2) / col_new;
im_big = [];
dem_big = [];
for i = 1:row_new:size(im_stack,1)
    im_big = [im_big, im_stack(i:i+row_new-1, :,:)];
    dem_big = [dem_big, dem_stack(i:i+row_new-1, :)];
end
imwrite(im_big, fullfile(basePath, testPath, 'rgb_all.png'));
imwrite(dem_big, fullfile(basePath, testPath, 'dem_all.png'));
