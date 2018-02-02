clear all; close all; clc;

MAX_16_BIT = 2^16 - 1;
SIZE_MAT = 96*2; % row and column sizes of each matrices

%typeInput = 'osavi';
typeInput = 'all';
%typeInput = 'rgb';
basePath = '../../data/03_biomass/test';
orthoPath = 'ortho';
demPath = 'dem';

orthoPath = fullfile(basePath, orthoPath);
demPath = fullfile(basePath, demPath);

numFiles = length(dir(fullfile(orthoPath, '*.mat')));

if strcmp(typeInput, 'rgb')
    numImageChn_1 = 3;
elseif strcmp(typeInput, 'all')
    numImageChn_1 = 5;
elseif strcmp(typeInput, 'osavi')
    numImageChn_1 = 1;
else
    error('Invalid input type.');
end

im_sample = single(zeros(SIZE_MAT, SIZE_MAT, numImageChn_1)); % memory for sample rgb
dem_sample = single(zeros(SIZE_MAT)); % memory for sample dem

for i = 1:numFiles
    fprintf('plot = %d\n', i);
    load(fullfile(demPath, [num2str(i), '.mat']));
    dem = image;
    load(fullfile(orthoPath, [num2str(i), '.mat']));
    if strcmp(typeInput, 'rgb') || strcmp(typeInput, 'all')
        im = single(image(:,:,1:numImageChn_1));
    elseif strcmp(typeInput, 'osavi')
        im = single(image(:,:,1:5));
    else
        error('Invalid input type.');
    end
    clear image;
    % preprocess dem
    dem(dem<0) = 0;
    dem = bsxfun(@minus, dem, min(dem(:)));
    % normalize image
    im = bsxfun(@rdivide, im, MAX_16_BIT);
    if strcmp(typeInput, 'osavi')
        im_out = bsxfun(@plus, im(:,:,4), im(:,:,3));
        im_out = bsxfun(@plus, im_out, 0.16);
        im_out = bsxfun(@rdivide, ...
            bsxfun(@minus, im(:,:,4), im(:,:,3)), im_out);
        im_out = bsxfun(@times, im_out, 1.16); 
        im = im_out;
        clear im_out;
    end
    % ------- preprocessing finished ------------%
    % save original copy first
    im_sample(1:SIZE_MAT/2, :, :) = im(:, 1:SIZE_MAT, :);
    im_sample(SIZE_MAT/2 + 1:end, :, :) = im(:, SIZE_MAT+1:end, :);
    dem_sample(1:SIZE_MAT/2, :) = dem(:, 1:SIZE_MAT);
    dem_sample(SIZE_MAT/2 + 1:end, :) = dem(:, SIZE_MAT+1:end);

    save(fullfile(orthoPath, [num2str(i), '.mat']), 'im_sample');
    save(fullfile(demPath, [num2str(i), '.mat']), 'dem_sample');
end
