clear all; close all; clc;

MAX_16_BIT = 2^16 - 1;
numSupPix = 500; % number of superpixels
thNumChanges = 50; % minimum 10 changes needed to define a new sample
numSamplesPerPlot = 750; % generate 750 samples from each plot image
SIZE_MAT = 96*2; % row and column sizes of each matrices
std_count_aug = 0.01; % std for augmented counts 1%

%typeInput = 'osavi';
%typeInput = 'all';
typeInput = 'rgb';
basePath = '../../data/03_biomass/train';
inOrthoPath = 'ortho';
inDemPath = 'dem';
inGtFileName = 'countTrain.mat';
outOrthoPath = 'ortho_aug';
outDemPath = 'dem_aug';
outGtFileName = 'countTrain_aug.mat';

inOrthoPath = fullfile(basePath, inOrthoPath);
inDemPath = fullfile(basePath, inDemPath);
outOrthoPath = fullfile(basePath, outOrthoPath);
outDemPath = fullfile(basePath, outDemPath);

if isdir(outOrthoPath)
    assert(rmdir(outOrthoPath, 's'), ...
        'Cannot remove old ortho directory\n %s', outOrthoPath);
end
assert(mkdir(outOrthoPath), 'Cannot create new ortho directory\n %s', outOrthoPath);
if isdir(outDemPath)
    assert(rmdir(outDemPath, 's'), ...
        'Cannot remove old dem directory\n %s', outDemPath);
end
assert(mkdir(outDemPath), 'Cannot create new dem directory\n %s', outDemPath);


load(fullfile(basePath, inGtFileName)); % load count file
counts = single(counts);

numFiles = length(dir(fullfile(inOrthoPath, '*.mat')));

if strcmp(typeInput, 'rgb')
    numImageChn_1 = 3;
elseif strcmp(typeInput, 'all')
    numImageChn_1 = 5;
elseif strcmp(typeInput, 'osavi')
    numImageChn_1 = 1;
    max_osavi = 0; % absolute maximum in case of osavi input
    max_osavi_file = 'max_osavi.mat';
else
    error('Invalid input type.');
end

im_sample = single(zeros(SIZE_MAT, SIZE_MAT, numImageChn_1)); % memory for sample rgb
dem_sample = single(zeros(SIZE_MAT)); % memory for sample dem
countSamplesTotal = 0;
counts_aug = [];

for i = 1:numFiles
    fprintf('plot = %d\n', i);
    load(fullfile(inDemPath, [num2str(i), '.mat']));
    dem = image;
    load(fullfile(inOrthoPath, [num2str(i), '.mat']));
    if strcmp(typeInput, 'rgb') || strcmp(typeInput, 'all')
        im = single(image(:,:,1:numImageChn_1));
    elseif strcmp(typeInput, 'osavi')
        im = single(image(:,:,1:5));
    else
        error('Invalid input type.');
    end
    
    % preprocess dem
    dem(dem<0) = 0;
    dem = bsxfun(@minus, dem, min(dem(:)));
    % normalize image
    im = bsxfun(@rdivide, im, MAX_16_BIT);
    im_gray = rgb2gray(cat(3, im(:,:,3), im(:,:,2), im(:,:,1))); % for superpixel
    if strcmp(typeInput, 'osavi')
        im_out = bsxfun(@plus, im(:,:,4), im(:,:,3));
        im_out = bsxfun(@plus, im_out, 0.16);
        im_out = bsxfun(@rdivide, ...
            bsxfun(@minus, im(:,:,4), im(:,:,3)), im_out);
        im_out = bsxfun(@times, im_out, 1.16); 
        im = im_out;
        clear im_out;
        tmp = max(abs(im(:)));
        if max_osavi < tmp
            max_osavi = tmp;
        end
    end
    % ------- preprocessing finished ------------%
    % save original copy first
    im_sample(1:SIZE_MAT/2, :, :) = im(:, 1:SIZE_MAT, :);
    im_sample(SIZE_MAT/2 + 1:end, :, :) = im(:, SIZE_MAT+1:end, :);
    dem_sample(1:SIZE_MAT/2, :) = dem(:, 1:SIZE_MAT);
    dem_sample(SIZE_MAT/2 + 1:end, :) = dem(:, SIZE_MAT+1:end);
    counts_aug = [counts_aug, counts(i)];
    countSamplesTotal = countSamplesTotal + 1;    
    fprintf('plot = %d, sample = 1\n', i);
    save(fullfile(outOrthoPath, [num2str(countSamplesTotal), '.mat']), 'im_sample');
    save(fullfile(outDemPath, [num2str(countSamplesTotal), '.mat']), 'dem_sample');
        
    im_tmp = im_sample;
    dem_tmp = dem_sample;
    % save vertically flipped mat files
    counts_aug = [counts_aug, normrnd(counts(i), std_count_aug * counts(i))];
    im_sample = flip(im_tmp);
    dem_sample = flip(dem_tmp);
    countSamplesTotal = countSamplesTotal + 1;    
    save(fullfile(outOrthoPath, [num2str(countSamplesTotal), '.mat']), 'im_sample');
    save(fullfile(outDemPath, [num2str(countSamplesTotal), '.mat']), 'dem_sample');

    % save 180 degree rotated mat files
    counts_aug = [counts_aug, normrnd(counts(i), std_count_aug * counts(i))];
    im_sample = fliplr(im_sample);
    dem_sample = fliplr(dem_sample);
    countSamplesTotal = countSamplesTotal + 1;    
    save(fullfile(outOrthoPath, [num2str(countSamplesTotal), '.mat']), 'im_sample');
    save(fullfile(outDemPath, [num2str(countSamplesTotal), '.mat']), 'dem_sample');        

    % save left-right flipped mat files
    counts_aug = [counts_aug, normrnd(counts(i), std_count_aug * counts(i))];
    im_sample = fliplr(im_tmp);
    dem_sample = fliplr(dem_tmp);
    countSamplesTotal = countSamplesTotal + 1;    
    save(fullfile(outOrthoPath, [num2str(countSamplesTotal), '.mat']), 'im_sample');
    save(fullfile(outDemPath, [num2str(countSamplesTotal), '.mat']), 'dem_sample');
    
    % ------------------------------------------------------------------- %
    % get superpixels from gray image
%    im_gray = rgb2gray(cat(3, im(:,:,3), im(:,:,2), im(:,:,1)));
    [L, numLabels] = superpixels(im_gray, numSupPix);
%    BW = boundarymask(L);
%    figure; imshow(imoverlay(im, BW, 'yellow'), 'InitialMagnification', 200);
%    hold on;
    linIdAll = label2idx(L); % get linear ids for all labels
    % collect box coordinates and box means
    boxMean = single(zeros(1, numLabels));
    boxCoords = struct();
    for label = 1:numLabels
        linId = linIdAll{label};
        [tmpRows, tmpCols] = ind2sub(size(im_gray), linId);
        boxCoords(label).rmin = min(tmpRows);
        boxCoords(label).rmax = max(tmpRows);
        boxCoords(label).cmin = min(tmpCols);
        boxCoords(label).cmax = max(tmpCols);
        boxMean(label) = mean(im_gray(linId));
%        rectangle('Position', [cmin, rmin, cmax-cmin+1, rmax-rmin+1], 'EdgeColor','r');
    end
    % sort superpixel coordinates
    [boxMean, indSorted] = sort(boxMean);
    boxCoords = boxCoords(indSorted);
%    bar(1:numLabels, boxMean);
    % generate random number of changes for each pseudo-sample
    numChanges = randi([thNumChanges, floor(numLabels/2)], 1, numSamplesPerPlot-1);
    for c = 1:length(numChanges)
        fprintf('plot = %d, sample = %d\n', i, c+1);
        im_tmp = im; % needed for modification
        dem_tmp = dem; % needed for modification
        chPostId = 2*randi(floor(numLabels/2), 1, numChanges(c));
        for pid = 1:length(chPostId)
            % get coordinates for consecutive superpixels
            rmin_pre = boxCoords(chPostId(pid)-1).rmin;
            cmin_pre = boxCoords(chPostId(pid)-1).cmin;
            rmax_pre = boxCoords(chPostId(pid)-1).rmax;
            cmax_pre = boxCoords(chPostId(pid)-1).cmax;
            rmin_post = boxCoords(chPostId(pid)).rmin;
            cmin_post = boxCoords(chPostId(pid)).cmin;
            rmax_post = boxCoords(chPostId(pid)).rmax;
            cmax_post = boxCoords(chPostId(pid)).cmax;
            dr_pre = rmax_pre - rmin_pre;
            dc_pre = cmax_pre - cmin_pre;
            dr_post = rmax_post - rmin_post;
            dc_post = cmax_post - cmin_post;
            % check the smaller box and swap it, if found
            if (dr_pre <= dr_post) && (dc_pre <= dc_post) % pre is smaller
                r_lb_pre = rmin_pre;
                r_ub_pre = rmax_pre;
                c_lb_pre = cmin_pre;
                c_ub_pre = cmax_pre;
                r_lb_post = rmin_post + floor((dr_post - dr_pre)/2); 
                r_ub_post = r_lb_post + dr_pre;
                c_lb_post = cmin_post + floor((dc_post - dc_pre)/2); 
                c_ub_post = c_lb_post + dc_pre;
            elseif (dr_pre > dr_post) && (dc_pre > dc_post) % pre is bigger
                r_lb_pre = rmin_pre + floor((dr_pre - dr_post)/2); 
                r_ub_pre = r_lb_pre + dr_post;
                c_lb_pre = cmin_pre + floor((dc_pre - dc_post)/2); 
                c_ub_pre = c_lb_pre + dc_post;
                r_lb_post = rmin_post;
                r_ub_post = rmax_post;
                c_lb_post = cmin_post;
                c_ub_post = cmax_post;                                
            elseif (dr_pre > dr_post) && (dc_pre <= dc_post)
                r_lb_pre = rmin_pre + floor((dr_pre - dr_post)/2); 
                r_ub_pre = r_lb_pre + dr_post;
                c_lb_pre = cmin_pre;
                c_ub_pre = cmax_pre;
                r_lb_post = rmin_post;
                r_ub_post = rmax_post;
                c_lb_post = cmin_post + floor((dc_post - dc_pre)/2);
                c_ub_post = c_lb_post + dc_pre;
            elseif (dr_pre <= dr_post) && (dc_pre > dc_post)
                r_lb_pre = rmin_pre; 
                r_ub_pre = rmax_pre;
                c_lb_pre = cmin_pre + floor((dc_pre - dc_post)/2);
                c_ub_pre = c_lb_pre + dc_post;
                r_lb_post = rmin_post + floor((dr_post - dr_pre)/2);
                r_ub_post = r_lb_post + dr_pre;
                c_lb_post = cmin_post;
                c_ub_post = cmax_post;                
            end
            im_tmp(r_lb_pre:r_ub_pre, c_lb_pre:c_ub_pre, :) = ...
                    im(r_lb_post:r_ub_post, c_lb_post:c_ub_post, :);  
            im_tmp(r_lb_post:r_ub_post, c_lb_post:c_ub_post, :) = ...
                    im(r_lb_pre:r_ub_pre, c_lb_pre:c_ub_pre, :);
            dem_tmp(r_lb_pre:r_ub_pre, c_lb_pre:c_ub_pre) = ...
                    dem(r_lb_post:r_ub_post, c_lb_post:c_ub_post);  
            dem_tmp(r_lb_post:r_ub_post, c_lb_post:c_ub_post) = ...
                    dem(r_lb_pre:r_ub_pre, c_lb_pre:c_ub_pre);                

        end
        im_sample(1:SIZE_MAT/2, :, :) = im_tmp(:, 1:SIZE_MAT, :);
        im_sample(SIZE_MAT/2 + 1:end, :, :) = im_tmp(:, SIZE_MAT+1:end, :);
        dem_sample(1:SIZE_MAT/2, :) = dem_tmp(:, 1:SIZE_MAT);
        dem_sample(SIZE_MAT/2 + 1:end, :) = dem_tmp(:, SIZE_MAT+1:end);
        
        % save mat files
        counts_aug = [counts_aug, normrnd(counts(i), std_count_aug * counts(i))];
        countSamplesTotal = countSamplesTotal + 1;    
        save(fullfile(outOrthoPath, [num2str(countSamplesTotal), '.mat']), 'im_sample');
        save(fullfile(outDemPath, [num2str(countSamplesTotal), '.mat']), 'dem_sample');
        
        im_tmp = im_sample;
        dem_tmp = dem_sample;
        % save vertically flipped mat files
        counts_aug = [counts_aug, normrnd(counts(i), std_count_aug * counts(i))];
        im_sample = flip(im_tmp);
        dem_sample = flip(dem_tmp);
        countSamplesTotal = countSamplesTotal + 1;    
        save(fullfile(outOrthoPath, [num2str(countSamplesTotal), '.mat']), 'im_sample');
        save(fullfile(outDemPath, [num2str(countSamplesTotal), '.mat']), 'dem_sample');
        
        % save 180 degree rotated mat files
        counts_aug = [counts_aug, normrnd(counts(i), std_count_aug * counts(i))];
        im_sample = fliplr(im_sample);
        dem_sample = fliplr(dem_sample);
        countSamplesTotal = countSamplesTotal + 1;    
        save(fullfile(outOrthoPath, [num2str(countSamplesTotal), '.mat']), 'im_sample');
        save(fullfile(outDemPath, [num2str(countSamplesTotal), '.mat']), 'dem_sample');        
       
        % save left-right flipped mat files
        counts_aug = [counts_aug, normrnd(counts(i), std_count_aug * counts(i))];
        im_sample = fliplr(im_tmp);
        dem_sample = fliplr(dem_tmp);
        countSamplesTotal = countSamplesTotal + 1;    
        save(fullfile(outOrthoPath, [num2str(countSamplesTotal), '.mat']), 'im_sample');
        save(fullfile(outDemPath, [num2str(countSamplesTotal), '.mat']), 'dem_sample');        
               
    end
end

% save augmented counts
counts = [];
counts = counts_aug;
save(fullfile(basePath, outGtFileName), 'counts'); 
if strcmp(typeInput, 'osavi')
    save(fullfile(basePath, max_osavi_file), 'max_osavi'); 
end
%system('shutdown now');
