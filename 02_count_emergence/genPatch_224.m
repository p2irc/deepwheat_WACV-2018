function genPatch_224()
%clear all; close all; clc;

SIZE_IMG = 224;

% process training data
basePath = '../../data/02_count_emergence/train';
inRgbPath = 'rgb_patch';
outRgbPath = 'rgb_patch_resize_224';

inRgbPath = fullfile(basePath, inRgbPath);
outRgbPath = fullfile(basePath, outRgbPath);

if isdir(outRgbPath)
    assert(rmdir(outRgbPath, 's'), ...
        'Cannot remove old data directory\n %s', outRgbPath);
end
assert(mkdir(outRgbPath), 'Cannot create new data directory\n %s', outRgbPath);

processImages(inRgbPath, outRgbPath, SIZE_IMG);

% process test data
basePath = '../../data/02_count_emergence/test';
inRgbPath = 'rgb_patch';
outRgbPath = 'rgb_patch_resize_224';

inRgbPath = fullfile(basePath, inRgbPath);
outRgbPath = fullfile(basePath, outRgbPath);

if isdir(outRgbPath)
    assert(rmdir(outRgbPath, 's'), ...
        'Cannot remove old data directory\n %s', outRgbPath);
end
assert(mkdir(outRgbPath), 'Cannot create new data directory\n %s', outRgbPath);

processImages(inRgbPath, outRgbPath, SIZE_IMG);

end

function processImages(inRgbPath, outRgbPath, SIZE_IMG)

imgList = dir(fullfile(inRgbPath, '*.png'));

for i = 1:length(imgList)
    fprintf('%d\n', i);
    im = imread(fullfile(inRgbPath, imgList(i).name));
    im_gray = rgb2gray(im);
    for r = 1:size(im_gray,1)
        if numel(find(im_gray(r,:)>0)) > 0
            rmin = r; break;
        end
    end
    for r = size(im_gray,1):-1:1
        if numel(find(im_gray(r,:)>0)) > 0
            rmax = r; break;
        end
    end    
    for c = 1:size(im_gray,2)
        if numel(find(im_gray(:,c)>0)) > 0
            cmin = c; break;
        end
    end
    for c = size(im_gray,1):-1:1
        if numel(find(im_gray(:,c)>0)) > 0
            cmax = c; break;
        end
    end 
    im = im(rmin:rmax, cmin:cmax, :);
    %imshow(im);
    dr = rmax-rmin+1;
    dc = cmax-cmin+1;
    if dr>=dc
        numRows = SIZE_IMG;
        numCols = floor((SIZE_IMG/dr)*dc);
    else
        numCols = SIZE_IMG;
        numRows = floor((SIZE_IMG/dc)*dr);    
    end
    
    im_out = uint8(zeros(SIZE_IMG, SIZE_IMG, 3));
    if (numRows>0) && (numCols>0)
        im = imresize(im, [numRows, numCols]);
        im_out(1:numRows, 1:numCols, :) = im;
    end
    imwrite(im_out, fullfile(outRgbPath, imgList(i).name));
end
end
