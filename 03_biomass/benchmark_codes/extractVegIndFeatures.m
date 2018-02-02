function extractVegIndFeatures()
%clear all; close all; clc;

% this file extracts the following Vegetation Index (VI) features and save
% altogether in features.mat file. Note that, the indices are approximated
% based on the band data available in our dataset.
% Note that, <struct_name> indicates the structure name for that VI in 
% this code.
% (1) <dvi> Difference VI (DVI) = NIR - Red
% (2) <ndvi> Normalized DVI (NDVI) = DVI/(NIR + Red)
% (3) <sr> Simple Ratio (SR) = NIR/Red
% (4) <rendvi> Red edge NDVI (ReNDVI) = (NIR - Re)/(NIR + Re)
% (5) <arvi> Atmospherically Resistant VI (ARVI) = (NIR-RB)/(NIR+RB)
%       where, RB = R-Gamma*(B-R); Gamma depends on the aerosol type
%       a good value is Gamma = 1 when the aerosol model in not available
%       So, we use, RB = R-(B-R) = 2R - B
% (6) <ari2> Anthocyanin Reflectance Index 2 (ARI2) = NIR(1/G - 1/Re)
% (7) <pri> Photochemical Reflectance Index (PRI) = (B-G)/(B+G)
% (8) <gri> Red-Green Ratio (GRI) = R/G;
% (9) <vrei1> Vogelmann Red Edge 1 (VReI) = NIR/Re
% (10) <savi> Soil Adjusted VI (SAVI) = (NIR-Red)*(1+L)/(NIR + Red + L)
%       where, L = soild brightness correction factor
%       L = 1 for high vegetation region and L = 0.5 (default)
%       based on our data, we use L = 1
% (11) <osavi> Optimized SAVI (OSAVI) = (NIR-Red)*(1+0.16)/(NIR + Red + 0.16)
% (12) <msavi> Modified SAVI (MSAVI) 
%       = 0.5(2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8(NIR - Red)))
% (13) <grvi> Green Red VI (GRVI) = (Green - Red) / (Green + Red)
% (14) <mgrvi> Modified GRVI (MGRVI) = (Green^2 - Red^2) / (Green^2 + Red^2)
% (15) <rgbvi> Red Green Blue VI (RGBVI) 
%                       = (Green^2 - Red*Blue) / (Green^2 + Red*Blue)

features_list = {'dvi','ndvi','sr','rendvi','arvi','ari2', ...
                    'pri','gri','vrei1','savi','osavi','msavi', ...
                    'grvi', 'mgrvi', 'rgbvi'};
basePath = '../../data/03_biomass';
trainPath = 'train/ortho';
testPath = 'test/ortho';

trainPath = fullfile(basePath, trainPath);
testPath = fullfile(basePath, testPath);

features = struct('train', [], 'test', []);

numFiles = length(dir(fullfile(trainPath, '*.mat')));
for i = 1:length(features_list)
    v = matlab.lang.makeValidName(features_list{i});
    eval(['features.train.', v, '= single(zeros(numFiles,1));']);
    for j = 1:numFiles
        fprintf('feature = %s, train plot = %d\n', features_list{i}, j);
        load(fullfile(trainPath, [num2str(j), '.mat']));
        im = single(image(:,:,:)); clear image;
        im_feat = extractVI(im, features_list{i});
        eval(['features.train.', v, '(j) = sum(im_feat(:));']);
    end
    % normalize features
    eval(['valMax = max(features.train.', v, ');']);
    eval(['features.train.', v, '= bsxfun(@rdivide, features.train.', ...
                v, ', valMax);']);
end

numFiles = length(dir(fullfile(testPath, '*.mat')));
for i = 1:length(features_list)
    v = matlab.lang.makeValidName(features_list{i});
    eval(['features.test.', v, '= single(zeros(numFiles,1));']);
    for j = 1:numFiles
        fprintf('feature = %s, test plot = %d\n', features_list{i}, j);
        load(fullfile(testPath, [num2str(j), '.mat']));
        im = single(image(:,:,:)); clear image;
        im_feat = extractVI(im, features_list{i});
        eval(['features.test.', v, '(j) = sum(im_feat(:));']);
    end
    % normalize features
    eval(['valMax = max(features.test.', v, ');']);
    eval(['features.test.', v, '= bsxfun(@rdivide, features.test.', ...
                v, ', valMax);']);    
end

save('features.mat', 'features');

end

function [im_out] = extractVI(im, viName)
% im = 5 channel image (Blue(1), Green(2), Red(3), NIR(4), RedEdge(5))
% featName = name of VI according to the list at the top
    if strcmp(viName, 'dvi')
        im_out = bsxfun(@minus, im(:,:,4), im(:,:,3));
    elseif strcmp(viName, 'ndvi')
        im_out = bsxfun(@rdivide, ...
            bsxfun(@minus, im(:,:,4), im(:,:,3)), ...
            bsxfun(@plus, im(:,:,4), im(:,:,3)));
    elseif strcmp(viName, 'sr')
        im_out = bsxfun(@rdivide, im(:,:,4), im(:,:,3));
    elseif strcmp(viName, 'rendvi')
        im_out = bsxfun(@rdivide, ...
            bsxfun(@minus, im(:,:,4), im(:,:,5)), ...
            bsxfun(@plus, im(:,:,4), im(:,:,5)));        
    elseif strcmp(viName, 'arvi')
        im_out = bsxfun(@minus, bsxfun(@times, im(:,:,3), 2), im(:,:,1));
        im_out = bsxfun(@rdivide, ...
            bsxfun(@minus, im(:,:,4), im_out), ...
            bsxfun(@plus, im(:,:,4), im_out));  
    elseif strcmp(viName, 'ari2')
        im_out = bsxfun(@minus, ...
            bsxfun(@rdivide, 1, im(:,:,2)), ...
            bsxfun(@rdivide, 1, im(:,:,5)));
        im_out = bsxfun(@times, im(:,:,4), im_out);
    elseif strcmp(viName, 'pri')
        im_out = bsxfun(@rdivide, ...
            bsxfun(@minus, im(:,:,1), im(:,:,2)), ...
            bsxfun(@plus, im(:,:,1), im(:,:,2)));        
    elseif strcmp(viName, 'gri')
        im_out = bsxfun(@rdivide, im(:,:,3), im(:,:,2));
    elseif strcmp(viName, 'vrei1')
        im_out = bsxfun(@rdivide, im(:,:,4), im(:,:,5));
    elseif strcmp(viName, 'savi')
        im_out = bsxfun(@plus, im(:,:,4), im(:,:,3));
        im_out = bsxfun(@plus, im_out, 1);
        im_out = bsxfun(@rdivide, ...
            bsxfun(@minus, im(:,:,4), im(:,:,3)), im_out);
        im_out = bsxfun(@times, im_out, 2);
    elseif strcmp(viName, 'osavi')
        im_out = bsxfun(@plus, im(:,:,4), im(:,:,3));
        im_out = bsxfun(@plus, im_out, 0.16);
        im_out = bsxfun(@rdivide, ...
            bsxfun(@minus, im(:,:,4), im(:,:,3)), im_out);
        im_out = bsxfun(@times, im_out, 1.16);        
    elseif strcmp(viName, 'msavi')
        im_out = bsxfun(@times, 8, ...
            bsxfun(@minus, im(:,:,4), im(:,:,3)));
        tmp = bsxfun(@plus, 1, bsxfun(@times, im(:,:,4), 2));
        im_out = sqrt(bsxfun(@minus, bsxfun(@power, tmp, 2), im_out));
        im_out = bsxfun(@minus, tmp, im_out);
        im_out = bsxfun(@times, im_out, 0.5);
    elseif strcmp(viName, 'grvi')
        im_out = bsxfun(@rdivide, ...
            bsxfun(@minus, im(:,:,2), im(:,:,1)), ...
            bsxfun(@plus, im(:,:,2), im(:,:,1)));
    elseif strcmp(viName, 'mgrvi')
        red_2 = bsxfun(@power, im(:,:,3), 2);
        green_2 = bsxfun(@power, im(:,:,2), 2);
        im_out = bsxfun(@rdivide, ...
            bsxfun(@minus, green_2, red_2), ...
            bsxfun(@plus, green_2, red_2));
    elseif strcmp(viName, 'rgbvi')
        green_2 = bsxfun(@power, im(:,:,2), 2);
        im_out = bsxfun(@times, im(:,:,1), im(:,:,3));
        im_out = bsxfun(@rdivide, ...
            bsxfun(@minus, green_2, im_out), ...
            bsxfun(@plus, green_2, im_out));
    else
        error('VI not listed.');
    end
end
