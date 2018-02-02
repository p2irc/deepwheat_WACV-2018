clear all; close all; clc;

train_or_test = 'test';

th_prec = 0.7; % threshold on precision
th_rec = 0.9; % threshold on recall
th_acc = 0.8; % threshold on accuracy

if strcmp(train_or_test, 'train')
	basePath = '../../data/01_bin_seg/train';
else
	basePath = '../../data/01_bin_seg/test';
end

inGtPath = 'gt';
inBinPath = 'seg';
% ---------------------------------------

inGtPath = fullfile(basePath, inGtPath);
inBinPath = fullfile(basePath, inBinPath);

imgList = dir(fullfile(inGtPath, '*.png'));

prec_avg = 0;
rec_avg = 0;
acc_avg = 0;
for i = 1:length(imgList)
    gt = imread(fullfile(inGtPath, imgList(i).name))>0;
    bs = imread(fullfile(inBinPath, imgList(i).name))>0;
    true_pos = numel(find(bs==1 & gt==1));
    true_neg = numel(find(bs==0 & gt==0));
    false_pos = numel(find(bs==1 & gt==0));
    false_neg = numel(find(bs==0 & gt==1));
    prec = true_pos/(true_pos + false_pos);
    rec = true_pos/(true_pos + false_neg);
    acc = (true_pos + true_neg)/numel(gt);
    fprintf('file = %d, prec = %f, rec = %f, acc = %f\n', i, prec, rec, acc);
    prec_avg = prec_avg + prec;
    rec_avg = rec_avg + rec;
    acc_avg = acc_avg + acc;
end

prec_avg = prec_avg/length(imgList);
rec_avg = rec_avg/length(imgList);
acc_avg = acc_avg/length(imgList);
fprintf('Precision (Average) = %f\n', prec_avg);
fprintf('Recall (Average) = %f\n', rec_avg);
fprintf('Accuracy (Average) = %f\n', acc_avg);
