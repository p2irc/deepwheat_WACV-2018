require 'image';
require 'paths';
local matio = require 'matio';

torch.setdefaulttensortype('torch.FloatTensor');

local pathData = '../../data/02_count_emergence/train/rgb_patch_resize_224';
local pathDatFiles = '../../data/02_count_emergence/train/898_emergence_dat';
local pathGt = '../../data/02_count_emergence/train';
local gtFileBasic = 'count_gt.mat';
local gtFileAug = 'count_gt_aug.mat';

gtFileBasic = paths.concat(pathGt, gtFileBasic);
gtFileAug = paths.concat(pathGt, gtFileAug);

os.execute("rm -rfv " .. pathDatFiles);
os.execute("mkdir -v " .. pathDatFiles);

local numFiles = #paths.dir(pathData) - 2;
print(string.format('Total number of files = %d', numFiles));

local count_gt = torch.squeeze(matio.load(gtFileBasic)['counts']);
local count_gt_aug = torch.ByteTensor(numFiles * 4):fill(0); -- gt file for augmented data

torch.manualSeed(72);
local outFileList = torch.randperm(numFiles * 4); -- original, hflip, vflip, hvflip(180)

for i = 1, numFiles, 1 do
	local tmp = i*4;
	print(string.format('%d, \t%d, %d, %d, %d', i, outFileList[tmp-3], outFileList[tmp-2], outFileList[tmp-1], outFileList[tmp]));
	local im = image.load(paths.concat(pathData, tostring(i) .. '.png'), 3, 'float'); -- no mean subtraction
	torch.save(paths.concat(pathDatFiles, tostring(outFileList[tmp-3]) .. '.dat'), im);
	torch.save(paths.concat(pathDatFiles, tostring(outFileList[tmp-2]) .. '.dat'), image.hflip(im) );
	torch.save(paths.concat(pathDatFiles, tostring(outFileList[tmp-1]) .. '.dat'), image.vflip(im) );
	torch.save(paths.concat(pathDatFiles, tostring(outFileList[tmp]) .. '.dat'), image.vflip(image.hflip(im)) );

	count_gt_aug[outFileList[tmp-3]] = count_gt[i];
	count_gt_aug[outFileList[tmp-2]] = count_gt[i];
	count_gt_aug[outFileList[tmp-1]] = count_gt[i];
	count_gt_aug[outFileList[tmp]] = count_gt[i];
end

matio.save(gtFileAug, {counts=count_gt_aug});
