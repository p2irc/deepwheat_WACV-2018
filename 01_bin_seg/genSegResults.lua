require 'torch';
require 'cutorch';
require 'cudnn';
require 'nn';
require 'cunn';

require 'paths';
require 'image';

cudnn.benchmark = true;

torch.setdefaulttensortype('torch.FloatTensor'); -- set default

local train_or_test = 'test'; -- 'train' or 'test' for generating binary segmentations

local imgSize, stepSize = 224, 112;
local basePath;
if train_or_test == 'train' then
	basePath = '../../data/01_bin_seg/train';
else
	basePath = '../../data/01_bin_seg/test';
end
local rgbPath = 'rgb';
local segPath = 'seg';

------------ load model -------------
local model = torch.load('./models/model_segnet_v1_epoch_30.t7');
model:evaluate();
model:cuda();
local mean_image = torch.load('mean_seg.dat');
-------------------------------------

rgbPath = paths.concat(basePath, rgbPath);
segPath = paths.concat(basePath, segPath);
os.execute("rm -rfv " .. segPath);
os.execute("mkdir -v " .. segPath);

local imgList = paths.dir(rgbPath);

--[[
local im; -- for full image
local im_sub; -- for subimage
local seg; -- for full seg image
local seg_sub; -- for seg subimage;
local r_cur, c_cur = 1, 1; -- current row and col positions, 1-based indexing
local r_lim, c_lim; -- limits for running subimage loop
]]--

local im_sub = torch.Tensor(1, 3, imgSize, imgSize);

local countFullImg = 0;
for i = 1, #imgList, 1 do
	if (imgList[i] ~= '.') and (imgList[i] ~= '..') then
		countFullImg = countFullImg + 1;
		local im = image.load(paths.concat(rgbPath, imgList[i]), 3, 'float') - mean_image;
		local seg = torch.zeros(2, im:size(2), im:size(3));
		local r_lim = im:size(2) - imgSize + 1;
		local c_lim = im:size(3) - imgSize + 1;
		local r_cur, c_cur = 1, 1;
		local count = 0;
		while(r_cur <= r_lim) do
			c_cur = 1;
			while(c_cur <= c_lim) do
				count = count + 1;
				print(string.format('image = %d, subimage = %d', countFullImg, count));
				im_sub[1] = im[{ {}, {r_cur, r_cur+imgSize-1}, {c_cur, c_cur+imgSize-1} }];
--				print(im_sub:size());
				local seg_prob = model:forward(im_sub:cuda());
				seg_prob = torch.squeeze(seg_prob):float();
--				print(seg_prob:size());
				-- add probabilities to specific region
				seg[{ {}, {r_cur, r_cur+imgSize-1}, {c_cur, c_cur+imgSize-1} }] =
						seg[{ {}, {r_cur, r_cur+imgSize-1}, {c_cur, c_cur+imgSize-1} }] + seg_prob;
				c_cur = c_cur + stepSize;
			end
			c_cur = c_cur - stepSize;
			if c_cur < c_lim then
				count = count + 1;
				print(string.format('image = %d, subimage = %d', countFullImg, count));
				c_cur = c_lim;
				im_sub[1] = im[{ {}, {r_cur, r_cur+imgSize-1}, {c_cur, c_cur+imgSize-1} }];
				local seg_prob = model:forward(im_sub:cuda());
				seg_prob = torch.squeeze(seg_prob):float();
				-- add probabilities to specific region
				seg[{ {}, {r_cur, r_cur+imgSize-1}, {c_cur, c_cur+imgSize-1} }] =
						seg[{ {}, {r_cur, r_cur+imgSize-1}, {c_cur, c_cur+imgSize-1} }] + seg_prob;
			end
			r_cur = r_cur + stepSize;
		end
		r_cur = r_cur - stepSize;
		if r_cur < r_lim then
			r_cur, c_cur = r_lim, 1;
			while(c_cur <= c_lim) do
				count = count + 1;
				print(string.format('image = %d, subimage = %d', countFullImg, count));
				im_sub[1] = im[{ {}, {r_cur, r_cur+imgSize-1}, {c_cur, c_cur+imgSize-1} }];
--				print(im_sub:size());
				local seg_prob = model:forward(im_sub:cuda());
				seg_prob = torch.squeeze(seg_prob):float();
--				print(seg_prob:size());
				-- add probabilities to specific region
				seg[{ {}, {r_cur, r_cur+imgSize-1}, {c_cur, c_cur+imgSize-1} }] =
						seg[{ {}, {r_cur, r_cur+imgSize-1}, {c_cur, c_cur+imgSize-1} }] + seg_prob;
				c_cur = c_cur + stepSize;
			end
			c_cur = c_cur - stepSize;
			if c_cur < c_lim then
				count = count + 1;
				print(string.format('image = %d, subimage = %d', countFullImg, count));
				c_cur = c_lim;
				im_sub[1] = im[{ {}, {r_cur, r_cur+imgSize-1}, {c_cur, c_cur+imgSize-1} }];
				local seg_prob = model:forward(im_sub:cuda());
				seg_prob = torch.squeeze(seg_prob):float();
				-- add probabilities to specific region
				seg[{ {}, {r_cur, r_cur+imgSize-1}, {c_cur, c_cur+imgSize-1} }] =
						seg[{ {}, {r_cur, r_cur+imgSize-1}, {c_cur, c_cur+imgSize-1} }] + seg_prob;
			end
		end
		_, seg = torch.max(seg, 1);
		seg = torch.squeeze(seg);
		seg = seg - 1;
		seg = seg:byte();
		seg = seg*255;
		-- save segmentation file
		image.save(paths.concat(segPath, imgList[i]), seg);
	end
end
