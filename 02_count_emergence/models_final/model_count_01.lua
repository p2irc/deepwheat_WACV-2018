require 'torch';
require 'cutorch';
require 'cudnn';
require 'nn';
require 'cunn';
local nninit = require 'nninit';

cudnn.benchmark = true;

local numFeatures = 64;
local sizeStart = 128;

local DimConcat = 2;

local function initModules(block)
	print("Starting Conv layer initialization ... ");
	for i = 1, block:size(), 1 do
		if (torch.type(block:get(i)) == 'cudnn.SpatialConvolution') or
				(torch.type(block:get(i)) == 'nn.SpatialConvolution') then
			-- initialize with 'Xavier'
			print(string.format("Intializing layer = %d, \t%s", i, block:get(i)));
			block:get(i):init('weight', nninit.xavier, {dist='normal', gain=1.0});
		end
	end
	return block;
end

local function initInception(block)
	print("Starting Conv layer initialization ... ");
	for i = 1, block:size(), 1 do
		for j = 1, block:get(i):size(), 1 do
			if (torch.type(block:get(i):get(j)) == 'cudnn.SpatialConvolution') or
				(torch.type(block:get(i):get(j)) == 'nn.SpatialConvolution') then
			-- initialize with 'Xavier'
				print(string.format("Intializing Inception layer = (%d %d), \t%s", i, j, block:get(i):get(j)));
				block:get(i):get(j):init('weight', nninit.xavier, {dist='normal', gain=1.0});
			end
		end
	end
	return block;
end

local Inception = function(nInput, nOutput, n1x1, n3x3r, n3x3, n5x5r, n5x5, nPoolProj)
-- source: https://github.com/eladhoffer/ImageNet-Training/blob/master/Models/GoogLeNet_Model.lua
    local InceptionModule = nn.DepthConcat(DimConcat)
    InceptionModule:add(
		nn.Sequential()
			:add(cudnn.SpatialConvolution(nInput, n1x1, 1, 1, 1, 1, 0, 0)) )

    InceptionModule:add(
		nn.Sequential()
			:add(cudnn.SpatialConvolution(nInput, n3x3r, 1, 1, 1, 1, 0, 0))
			:add(cudnn.ReLU(true))
			:add(cudnn.SpatialConvolution(n3x3r, n3x3, 3, 3, 1, 1, 1, 1)) )

    InceptionModule:add(
		nn.Sequential()
			:add(cudnn.SpatialConvolution(nInput, n5x5r, 1, 1, 1, 1, 0, 0))
			:add(nn.ReLU(true))
			:add(cudnn.SpatialConvolution(n5x5r, n5x5, 5, 5, 1, 1, 2, 2)) )

    InceptionModule:add(
		nn.Sequential()
			:add(cudnn.SpatialMaxPooling(3,3, 1,1, 1,1))
			:add(cudnn.SpatialConvolution(nInput, nPoolProj, 1,1, 1,1, 0,0)) )

	InceptionModule = initInception(InceptionModule);

    return InceptionModule
end

local function basicInceptionBlock(numMapsIn, numMapsOut, numMapsLrn)
    local block = nn.Sequential()
    :add(Inception(numMapsIn, numMapsOut, numMapsOut/8, numMapsOut/2, numMapsOut/2, numMapsOut/4, numMapsOut/4, numMapsOut/8 ))
    :add(cudnn.SpatialCrossMapLRN(numMapsLrn))
    :add(cudnn.ReLU(true));

    return block;
end

local function resInceptionBlock(numMapsIn, numMapsOut, numMapsLrn)
    -- taken and modified from https://github.com/facebook/fb.resnet.torch

    local inBlock = nn.Sequential()
        :add(Inception(numMapsIn, numMapsOut, numMapsOut/8, numMapsOut/2, numMapsOut/2, numMapsOut/4, numMapsOut/4, numMapsOut/8 ))
        :add(cudnn.ReLU(true))
        :add(Inception(numMapsIn, numMapsOut, numMapsOut/8, numMapsOut/2, numMapsOut/2, numMapsOut/4, numMapsOut/4, numMapsOut/8 ))
        :add(cudnn.SpatialCrossMapLRN(numMapsLrn));

    return nn.Sequential()
	        :add(nn.ConcatTable():add(inBlock):add(nn.Identity()) ) -- end of ConcatTable
   		    :add(nn.CAddTable(true))
   		    :add(cudnn.ReLU(true));
end

local function basicBlock(numMapsIn, numMapsOut, sizeFilter, stride, pad, numMapsLrn)
    local block = nn.Sequential()
    :add(cudnn.SpatialConvolution(numMapsIn, numMapsOut, sizeFilter, sizeFilter, stride, stride, pad, pad))
    :add(cudnn.SpatialCrossMapLRN(numMapsLrn))
    :add(cudnn.ReLU(true));

	block = initModules(block);

    return block;
end

local function resBlock(numMapsIn, numMapsOut, sizeFilter, stride, pad, numMapsLrn)
    -- taken and modified from https://github.com/facebook/fb.resnet.torch

    local inBlock = nn.Sequential()
        :add(cudnn.SpatialConvolution(numMapsIn, numMapsOut, sizeFilter, sizeFilter, stride, stride, pad, pad))
        :add(cudnn.SpatialCrossMapLRN(numMapsLrn))
        :add(cudnn.ReLU(true))
        :add(cudnn.SpatialConvolution(numMapsIn, numMapsOut, sizeFilter, sizeFilter, stride, stride, pad, pad))
        :add(cudnn.SpatialCrossMapLRN(numMapsLrn));

	inBlock = initModules(inBlock);

    return nn.Sequential()
	        :add(nn.ConcatTable():add(inBlock):add(nn.Identity()) ) -- end of ConcatTable
   		    :add(nn.CAddTable(true))
   		    :add(cudnn.ReLU(true));
end


local features = nn.Sequential()
	:add(basicBlock(3, sizeStart, 7, 1, 3, 5) )  -- 128 x 224 x 224

	:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 128 x 112 x 112

	:add(resBlock(sizeStart, sizeStart, 3, 1, 1, 5) ) -- 128 x 112 x 112
	:add(resBlock(sizeStart, sizeStart, 3, 1, 1, 5) ) -- 128 x 112 x 112
	:add(resBlock(sizeStart, sizeStart, 3, 1, 1, 5) ) -- 128 x 112 x 112

	:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 128 x 56 x 56

	-- inception starts
	:add(basicInceptionBlock(sizeStart, 2*sizeStart, 10)) -- 256 x 56 x 56

	:add(resInceptionBlock(2*sizeStart, 2*sizeStart, 10)) -- 256 x 56 x 56
	:add(resInceptionBlock(2*sizeStart, 2*sizeStart, 10)) -- 256 x 56 x 56
	:add(resInceptionBlock(2*sizeStart, 2*sizeStart, 10)) -- 256 x 56 x 56

	:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 256 x 28 x 28

	:add(basicInceptionBlock(2*sizeStart, 4*sizeStart, 10) )  -- 512 x 28 x 28

	:add(resInceptionBlock(4*sizeStart, 4*sizeStart, 10) )  -- 512 x 28 x 28
	:add(resInceptionBlock(4*sizeStart, 4*sizeStart, 10) )  -- 512 x 28 x 28
	:add(resInceptionBlock(4*sizeStart, 4*sizeStart, 10) )  -- 512 x 28 x 28

	:add(cudnn.SpatialAveragePooling(28, 28, 28, 28)) -- 512 x 1 x 1

local regressor = nn.Sequential()
	:add(nn.View(4*sizeStart * 1 * 1))
--	:add(cudnn.ReLU(true))
	:add(nn.Linear(4*sizeStart, 1));

model = nn.Sequential():add(features):add(regressor);
model:cuda();

--criterion = nn.SmoothL1Criterion();
criterion = nn.AbsCriterion();
criterion:cuda();

print(model);
print(criterion);
