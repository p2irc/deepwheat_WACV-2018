require 'torch';
require 'cutorch';
require 'cudnn';
require 'nn';
require 'cunn';
local nninit = require 'nninit';

cudnn.benchmark = true;

--local typeInput = 'osavi';
--local typeInput = 'all';
--local typeInput = 'rgb';
local typeInput = 'dem';

local numInputChn;
if (typeInput == 'dem') then
	numInputChn = 1;
elseif (typeInput == 'rgb') then
	numInputChn = 4;
elseif (typeInput == 'all') then
	numInputChn = 6;
elseif (typeInput == 'osavi') then
	numInputChn = 2;
end

local numFeatures = 512;

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
	:add(basicBlock(numInputChn, 64, 3, 1, 1, 5) )  -- 64 x 192 x 192

	:add(resBlock(64, 64, 3, 1, 1, 5) )  -- 64 x 192 x 192

	:add(basicBlock(64, 128, 3, 1, 1, 10) )  -- 128 x 192 x 192
	:add(basicBlock(128, 128, 3, 3, 1, 10) )  -- 128 x 64 x 64 

	:add(resBlock(128, 128, 3, 1, 1, 10) )  -- 128 x 64 x 64

	:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 128 x 32 x 32

	:add(basicBlock(128, 256, 3, 1, 1, 10) )  -- 256 x 32 x 32

	:add(resBlock(256, 256, 3, 1, 1, 10) )  -- 256 x 32 x 32

	:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 256 x 16 x 16

	:add(basicBlock(256, 512, 3, 1, 1, 15) )  -- 512 x 16 x 16

	:add(resBlock(512, 512, 3, 1, 1, 15) )  -- 512 x 16 x 16

	:add(resBlock(512, 512, 3, 1, 1, 15) )  -- 512 x 16 x 16

	:add(basicBlock(512, 512, 3, 3, 1, 15) )  -- 512 x 6 x 6

	:add(resBlock(512, 512, 3, 1, 1, 15) )  -- 512 x 6 x 6

	:add(basicBlock(512, 512, 3, 1, 0, 15) )  -- 512 x 4 x 4

	:add(resBlock(512, 512, 3, 1, 1, 15) )  -- 512 x 4 x 4

	:add(basicBlock(512, 512, 3, 1, 0, 15) )  -- 512 x 2 x 2

local regressor = nn.Sequential()
	:add(nn.View(512 * 2 * 2))
	:add(nn.Linear(512 * 2 * 2, numFeatures))
--	:add(nn.Normalize(1))
--	:add(cudnn.ReLU(true))
	:add(nn.Linear(numFeatures, numFeatures))
--	:add(nn.Normalize(1))
--	:add(cudnn.ReLU(true))
	:add(nn.Linear(numFeatures, 1));

model = nn.Sequential():add(features):add(regressor);

--[[
print("Starting Conv layer initialization ... ");
for i = 1, model:size(), 1 do
	if (torch.type(model:get(i)) == 'cudnn.SpatialConvolution') or
			(torch.type(model:get(i)) == 'nn.SpatialConvolution') then
		-- initialize with 'Xavier'
		print(string.format("Intializing layer = %d", i));
		model:get(i):init('weight', nninit.xavier, {dist='normal', gain=1.0});
	end
end
]]--

model:cuda();

criterion = nn.AbsCriterion();
criterion.sizeAverage = false;
criterion:cuda();

print(model);
print(criterion);

