require 'torch';
require 'cutorch';
require 'cudnn';
require 'nn';
require 'cunn';
local nninit = require 'nninit';

cudnn.benchmark = true;

local numFeatures = 64;
local sizeStart = 128;

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

local features = nn.Sequential()
	:add(cudnn.SpatialConvolution(3, sizeStart, 7, 7, 1, 1, 3, 3)) -- 128 x 224 x 224
	:add(cudnn.SpatialCrossMapLRN(5))
	:add(cudnn.ReLU(true))

	:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 128 x 112 x 112

	-- replacement of resBlock
	:add(cudnn.SpatialConvolution(sizeStart, sizeStart, 3, 3, 1, 1, 1, 1)) -- 128 x 112 x 112
	:add(cudnn.SpatialCrossMapLRN(5))
	:add(cudnn.ReLU(true))
	:add(cudnn.SpatialConvolution(sizeStart, sizeStart, 3, 3, 1, 1, 1, 1)) -- 128 x 112 x 112
	:add(cudnn.SpatialCrossMapLRN(5))
	:add(cudnn.ReLU(true))

	-- replacement of resBlock
	:add(cudnn.SpatialConvolution(sizeStart, sizeStart, 3, 3, 1, 1, 1, 1)) -- 128 x 112 x 112
	:add(cudnn.SpatialCrossMapLRN(5))
	:add(cudnn.ReLU(true))
	:add(cudnn.SpatialConvolution(sizeStart, sizeStart, 3, 3, 1, 1, 1, 1)) -- 128 x 112 x 112
	:add(cudnn.SpatialCrossMapLRN(5))
	:add(cudnn.ReLU(true))
	-- replacement of resBlock
	:add(cudnn.SpatialConvolution(sizeStart, sizeStart, 3, 3, 1, 1, 1, 1)) -- 128 x 112 x 112
	:add(cudnn.SpatialCrossMapLRN(5))
	:add(cudnn.ReLU(true))
	:add(cudnn.SpatialConvolution(sizeStart, sizeStart, 3, 3, 1, 1, 1, 1)) -- 128 x 112 x 112
	:add(cudnn.SpatialCrossMapLRN(5))
	:add(cudnn.ReLU(true))

	:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 128 x 56 x 56

	-- replacement of basinInception
	:add(cudnn.SpatialConvolution(sizeStart, 2*sizeStart, 3, 3, 1, 1, 1, 1)) -- 256 x 56 x 56
	:add(cudnn.SpatialCrossMapLRN(10))
	:add(cudnn.ReLU(true))

	-- replacement of resInception
	:add(cudnn.SpatialConvolution(2*sizeStart, 2*sizeStart, 3, 3, 1, 1, 1, 1)) -- 256 x 56 x 56
	:add(cudnn.ReLU(true))
	:add(cudnn.SpatialConvolution(2*sizeStart, 2*sizeStart, 3, 3, 1, 1, 1, 1)) -- 256 x 56 x 56
	:add(cudnn.SpatialCrossMapLRN(10))
	:add(cudnn.ReLU(true))
	-- replacement of resInception
	:add(cudnn.SpatialConvolution(2*sizeStart, 2*sizeStart, 3, 3, 1, 1, 1, 1)) -- 256 x 56 x 56
	:add(cudnn.ReLU(true))
	:add(cudnn.SpatialConvolution(2*sizeStart, 2*sizeStart, 3, 3, 1, 1, 1, 1)) -- 256 x 56 x 56
	:add(cudnn.SpatialCrossMapLRN(10))
	:add(cudnn.ReLU(true))
	-- replacement of resInception
	:add(cudnn.SpatialConvolution(2*sizeStart, 2*sizeStart, 3, 3, 1, 1, 1, 1)) -- 256 x 56 x 56
	:add(cudnn.ReLU(true))
	:add(cudnn.SpatialConvolution(2*sizeStart, 2*sizeStart, 3, 3, 1, 1, 1, 1)) -- 256 x 56 x 56
	:add(cudnn.SpatialCrossMapLRN(10))
	:add(cudnn.ReLU(true))

	:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 256 x 28 x 28

	-- replacement of basinInception
	:add(cudnn.SpatialConvolution(2*sizeStart, 4*sizeStart, 3, 3, 1, 1, 1, 1)) -- 512 x 28 x 28
	:add(cudnn.SpatialCrossMapLRN(10))
	:add(cudnn.ReLU(true))

	-- replacement of resInception
	:add(cudnn.SpatialConvolution(4*sizeStart, 4*sizeStart, 3, 3, 1, 1, 1, 1)) -- 512 x 28 x 28
	:add(cudnn.ReLU(true))
	:add(cudnn.SpatialConvolution(4*sizeStart, 4*sizeStart, 3, 3, 1, 1, 1, 1)) -- 512 x 28 x 28
	:add(cudnn.SpatialCrossMapLRN(10))
	:add(cudnn.ReLU(true))
	-- replacement of resInception
	:add(cudnn.SpatialConvolution(4*sizeStart, 4*sizeStart, 3, 3, 1, 1, 1, 1)) -- 512 x 28 x 28
	:add(cudnn.ReLU(true))
	:add(cudnn.SpatialConvolution(4*sizeStart, 4*sizeStart, 3, 3, 1, 1, 1, 1)) -- 512 x 28 x 28
	:add(cudnn.SpatialCrossMapLRN(10))
	:add(cudnn.ReLU(true))
	-- replacement of resInception
	:add(cudnn.SpatialConvolution(4*sizeStart, 4*sizeStart, 3, 3, 1, 1, 1, 1)) -- 512 x 28 x 28
	:add(cudnn.ReLU(true))
	:add(cudnn.SpatialConvolution(4*sizeStart, 4*sizeStart, 3, 3, 1, 1, 1, 1)) -- 512 x 28 x 28
	:add(cudnn.SpatialCrossMapLRN(10))
	:add(cudnn.ReLU(true))

	:add(cudnn.SpatialAveragePooling(28, 28, 28, 28)) -- 512 x 1 x 1

features = initModules(features);

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
