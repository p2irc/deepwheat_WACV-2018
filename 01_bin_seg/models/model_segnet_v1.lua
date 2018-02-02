--[[ Paper title: SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
Caffe source : https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Models/segnet_train.prototxt

The difference between original segnet and this one is that this implementation
uses the same set of training parameters in all the layers whereas the original
Caffe implementation uses layerwise training parameters.
]]--

require 'torch';
require 'cutorch';
require 'cudnn';
require 'nn';
require 'cunn';
local nninit = require 'nninit';

cudnn.benchmark = true;


-- needed for unpooling operation
local pool_layer_1 = nn.SpatialMaxPooling(2, 2, 2, 2);
local pool_layer_2 = nn.SpatialMaxPooling(2, 2, 2, 2);
local pool_layer_3 = nn.SpatialMaxPooling(2, 2, 2, 2);
local pool_layer_4 = nn.SpatialMaxPooling(2, 2, 2, 2);
local pool_layer_5 = nn.SpatialMaxPooling(2, 2, 2, 2);

local weights = torch.Tensor(2); -- weights for loss calculation
weights[1] = 1.0/1.0; -- background
weights[2] = 2;--1/0.8; -- foreground

model = nn.Sequential()
	-- P = ((W-1)S - W + F)/2
	-- conv layer 1 on 32x32
	:add(cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1)) -- 64 x 224 x 224
	:add(cudnn.SpatialBatchNormalization(64))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))  -- 64 x 224 x 224
	:add(cudnn.SpatialBatchNormalization(64))
	:add(cudnn.ReLU())
	:add(pool_layer_1) -- becomes 64 x 112 x 112

	-- conv layer 2
	:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)) -- 128 x 112 x 112
	:add(cudnn.SpatialBatchNormalization(128))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(128))
	:add(cudnn.ReLU())
	:add(pool_layer_2) -- becomes 64 x 56 x 56

	-- conv layer 3
	:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)) -- 256 x 56 x 56
	:add(cudnn.SpatialBatchNormalization(256))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(256))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(256))
	:add(cudnn.ReLU())
	:add(pool_layer_3) -- becomes 256 x 28 x 28

	-- conv layer 4
	:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)) -- 512 x 28 x 28
	:add(cudnn.SpatialBatchNormalization(512))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(512))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(512))
	:add(cudnn.ReLU())
	:add(pool_layer_4) -- becomes 512 x 14 x 14

	-- conv layer 5
	:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)) -- 512 x 14 x 14
	:add(cudnn.SpatialBatchNormalization(512))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(512))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(512))
	:add(cudnn.ReLU())
	:add(pool_layer_5) -- becomes 512 x 7 x 7

	-- deconv layer 5
	:add(nn.SpatialMaxUnpooling(pool_layer_5)) -- becomes 512 x 14 x 14
	:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(512))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(512))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(512))
	:add(cudnn.ReLU())

	-- deconv layer 4
	:add(nn.SpatialMaxUnpooling(pool_layer_4)) -- becomes 512 x 28 x 28
	:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(512))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(512))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(512, 256, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(256))
	:add(cudnn.ReLU())

	-- deconv layer 3
	:add(nn.SpatialMaxUnpooling(pool_layer_3)) -- becomes 256 x 56 x 56
	:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(256))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(256))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(128))
	:add(cudnn.ReLU())

	-- doconv layer 2
	:add(nn.SpatialMaxUnpooling(pool_layer_2)) -- becomes 128 x 112 x 112
	:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(128))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(64))
	:add(cudnn.ReLU())

	-- deconv layer 1
	:add(nn.SpatialMaxUnpooling(pool_layer_1)) -- becomes 64 x 224 x 224
	:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
	:add(cudnn.SpatialBatchNormalization(64))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(64, 2, 3, 3, 1, 1, 1, 1)) -- becomes 2 x 224 x 224

print("Starting Conv layer initialization ... ");
for i = 1, model:size(), 1 do
	if (torch.type(model:get(i)) == 'cudnn.SpatialConvolution') or
			(torch.type(model:get(i)) == 'nn.SpatialConvolution') then
		-- initialize with 'Xavier'
		print(string.format("Intializing layer = %d", i));
		model:get(i):init('weight', nninit.xavier, {dist='normal', gain=1.0});
	end
end

model:cuda();

criterion = cudnn.SpatialCrossEntropyCriterion(weights)
criterion:cuda()
--cudnn.convert(criterion, cudnn)

print(model);
print(criterion);

