-- load pretrained models
require 'torch'
require 'cutorch'
require 'cudnn'
require 'nn'
require 'cunn'

cudnn.benchmark = true;

local weights = torch.Tensor(2); -- weights for loss calculation
weights[1] = 1.0; -- background
weights[2] = 1.5; -- foreground, 2.0(1-10), 

model = torch.load('./models/model_segnet_v1_epoch_20.t7');
model:cuda();

criterion = cudnn.SpatialCrossEntropyCriterion(weights)
criterion:cuda();

