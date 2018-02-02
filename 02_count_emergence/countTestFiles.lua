require 'torch';
require 'paths';
require 'image';
local matio = require 'matio';
matio.use_lua_strings = true; -- to read file names as strings than char tensors

torch.setdefaulttensortype('torch.FloatTensor');

---------- model loading start ---------
require 'torch';
require 'cutorch';
require 'cudnn';
require 'nn';
require 'cunn';

cudnn.benchmark = true; -- true does not work with variable size

model = torch.load('./models_final/model_count_01_epoch_50.t7'); 
--model = torch.load('./models_inception_final/model_count_01_epoch_50.t7'); 
--model = torch.load('./models_plain_final/model_count_01_epoch_100.t7'); 
model:evaluate();
--model:training();
model:cuda();
---------- model loading finished ---------

--local mean_seg_finetune = torch.load('mean_seg_finetune.dat'); -- image mean
local basePath = '../../data/02_count_emergence/test';
local dataPath = 'rgb_patch_resize_224';
local outFileName = 'count_test_output.mat';

dataPath = paths.concat(basePath, dataPath);

function runTest()
	testFiles();
end

--local inputsCpu = torch.Tensor(1, 3, 224, 224);
unsqueeze = nn.Unsqueeze(1);

function testFiles()
	local numFiles = #paths.dir(dataPath) - 2;
	print(string.format("Number of files = %d", numFiles));
	local countList = torch.Tensor(numFiles):fill(0.0);
	for i = 1, numFiles, 1 do
--		print(tostring(fileList[i]));
		local inputsCpu = image.load(paths.concat(dataPath, tostring(i) .. '.png'), 3, 'float'); -- - mean_seg_finetune;
		inputsCpu = unsqueeze:forward(inputsCpu);
		local outputs = model:forward(inputsCpu:cuda());
--		print(torch.type(outputs));
--		print(outputs:size());
		countList[i] = outputs:float()[1];
		print(tostring(i), tostring(outputs:float()[1]));
	end

	matio.save(paths.concat(basePath, outFileName), {counts=countList});

end

