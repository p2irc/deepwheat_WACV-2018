-- run with CUDA_VISIBLE_DEVICES=0 th <yourscript.lua>
require 'paths';
require 'optim';
require 'torch';
require 'cutorch';
require 'cudnn';
require 'nn';
require 'cunn';
local matio = require 'matio';

torch.setdefaulttensortype('torch.FloatTensor');

cudnn.benchmark = true; -- true does not work with variable size

epochTest = 50;

rowMat, colMat = 96, 384;

sizeImg = colMat/2;

model = torch.load('./models/model_count_resnet_01_dem_epoch_' .. tostring(epochTest) .. '.t7'); 
model:evaluate();
model:cuda();
---------- model loading finished ---------

local basePath = '../../data/03_biomass/test';
local orthoPath = 'ortho';
local demPath = 'dem';
local resPath = 'result_dem';
local outFileName = 'countTest_epoch_' .. tostring(epochTest) .. '.mat';

orthoPath = paths.concat(basePath, orthoPath);
demPath = paths.concat(basePath, demPath);
resPath = paths.concat(basePath, resPath);

os.execute("mkdir -v " .. resPath);

function runTest()
	testFiles();
end

--local inputsCpu = torch.Tensor(4, sizeImg, sizeImg);
local inputsCpu = torch.Tensor(1, 1, sizeImg, sizeImg);
local unsqueeze = nn.Unsqueeze(1);

function testFiles()
	local numFiles = #paths.dir(demPath) - 2;
	print(string.format("Number of files = %d", numFiles));
	local countList = torch.Tensor(numFiles):fill(0.0);
	for i = 1, numFiles, 1 do
--		print(tostring(fileList[i]));
--		local ortho = matio.load(paths.concat(orthoPath, tostring(i) .. '.mat'), 'im_sample'); -- no mean subtraction
		local dem = matio.load(paths.concat(demPath, tostring(i) .. '.mat'), 'dem_sample');
--		ortho = ortho:permute(3, 1, 2);
--		dem = unsqueeze:forward(dem);
--		inputsCpu = torch.cat(ortho, dem, 1);
		inputsCpu[1] = unsqueeze:forward(dem);
--		inputsCpu[{ {}, {1, sizeImg/2}, {} }] = ortho[{ {}, {}, {1, sizeImg} }];
--		inputsCpu[{ {}, {sizeImg/2 + 1, sizeImg}, {} }] = ortho[{ {}, {}, {sizeImg+1, 2*sizeImg} }];
 
		local outputs = model:forward(inputsCpu:cuda());
--		print(torch.type(outputs));
--		print(outputs:size());
		countList[i] = outputs:float()[1];
		print(tostring(i), tostring(outputs:float()[1]));
	end

	matio.save(paths.concat(resPath, outFileName), {counts=countList});

end

