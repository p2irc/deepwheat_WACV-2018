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
--local typeInput = 'osavi';
--local typeInput = 'all';
local typeInput = 'rgb';

local modelFileName = "model_count_01_" .. typeInput;

rowMat, colMat = 96, 384;

sizeImg = colMat/2;

model = torch.load('./models/' .. modelFileName .. '_epoch_' .. tostring(epochTest) .. '.t7'); 
model:evaluate();
model:cuda();
---------- model loading finished ---------

local basePath = '../../data/03_biomass/test';
local orthoPath = 'ortho';
local demPath = 'dem';
local resPath; 
local pathOsaviMax = '../../data/03_biomass/train/max_osavi.mat';
local outFileName = 'countTest_epoch_' .. tostring(epochTest) .. '.mat';

if (typeInput == 'rgb') then
	resPath = 'result_rgb';
elseif (typeInput == 'all') then
	resPath = 'result_full';
elseif (typeInput == 'osavi') then
	resPath = 'result_osavi';
end

local mult_osavi;
if (typeInput == 'osavi') then
	local max_osavi = matio.load(pathOsaviMax, 'max_osavi');
	mult_osavi = 1/max_osavi[1][1];
end

orthoPath = paths.concat(basePath, orthoPath);
demPath = paths.concat(basePath, demPath);
resPath = paths.concat(basePath, resPath);

os.execute("mkdir -v " .. resPath);

function runTest()
	testFiles();
end

if typeInput == 'rgb' then
	numInputChn = 4;
elseif typeInput == 'all' then
	numInputChn = 6;
elseif typeInput == 'osavi' then
	numInputChn = 2;
else
	error("Invalid number of input channels.")
end
local inputsCpu = torch.Tensor(numInputChn, sizeImg, sizeImg);
local unsqueeze = nn.Unsqueeze(1);

function testFiles()
	local numFiles = #paths.dir(demPath) - 2;
	print(string.format("Number of files = %d", numFiles));
	local countList = torch.Tensor(numFiles):fill(0.0);
	for i = 1, numFiles, 1 do
--		print(tostring(fileList[i]));
		local ortho = matio.load(paths.concat(orthoPath, tostring(i) .. '.mat'), 'im_sample'); -- no mean subtraction
		local dem = matio.load(paths.concat(demPath, tostring(i) .. '.mat'), 'dem_sample');
		if (typeInput == 'all') or (typeInput == 'rgb') then
			ortho = ortho:permute(3, 1, 2);
		elseif (typeInput == 'osavi') then
			ortho = unsqueeze:forward(ortho);
			ortho:mul(mult_osavi);
		end
		dem = unsqueeze:forward(dem);
		inputsCpu = torch.cat(ortho, dem, 1);
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

