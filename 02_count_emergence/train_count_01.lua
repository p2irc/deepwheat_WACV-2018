-- run with CUDA_VISIBLE_DEVICES=0 th <yourscript.lua>
require 'paths';
require 'optim';
require 'torch';
require 'cutorch';
require 'cudnn';
require 'nn';
require 'cunn';
local matio = require 'matio';
local Threads = require 'threads';
Threads.serialization('threads.sharedserialize');

torch.setdefaulttensortype('torch.FloatTensor'); -- set default

--cutorch.setDevice(2);

local outModelFile = "model_count_01";

os.execute("mkdir logs"); -- create log directory from terminal

paths.dofile('./models_final/model_count_01.lua');
--paths.dofile('./models_plain_final/model_count_01.lua');
--paths.dofile('./models_inception_final/model_count_01.lua');
--require './models/model_count_pretrained_01.lua';
model:training();

local inSize, inDim, outSize = 224, 3, 224;
local batchSize = 8;
local numEpochs = 100;
----------------------------------------------------
local pathTrainData = '../../data/02_count_emergence/train/898_emergence_dat';
local pathTrainLabels = '../../data/02_count_emergence/train';
----------------------------------------------------
local labelFileName = 'count_gt_aug.mat';

local numThreads = batchSize; -- cpu thread

-- specify log file
trainLogger = optim.Logger('./logs/train_count_resnet_01_part1.log');

--[[
local optimState = { -- for sgd
	learningRate = 0.01,
	learningRateDecay = 0.0,
	momentum = 0.9,
	dampening = 0.0,
	weightDecay = 0.0001
};
--]]

local optimState = { -- for sgd
	learningRate = 0.0001, 
	weightDecay = 0
}

-- =================== System inputs end here ==============================

-- GPU inputs (preallocate)
local inputsCpu = torch.Tensor(batchSize, inDim, inSize, inSize);
local labelsCpu = torch.Tensor(batchSize);
local inputs = torch.CudaTensor(batchSize, inDim, inSize, inSize);
local labels = torch.CudaTensor(batchSize);

local numFiles = #paths.dir(pathTrainData) - 2;
local numBatches = torch.ceil(numFiles/batchSize);
local weight_mult = 1/numBatches;
local overestim_epoch, underestim_epoch, loss_epoch;
local printVal = false;
function train()

	donkeys = Threads( -- initializing parallel threads
		numThreads,
		function()
			require 'torch';
		end,
		function(idx)
			tid = idx;
			pathTrainData = pathTrainData;
			print(string.format('Starting donkey with id: %d', tid));
		end
	);

	-- load all labels from mat file
	local dataLabels = torch.squeeze(matio.load(paths.concat(pathTrainLabels, labelFileName), 'counts'));
	print(string.format("Total Emergence Count = %d", dataLabels:sum()));

	for epoch = 1, numEpochs, 1 do
		local time_epoch = torch.Timer();
		overestim_epoch, underestim_epoch, loss_epoch = 0, 0, 0;
		local fileList = torch.randperm(numFiles); -- load different permutations in different epochs
		local batch_st, batch_ed = 0, 0;
		for batch = 1, numBatches, 1 do
			if batch ~= numBatches then
				batch_st = batch_ed + 1;
				batch_ed = batch_ed + batchSize;
			else
				batch_ed = numFiles;
				batch_st = numFiles - batchSize + 1;
			end

			------------ load files for a single batch -----------
			-- local jobDone = 0;
			for fileId = batch_st, batch_ed, 1 do
				donkeys:addjob(
					function()
						local placeId = fileId - batch_st + 1;
						inputsCpu[placeId] = torch.load(paths.concat(pathTrainData, tostring(fileList[fileId]) .. '.dat'));
						labelsCpu[placeId] = dataLabels[fileList[fileId]];
						collectgarbage();
						collectgarbage();
						return __threadid;
					end,
					function(id)
--						print(string.format("loaded file %d (ran on thread ID %d)", batch, id));
--						jobDone = jobDone + 1
					end
				);
			end
			donkeys:synchronize();
			trainBatch(epoch, batch);
		end

		-- clear the intermediate states in the model before saving to disk
	   	-- this saves lots of disk space
	   	model:clearState();
		local fullModelFile = "./models/" .. outModelFile .. "_epoch_" .. tostring(epoch) .. ".t7";
	   	torch.save(fullModelFile , model) -- save model after each epoch

	   	trainLogger:add{
			['Epoch '] = epoch,
			['LearningRate '] = optimState.learningRate,
--			['Momentum '] = optimState.momentum,
			['WeightDecay '] = optimState.weightDecay,
			['TotalCount '] = dataLabels:sum(),
			[' Overestimate '] = overestim_epoch,
			[' Underestimate '] = underestim_epoch,
			[' Loss(SL1) '] = loss_epoch,
	   	}
	   	print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f,\t'
                          .. 'Loss: %.2f, \t '
                          .. 'Underestimate: %.2f, \t'
                          .. 'Overestimate: %.2f, \t',
                       epoch, time_epoch:time().real, loss_epoch, underestim_epoch, overestim_epoch));
		collectgarbage();

	end -- end of training epochs

	donkeys:terminate();
end
---------------- end of function train() ------------------

local timer = torch.Timer()
local dataTimer = torch.Timer()
local parameters, gradParameters = model:getParameters();
--------------------------------------------------------
function trainBatch(epoch, batch)
	cutorch.synchronize()
	collectgarbage()
	local dataLoadingTime = dataTimer:time().real
	timer:reset()

	--inputs:resize(inputsCpu:size()):copy(inputsCpu)
	--labels:resize(labelsCpu:size()):copy(labelsCpu)
	inputs:copy(inputsCpu);
	labels:copy(labelsCpu);

	local err, outputs
	feval = function(x)
		model:zeroGradParameters()
		outputs = model:forward(inputs);
		err = criterion:forward(outputs, labels);
		local gradOutputs = criterion:backward(outputs, labels);
		model:backward(inputs, gradOutputs)
		return err, gradParameters
	end

--	optim.sgd(feval, parameters, optimState)
	optim.adam(feval, parameters, optimState)
	cutorch.synchronize();

	outputs = torch.squeeze(outputs);
	local dif = outputs - labels;
	local tmp = dif[dif:lt(0)]; -- less than
	local underest, overest = 0, 0, 0;
	if tmp:dim() > 0 then
		underest = torch.abs(tmp:sum());
		underestim_epoch = underestim_epoch + underest;
	end
	tmp = dif[dif:gt(0)]; -- greater than
	if tmp:dim() > 0 then
		overest = tmp:sum();
		overestim_epoch = overestim_epoch + overest;
	end
	loss_epoch = loss_epoch + err * weight_mult;

	-- print information
	print(('Epoch: [%d], Batch: [%d/%d], \tTime %.3f, Error(SL1) %.4f, Count(GT) %d, Difference(Abs) %.4f, Overest %.4f, Underest %.4f, DataLoadingTime %.3f'):format(epoch, batch, numBatches, timer:time().real, err, labels:sum(), (underest + overest), overest, underest, dataLoadingTime));

   	dataTimer:reset();
end
-------------------------------------------------------
