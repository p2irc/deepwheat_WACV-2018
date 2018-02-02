require 'torch';
require 'image';
require 'paths';
torch.setdefaulttensortype('torch.FloatTensor'); -- set default
local numThreads = 12;

local basePath = '../../data/phenowheat_seg_dat/';
local rgbPath = 'rgb_aug';
local gtPath = 'gt_aug';

local rgbPath = paths.concat(basePath, rgbPath);
local gtPath = paths.concat(basePath, gtPath);

local numDirs = #paths.dir(rgbPath) - 2;

local Threads = require 'threads';
Threads.serialization('threads.sharedserialize');

donkeys = Threads(
	numThreads,
	function()
		require 'torch';
		require 'image';
		require 'paths';
	end,
	function(idx)
		local rgbPath = rgbPath;
		local gtPath = gtPath;
		torch.setdefaulttensortype('torch.FloatTensor');
	    tid = idx;
		torch.manualSeed(tid);
	    print(string.format('Starting donkey with id: %d', tid));
	end
);

local jobDone = 0 -- this must be local
for d = 1, numDirs, 1 do
	donkeys:addjob(
		function()
			local rgbPath = paths.concat(rgbPath, d);
			local gtPath = paths.concat(gtPath, d);

			local numFiles = #paths.dir(rgbPath) - 2;
			if (numFiles % 2) == 1 then -- odd checker to assure consecutive swap
				numFiles = numFiles - 1;
			end
			for i = 1, numFiles, 2 do
				local fileShuffleList = torch.randperm(numFiles);
				print(string.format("subdir = %d, file = %d, fileId1 = %d, fileId2 = %d", d, i, fileShuffleList[i], fileShuffleList[i+1]));
				local im1 = torch.load(paths.concat(rgbPath, tostring(fileShuffleList[i]) .. '.dat'));
				local gt1 = torch.load(paths.concat(gtPath, tostring(fileShuffleList[i]) .. '.dat'));
				local im2 = torch.load(paths.concat(rgbPath, tostring(fileShuffleList[i+1]) .. '.dat'));
				local gt2 = torch.load(paths.concat(gtPath, tostring(fileShuffleList[i+1]) .. '.dat'));
				-- shuffle consecutive files
				torch.save(paths.concat(rgbPath, tostring(fileShuffleList[i]) .. '.dat'), im2);
				torch.save(paths.concat(gtPath, tostring(fileShuffleList[i]) .. '.dat'), gt2);
				torch.save(paths.concat(rgbPath, tostring(fileShuffleList[i+1]) .. '.dat'), im1);
				torch.save(paths.concat(gtPath, tostring(fileShuffleList[i+1]) .. '.dat'), gt1);
				-- check loader
				while (not pcall ( function()
							-- check if loading is successful
							local loadChecker = torch.load(paths.concat(rgbPath, tostring(fileShuffleList[i]) .. '.dat'));
							loadChecker = torch.load(paths.concat(gtPath, tostring(fileShuffleList[i]) .. '.dat'));			
							loadChecker = torch.load(paths.concat(rgbPath, tostring(fileShuffleList[i+1]) .. '.dat'));
							loadChecker = torch.load(paths.concat(gtPath, tostring(fileShuffleList[i+1]) .. '.dat'));			
						end -- end of function
						)) do -- end of pcall and while condition
					torch.save(paths.concat(rgbPath, tostring(fileShuffleList[i]) .. '.dat'), im2);
					torch.save(paths.concat(gtPath, tostring(fileShuffleList[i]) .. '.dat'), gt2);
					torch.save(paths.concat(rgbPath, tostring(fileShuffleList[i+1]) .. '.dat'), im1);
					torch.save(paths.concat(gtPath, tostring(fileShuffleList[i+1]) .. '.dat'), gt1);
				end			
			end
			collectgarbage();
			collectgarbage();
			return __threadid, numFiles, d;
		end,
		function(id, numFiles, dirId)
			print(string.format("Finished directory = %d, #files = %d (ran on thread ID %d)", dirId, numFiles, id));
			jobDone = jobDone + 1;
		end
	);
end

donkeys:synchronize();
print(string.format('%d jobs done', jobDone));
donkeys:terminate();

