require 'torch';
require 'image';
require 'paths';
torch.setdefaulttensortype('torch.FloatTensor'); -- set default
local numThreads = 12;

--local rgbPath = '/media/aich/DATA/databases/leaf_cvppp2017/seg_blur/rgb/';
local inBasePath = '../../data/01_bin_seg/train';
local inRgbPath = 'rgb_aug';
local inGtPath = 'gt_aug';

local outBasePath = '../../data/phenowheat_seg_dat/';
local outRgbPath = 'rgb_aug';
local outGtPath = 'gt_aug';

local numDirs = #paths.dir(paths.concat(inBasePath, inRgbPath)) - 2;

local inRgbPath = paths.concat(inBasePath, inRgbPath);
local inGtPath = paths.concat(inBasePath, inGtPath);
local outRgbPath = paths.concat(outBasePath, outRgbPath);
local outGtPath = paths.concat(outBasePath, outGtPath);
os.execute("rm -rf " .. outBasePath);
os.execute("mkdir -pv " .. outRgbPath);
os.execute("mkdir -pv " .. outGtPath);

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
		local inRgbPath = inRgbPath;
		local inGtPath = inGtPath;
		local outRgbPath = outRgbPath;
		local outGtPath = outGtPath;
		torch.setdefaulttensortype('torch.FloatTensor');
	    tid = idx;
	    print(string.format('Starting donkey with id: %d', tid));
	end
);

local jobDone = 0 -- this must be local
for d = 1, numDirs, 1 do
	donkeys:addjob(
		function()
			local inRgbPath = paths.concat(inRgbPath, d);
			local inGtPath = paths.concat(inGtPath, d);
			local outRgbPath = paths.concat(outRgbPath, d);
			local outGtPath = paths.concat(outGtPath, d);

			local fileList = paths.dir(inRgbPath);
			os.execute("mkdir " .. outRgbPath);
			os.execute("mkdir " .. outGtPath);

			local numFiles = #fileList - 2;
			for i = 1, numFiles, 1 do
				print(string.format("subdir = %d, file = %d", d, i));
				local im = image.load(paths.concat(inRgbPath, tostring(i) .. '.png'), 3, 'byte');
				local gt = image.load(paths.concat(inGtPath, tostring(i) .. '.png'), 1, 'byte');
				gt[gt:eq(255)] = 2; -- needed for training with Spatial-CE
				gt[gt:eq(0)] = 1; -- needed for training with Spatial-CE
				torch.save(paths.concat(outRgbPath, tostring(i) .. '.dat'), im);
				torch.save(paths.concat(outGtPath, tostring(i) .. '.dat'), gt);
				-- check loader
				while (not pcall ( function()
							-- check if loading is successful
							local loadChecker = torch.load(paths.concat(outRgbPath, tostring(i) .. '.dat'));
							loadChecker = torch.load(paths.concat(outGtPath, tostring(i) .. '.dat'));
						end -- end of function
						)) do -- end of pcall and while condition
					torch.save(paths.concat(outRgbPath, tostring(i) .. '.dat'), im);
					torch.save(paths.concat(outGtPath, tostring(i) .. '.dat'), gt);
				end
			end
			collectgarbage();
			collectgarbage();
			return __threadid, (#fileList-2), d;
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
