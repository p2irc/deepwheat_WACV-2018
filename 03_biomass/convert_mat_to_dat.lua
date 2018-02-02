require 'paths';
local matio = require 'matio';
require 'nn';

torch.setdefaulttensortype('torch.FloatTensor');

--local typeInput = 'osavi';
--local typeInput = 'all';
local typeInput = 'rgb';
local pathOrtho = '../../data/03_biomass/train/ortho_aug';
local pathOrthoDat = '../../data/03_biomass/train/ortho_aug_dat';
local pathDem = '../../data/03_biomass/train/dem_aug';
local pathDemDat = '../../data/03_biomass/train/dem_aug_dat';
local pathOsaviMax = '../../data/03_biomass/train/max_osavi.mat';

os.execute("rm -rfv " .. pathOrthoDat);
os.execute("rm -rfv " .. pathDemDat);
os.execute("mkdir -v " .. pathOrthoDat);
os.execute("mkdir -v " .. pathDemDat);

local numFiles = #paths.dir(pathDem) - 2;
print(string.format('Total number of files = %d', numFiles));

unsqueeze = nn.Unsqueeze(1);

local mult_osavi;
if (typeInput == 'osavi') then
	local max_osavi = matio.load(pathOsaviMax, 'max_osavi');
	mult_osavi = 1/max_osavi[1][1];
end

for i = 1, numFiles, 1 do
	print(i);
	local im = matio.load(paths.concat(pathOrtho, tostring(i) .. '.mat'), 'im_sample'); -- no mean subtraction
	local dem = matio.load(paths.concat(pathDem, tostring(i) .. '.mat'), 'dem_sample');
	if (typeInput == 'all') or (typeInput == 'rgb') then
		im = im:permute(3, 1, 2);
	elseif (typeInput == 'osavi') then
		im = unsqueeze:forward(im);
		im:mul(mult_osavi);
	end
	dem = unsqueeze:forward(dem);
	torch.save(paths.concat(pathOrthoDat, tostring(i) .. '.dat'), im);
	torch.save(paths.concat(pathDemDat, tostring(i) .. '.dat'), dem);
end

