require 'torch';
require 'paths';
require 'image';

torch.setdefaulttensortype('torch.FloatTensor');

local dataPath = '/media/aich/DATA/898_project/emergence/train_seg/rgb';
local imgList = paths.dir(dataPath);
local mult = 1/(#imgList - 2);
local mean_image = 0;

for i = 1, #imgList, 1 do
	print(i);
	if (imgList[i]~='.') and (imgList[i]~='..') then
		mean_image = mean_image + mult *
				image.load(paths.concat(dataPath, imgList[i]), 3, 'float'):mean();
	end
end

print(string.format('mean = %f', mean_image));
torch.save('mean_seg.dat', mean_image);
