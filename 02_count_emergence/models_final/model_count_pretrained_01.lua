require 'torch';
require 'cutorch';
require 'cudnn';
require 'nn';
require 'cunn';
require 'dpnn';

cudnn.benchmark = true;

model = torch.load('./models/model_count_resnet_01_epoch_50.t7');
model:cuda();

criterion = nn.AbsCriterion();
criterion:cuda();

print(model);
print(criterion);

