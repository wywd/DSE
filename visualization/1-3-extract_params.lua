require 'nn'
require 'cunn'
require 'cudnn'

model = torch.load('model/model_best.t7')

params,gradparams=model:parameters();

param = params[420];

torch.save('data/cub_params.t7', param);
