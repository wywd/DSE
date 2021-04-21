local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 ResNet fine-tuning script')
   cmd:text()
   cmd:text('Options:')
   ------------ arg options ----------------------------------------------
   cmd:option('-dataset',         'cubsphere', 'Options: cubsphere | airsphere | carsphere')
   cmd:option('-preprocess',      1,    'Options: 1 | 2 | 3 ,How to preprocess the dataset')
   cmd:option('-depth',           101,        'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
   cmd:option('-resetClassifier', 401,    'How to reset the model for fine-tuning')
   cmd:option('-batchSize',       16,        'mini-batch size (1 = pure stochastic)')
   cmd:option('-LR',              0.001,      'initial learning rate')
   cmd:option('-nEpochs',         400,       'Number of total epochs to run')
   -----------File path options--------------------------------------------
   cmd:option('-retrain',         'tmp/models/resnet-101.t7',   'Path to model to retrain with')
   cmd:option('-save',            'tmp/results/cub_fine_tuned_model', 'Directory in which to save checkpoints')
   ------------- Model options --------------------
   cmd:option('-shareGradInput',  'true',   'Share gradInput tensors to reduce memory usage')
   cmd:option('-projDimension',    8192,    'The dimension for convolution project')
   ------------------------------------------------------------------
   ------------- General options-Data options------------------------
   cmd:option('-data',       '',       'Path to dataset')
   cmd:option('-nClasses',   0,        'Number of classes in the dataset')
   cmd:option('-nGPU',       0,        'Number of GPUs to use by default')
   cmd:option('-nThreads',   0,        'number of data loading threads')
   cmd:option('-gen',        'tmp/dataset',      'Path to save generated files')
   ------------- fixed options ------------------------
   cmd:option('-testOnly',        'false',   'Run on validation set only')
   cmd:option('-epochNumber',     1,         'Manual epoch number (useful on restarts)')
   cmd:option('-optimState',      'none',    'Path to an optimState to reload from')
   cmd:option('-momentum',        0.9,       'momentum')
   cmd:option('-weightDecay',     1e-4,      'weight decay')
   cmd:option('-netType',         'resnet',  'Options: resnet | preresnet')
   cmd:option('-shortcutType',    'B',       'Options: A | B | C')
   cmd:option('-manualSeed',      0,         'Manually set RNG seed')
   cmd:option('-backend',         'cudnn',   'Options: cudnn | cunn')
   cmd:option('-cudnn',           'fastest', 'Options: fastest | default | deterministic')
   cmd:option('-precision',       'single',  'Options: single | double | half')
   cmd:option('-tenCrop',         'false',   'Ten-crop testing')
   cmd:option('-optnet',          'false',   'Use optnet to reduce memory usage')
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end

   if opt.dataset == 'cubsphere' then
      -- Handle the most common case of missing -data flag
      opt.data = opt.data == '' and 'tmp/dataset/CUB' or opt.data
      local trainDir = paths.concat(opt.data, 'train')
      if not paths.dirp(opt.data) then
         cmd:error('error: missing CUB data directory')
      elseif not paths.dirp(trainDir) then
         cmd:error('error: CUB missing `train` directory: ' .. trainDir)
      end
      opt.nClasses = opt.nClasses == 0 and 200 or opt.nClasses
      opt.nGPU = opt.nGPU == 0 and 1 or opt.nGPU
      opt.nThreads = opt.nThreads == 0 and 3 or opt.nThreads
   end

   if opt.depth == 101 then 
      opt.retrain = opt.retrain == 'none' and 'tmp/models/resnet-101.t7' or opt.retrain
   end

   if opt.precision == nil or opt.precision == 'single' then
      opt.tensorType = 'torch.CudaTensor'
   elseif opt.precision == 'double' then
      opt.tensorType = 'torch.CudaDoubleTensor'
   elseif opt.precision == 'half' then
      opt.tensorType = 'torch.CudaHalfTensor'
   else
      cmd:error('unknown precision: ' .. opt.precision)
   end

   if opt.shareGradInput and opt.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end

   return opt
end

return M
