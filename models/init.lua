--
--  Generic model creating.
--

require 'nn'
require 'cunn'
require 'cudnn'

local M = {}

function M.setup(opt)
   local model
   if opt.retrain ~= 'none' then
      assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
      print('Loading model from file: ' .. opt.retrain)
      model = torch.load(opt.retrain):type(opt.tensorType)
      model.__memoryOptimized = nil
   else
      print('=> Creating model from file: models/' .. opt.netType .. '.lua')
      model = require('models/' .. opt.netType)(opt)
   end

   -- Remove any DataParallelTable
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   if opt.shareGradInput then
      M.shareGradInput(model, opt)
   end

   -- Resetting the classifier
   if opt.resetClassifier == 401 then
      print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier and sphere feature')
      local orig = model:get(#model.modules)
      assert(torch.type(orig) == 'nn.Linear',
         'expected last layer to be fully connected')
      nFeatures = orig.weight:size(2)
      model:remove(#model.modules)
      model:remove(#model.modules)
      model:remove(#model.modules)
      model:add(cudnn.SpatialConvolution(nFeatures, opt.projDimension, 1, 1):type(opt.tensorType))
      model:add(nn.Reshape(opt.projDimension, 196, true):type(opt.tensorType))
      model:add(nn.Sum(2, 2, false, true):type(opt.tensorType))
      model:add(nn.Normalize(2):type(opt.tensorType))
      model:add(nn.Mul():type(opt.tensorType))
      local newlayer = model:get(#model.modules)
      assert(torch.type(newlayer) == 'nn.Mul',
         'expected new layer to be nn.mul()')
      newlayer.weight:fill(1)
      model:add(nn.Linear(opt.projDimension, opt.nClasses, false):type(opt.tensorType))
   end

   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:type(opt.tensorType)
   end

   local criterion = nn.CrossEntropyCriterion():type(opt.tensorType)
   return model, criterion
end

function M.shareGradInput(model, opt)
   local function sharingKey(m)
      local key = torch.type(m)
      if m.__shareGradInputKey then
         key = key .. ':' .. m.__shareGradInputKey
      end
      return key
   end

   -- Share gradInput for memory efficient backprop
   local cache = {}
   model:apply(function(m)
      local moduleType = torch.type(m)
      if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
         local key = sharingKey(m)
         if cache[key] == nil then
            cache[key] = torch[opt.tensorType:match('torch.(%a+)'):gsub('Tensor','Storage')](1)
         end
         m.gradInput = torch[opt.tensorType:match('torch.(%a+)')](cache[key], 1, 0)
      end
   end)
   for i, m in ipairs(model:findModules('nn.ConcatTable')) do
      if cache[i % 2] == nil then
         cache[i % 2] = torch[opt.tensorType:match('torch.(%a+)'):gsub('Tensor','Storage')](1)
      end
      m.gradInput = torch[opt.tensorType:match('torch.(%a+)')](cache[i % 2], 1, 0)
   end
end

return M
