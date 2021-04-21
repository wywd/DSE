require 'torch'
require 'paths'
require 'optim'
require 'nn'
local opts = require 'opts'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- Create model
local model, criterion = models.setup(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt)

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

local startEpoch = opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testTop1, testTop5 = trainer:test(epoch, valLoader)

   local bestModel = false
   if testTop1 < bestTop1 then
      bestModel = true
      bestTop1 = testTop1
      bestTop5 = testTop5
      print(' * Best model ', testTop1, testTop5)
   end

   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
end

print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))
