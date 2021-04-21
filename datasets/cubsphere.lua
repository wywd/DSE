--
--  CUB dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local CUBsphereDataset = torch.class('resnet.CUBsphereDataset', M)

function CUBsphereDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = paths.concat(opt.data, split)
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function CUBsphereDataset:get(i)
   local path = ffi.string(self.imageInfo.imagePath[i]:data())

   local image = self:_loadImage(paths.concat(self.dir, path))
   local class = self.imageInfo.imageClass[i]

   return {
      input = image,
      target = class,
   }
end

function CUBsphereDataset:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
   end)

   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, 3, 'float')
   end

   return input
end

function CUBsphereDataset:size()
   return self.imageInfo.imageClass:size(1)
end

local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

function CUBsphereDataset:preprocess()
   if self.split == 'train' then
      if self.opt.preprocess == 1 then
         return t.Compose{
            t.Scale(512),
            t.RandomSizedCrop(448),
            t.ColorNormalize(meanstd),
            t.HorizontalFlip(0.5),
         }
      else
         error('invalid preprocess: ' .. self.opt.preprocess)
      end
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      if self.opt.preprocess == 1 then
         return t.Compose{
            t.Scale(512),
            t.ColorNormalize(meanstd),
            Crop(448),
         }
      else
         error('invalid preprocess: ' .. self.opt.preprocess)
      end
   else
      error('invalid split: ' .. self.split)
   end
end

return M.CUBsphereDataset
