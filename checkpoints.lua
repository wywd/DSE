local checkpoint = {}

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

function checkpoint.save(epoch, model, optimState, isBestModel, opt)
   if isBestModel then
      if torch.type(model) == 'nn.DataParallelTable' then
         model = model:get(1)
      end

      model = deepCopy(model):float():clearState()

      torch.save(paths.concat(opt.save, 'model_best.t7'), model)
   end
end

return checkpoint
