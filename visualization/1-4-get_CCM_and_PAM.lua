--
--  extracts features using a trained model
--

require 'torch'
require 'paths'
require 'nn'
require 'cudnn'
require 'cunn'
require 'image'
local t = require 'transforms'
local ffi = require 'ffi'

imageInfo = torch.load('data/cubsphere.t7')
list_of_filenames = imageInfo['val']['imagePath']
image_base = paths.concat(imageInfo['basedir'],'val')

-- Load the model
local model = torch.load('model/model_best.t7'):cuda()

-- Remove the fully connected layer
assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
model:remove(#model.modules)
model:remove(#model.modules)
model:remove(#model.modules)
model:remove(#model.modules)
model:remove(#model.modules)

-- Evaluate mode
model:evaluate()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(512),
   t.ColorNormalize(meanstd),
   t.CenterCrop(448),
}

local features

for i=1,5794 do
   print(i)
   local img_batch = torch.FloatTensor(1, 3, 448, 448) -- batch numbers are the 3 channels and size of transform 

   image_name=ffi.string(list_of_filenames[i]:data())
   local img=image.load(paths.concat(image_base,image_name), 3, 'float')
   img=transform(img)
   img_batch[{1, {}, {}, {} }] = img

   local output = model:forward(img_batch:cuda())

   if not features then
       features = torch.FloatTensor(5794, 8192, 14, 14):zero()
   end
       features[{ i, {}, {}, {} } ]:copy(output)
end

print('extract features done')

labels=torch.load('data/cubsphere_labels.t7')  --5794 vector
params=torch.load('data/cub_params.t7')  --200*8192 array

-------------------Class Contribution Map---------------------

contribution_matrix=torch.FloatTensor(5794, 14, 14):zero()
for i=1,5794 do
  param=params[labels[i]]
  for j=1,14 do
    for k=1,14 do
      feature=features[{ i, {}, j, k }]
      contribution_matrix[i][j][k]=torch.dot(feature,param)/torch.norm(param)
    end
  end
end
torch.save('output/cub_contribution_map.t7',contribution_matrix)
print('contribution map done')

-------------------Phase Activation Map---------------------

activation_matrix=torch.FloatTensor(5794, 14, 14):zero()
for i=1,5794 do
  param=params[labels[i]]
  for j=1,14 do
    for k=1,14 do
      feature=features[{ i, {}, j, k }]
      x_norm=torch.norm(feature)
      cos_x_w=torch.dot(feature,param)/(torch.norm(feature)*torch.norm(param))
      beta=0.4
      gamma=0.1
      if cos_x_w < 0 then
        activation_matrix[i][j][k]=0
      else
        activation_matrix[i][j][k]=torch.pow(x_norm,beta)*torch.pow(cos_x_w,gamma)
      end
    end
  end
end
torch.save('output/cub_activation_map.t7',activation_matrix)
print('activation map done')
