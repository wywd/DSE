--
--  extracts the images after transforms
--

require 'torch'
require 'paths'
require 'image'
local t = require 'transforms'
local ffi = require 'ffi'

imageInfo = torch.load('data/cubsphere.t7')
list_of_filenames = imageInfo['val']['imagePath']
image_base = paths.concat(imageInfo['basedir'],'val')

local transform = t.Compose{
   t.Scale(512),
   t.CenterCrop(448),
}

new_image_base='output/cub_input_image/'
for i=1,5794 do
   print(i)
   image_name=ffi.string(list_of_filenames[i]:data())
   local img=image.load(paths.concat(image_base,image_name), 3, 'float')
   img=transform(img)
   image_path=new_image_base..tostring(i)..'.jpg'
   image.save(image_path,img)
end

print('saved images after transforms')
