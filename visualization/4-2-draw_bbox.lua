require 'image'
matfile=require('matio')

base_path='output/cub_input_image/'
new_path='output/cub_attention_image/'
data_object=matfile.load('output/draw_bbox_results_object.mat')['object']
data_local=matfile.load('output/draw_bbox_results_local.mat')

for i=1,5794 do
	origin_image_path=base_path..tostring(i)..'.jpg'
	origin_image=image.load(origin_image_path, 3, 'float')

	object_bbox=data_object[i]
	new_image=image.drawRect(origin_image, object_bbox[2]+1, object_bbox[1]+1, object_bbox[4], object_bbox[3], {lineWidth = 3, color = {255, 0, 0}})

	local_bbox=data_local[tostring(i)]
	for j=1,local_bbox:size()[1] do
		if local_bbox[j][1] >= 5 then
			contribute_rate_sum=local_bbox[j][1]
			contribute_rate_sum=contribute_rate_sum/100
			new_image=image.drawRect(new_image, local_bbox[j][4]+1, local_bbox[j][3]+1, local_bbox[j][6], local_bbox[j][5], {lineWidth = 2, color = {0, 255, 0}})
			new_image=image.drawText(new_image, tostring(contribute_rate_sum), local_bbox[j][4]+1,local_bbox[j][3]+1,{color = {255, 255, 255},bg = {0, 0, 0}, size = 2.5})
		end
	end
	new_image_path=new_path..tostring(i)..'.jpg'
	image.save(new_image_path,new_image)
end
