import os
import sys
import numpy as np
from skimage import io as imgio

def preprocess(img,xy):
    img_h,img_w = img.shape[:2]

    if img_h < img_w:
        scale = 512 / img_h
    else:
        scale = 512 / img_w

    low_x,low_y = img_w*scale/2-224,img_h*scale/2-224
    high_x,high_y = low_x + 447, low_y + 447

    processed_xy = [0,0]
    flag = 1
    if low_x > xy[0]*scale or xy[0]*scale >= high_x:
        flag = 0
    if low_y > xy[1]*scale or xy[1]*scale >= high_y:
        flag = 0
    processed_xy[0] = int(min(max(low_x,xy[0]*scale),high_x)-low_x)
    processed_xy[1] = int(min(max(low_y,xy[1]*scale),high_y)-low_y)

    return processed_xy,flag

def main():
    # load image ids
    print("start to load image ids...")
    image_ids = []
    with open("data/CUB_200_2011/train_test_split.txt") as f:
        for single_line in f.readlines():
            single_line = single_line.rstrip("\n")
            image_id,is_training = single_line.split(" ")
            if is_training == "0":
                image_ids.append(image_id)

    # load image_paths
    print("start to load image paths...")
    img_paths = []
    image_id_index = 0
    with open("data/CUB_200_2011/images.txt") as f:
        for single_line in f.readlines():
            single_line = single_line.rstrip("\n")
            image_id,img_path = single_line.split(" ")
            if image_id == image_ids[image_id_index]:
                img_paths.append(os.path.join("data","CUB_200_2011","images",img_path))
                image_id_index += 1

    # load boundingbox
    print("start to load bbox ...")
    bbox = []
    image_id_index = 0
    with open("data/CUB_200_2011/bounding_boxes.txt") as f:
        for single_line in f.readlines():
            single_line = single_line.rstrip("\n")
            image_id,x,y,w,h, = single_line.split(" ")
            if image_id == image_ids[image_id_index]:
                bbox.append([float(x),float(y),float(x)+float(w),float(y)+float(h)])
                image_id_index += 1

    # proprocess bbox
    print("start to preprocess bbox ...")
    processed_bbox = []
    for index,single_bbox in enumerate(bbox):
        if index%500 == 0: print(f"total:{len(bbox)},now:{index}")
        processed_bbox.append([])
        img = imgio.imread(img_paths[index])
        processed_bbox[-1].extend(preprocess(img,single_bbox[:2])[0])
        tmp = preprocess(img,single_bbox[2:])[0]
        processed_bbox[-1].extend([tmp[0]+1, tmp[1]+1])

    # save processed bbox
    np.savez("data/processed_label/processed_bbox.npz",image_ids=image_ids,bbox=processed_bbox)

    # load parts xy
    print("start to load xy for parts ...")
    parts_xy = []
    image_id_index = 0
    part_count = 0
    with open("data/CUB_200_2011/parts/part_locs.txt") as f:
        for single_line in f.readlines():
            single_line = single_line.rstrip("\n")
            image_id,part_index,part_x,part_y,visualable = single_line.split(" ")

            if image_id == image_ids[image_id_index]:
                if part_count == 0: parts_xy.append([])
                parts_xy[-1].append([float(part_x),float(part_y),int(visualable)])
                part_count += 1
                if part_count == 15: 
                    image_id_index += 1
                    part_count = 0

    # preprocess xy
    print("start to preprocess xy ...")
    processed_xy = []
    for index,part_xy in enumerate(parts_xy):
        if index%500 == 0: print(f"total:{len(parts_xy)},now:{index}")
        processed_xy.append([])
        img = imgio.imread(img_paths[index])
        for part_index in range(15):
        	x_y,flag = preprocess(img,part_xy[part_index][:2])
        	processed_xy[-1].append([x_y[0],x_y[1],int(flag)])
    # save processed xy
    np.savez("data/processed_label/processed_xy.npz",xy=processed_xy)

if __name__ == "__main__":
    main()
