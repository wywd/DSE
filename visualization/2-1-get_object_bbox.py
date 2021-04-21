import numpy as np
from skimage.measure import label
from skimage.measure import regionprops
from scipy.io import loadmat
from skimage import transform

def get_boundingbox(labels):
    regions = regionprops(label(labels))
    max_number = 0
    bbox = []
    for single_region in regions:
        if single_region.area > max_number:
            max_number = single_region.area
            bbox = single_region.bbox
    return bbox

def main():
    Activation_Map = loadmat("output/cub_activation_map.mat")["x"]

    predicted_bbox = []
    for single_id in range(1,5795):
        if single_id%500 == 0: print(f"total:5794, now:{single_id}")
        Activation_Map_image = Activation_Map[single_id-1,:,:]
        Activation_Map_image = transform.resize(Activation_Map_image, (448,448))
        predicted_label = (Activation_Map_image > (0.43 * np.max(Activation_Map_image))).astype(np.int32)
        predicted_bbox.append(get_boundingbox(predicted_label))
    np.savez("output/predicted_bbox.npz",bbox=predicted_bbox)

if __name__ == "__main__":
    main()
