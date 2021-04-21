import numpy as np
from scipy.io import loadmat
from skimage import transform

def main():
    # load xy for parts
    parts_xy = np.load("data/processed_label/processed_xy.npz")["xy"]

    contribution_map = loadmat("output/cub_contribution_map.mat")["x"]

    parts_value = []
    for single_id in range(1,5795):
        if single_id%500 == 0: print(f"total:5794, now:{single_id}")
        contribution_image = contribution_map[single_id-1,:,:]
        contribution_image = transform.resize(contribution_image, (448,448))
        parts_value.append([])
        sum_of_this_picture = 0
        for single_xy in parts_xy[single_id-1]:
            if single_xy[2] == 1:
                parts_value[-1].append(contribution_image[single_xy[1],single_xy[0]])
            else:
                parts_value[-1].append("-")

        if parts_value[-1][10] != "-":
            if parts_value[-1][6] != "-":
                parts_value[-1][6] = (parts_value[-1][6]+parts_value[-1][10])/2
            else:
                parts_value[-1][6] = parts_value[-1][10]

        if parts_value[-1][11] != "-":
            if parts_value[-1][7] != "-":
                parts_value[-1][7] = (parts_value[-1][7]+parts_value[-1][11])/2
            else:
                parts_value[-1][7] = parts_value[-1][11]

        if parts_value[-1][12] != "-":
            if parts_value[-1][8] != "-":
                parts_value[-1][8] = (parts_value[-1][8]+parts_value[-1][12])/2
            else:
                parts_value[-1][8] = parts_value[-1][12]

        del parts_value[-1][10]
        del parts_value[-1][11]
        del parts_value[-1][12]

        sum_of_this_picture += sum([single_value for single_value in parts_value[-1] if single_value !="-"])
        parts_value[-1] = [f"{parts_value[-1][index]/sum_of_this_picture:.4}" if parts_value[-1][index] != "-" else "-"  for index in range(len(parts_value[-1]))]

    # save values of part
    with open("output/parts_value_contribution.txt","w") as f:
        for part_value in parts_value:
            f.write("\t".join(part_value)+"\n")
    #
    results = []
    with open("test_images.txt","r") as f:
        last_category = "0"
        last_values = [0 for _ in range(12)]
        last_counts = [0 for _ in range(12)]
        sum_of_this_class=0
        for index,line in enumerate(f.readlines()):
            line = line.strip("\n")
            _,cur_category = line.split(" ")
            if cur_category != last_category:
                for index2 in range(12):
                    last_values[index2] /= last_counts[index2] if last_counts[index2] != 0 else 1
                    last_values[index2] = last_values[index2] if last_counts[index2] != 0 else "-"
                last_category = cur_category

                sum_of_this_class = sum([single_value for single_value in last_values if single_value !="-"])
                results.append( [f"{'%.1f'%(100*single_value/sum_of_this_class)}" if single_value != "-" else "-"  for single_value in last_values] )
                last_values = [0 for _ in range(12)]
                last_counts = [0 for _ in range(12)]

            cur_parts_values = [float(single_value) if single_value != "-" else "-" for single_value in parts_value[index]]
            for index2 in range(12):
                last_values[index2] += cur_parts_values[index2] if cur_parts_values[index2] != "-" else 0
                last_counts[index2] += 1 if cur_parts_values[index2] != "-" else 0

    with open("output/class_parts_results.txt","w") as f:
        for part_value in results:
            f.write("\t".join(part_value)+"\n")

if __name__ == "__main__":
    main()
