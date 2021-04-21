import numpy as np

def get_results(label_bbox,predicted_bbox,th_number=100):

    # miou
    intersection_area = 0 
    union_area = 0 
    # accu
    positive_count = [0 for _ in range(th_number)]
    total_coount = 0
    for index in range(len(label_bbox)):
        single_label_bbox = label_bbox[index]
        single_predicted_bbox = predicted_bbox[index]
        l_x0,l_y0,l_x1,l_y1 = single_label_bbox
        p_y0,p_x0,p_y1,p_x1 = single_predicted_bbox

        S_label = (l_y1 - l_y0) * (l_x1 - l_x0)
        S_predict = (p_y1 - p_y0) * (p_x1 - p_x0)
        sum_area = S_label + S_predict

        left_line = max(l_x0, p_x0)
        right_line = min(l_x1, p_x1)
        top_line = max(l_y0, p_y0)
        bottom_line = min(l_y1, p_y1)

        if left_line >= right_line or top_line >= bottom_line:
            overlap_area = 0
        else:
            overlap_area = (right_line - left_line) * (bottom_line - top_line)

        single_iou = overlap_area/(sum_area-overlap_area)
        intersection_area += overlap_area
        union_area += (sum_area - overlap_area)

        total_coount += 1
        for th_step in range(th_number):
            if single_iou >  th_step / th_number:
                positive_count[th_step] += 1

    return intersection_area / union_area, [positive_count[index]/total_coount for index in range(th_number)]

def main():
    # load label
    label_bbox = np.load("data/processed_label/processed_bbox.npz",allow_pickle=True)["bbox"]
    # load predicted
    predicted_bbox = np.load("output/predicted_bbox.npz",allow_pickle=True)["bbox"]

    miou_for_ver,accu_for_ver = get_results(label_bbox,predicted_bbox)

    print(f"miou:{miou_for_ver:.4}")

    # accu for different threshold
    accu = []
    for th_step in range(0,100):
        accu.append(f"{accu_for_ver[th_step]:.4}")
    print("accu:")
    print("\t".join(accu))

if __name__ == "__main__":
    main()
