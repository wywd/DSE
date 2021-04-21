import os
import shutil

image_root = "CUB_200_2011/images"
new_root = "CUB"
with open("test_images.txt", "r") as f:
    i = 0
    for line in f.readlines():
        i = i + 1
        line = line.rstrip("\n").strip(" ")
        path, category = line.split(" ")
        file_tmp = os.path.join(image_root, path)
        file_name = os.path.basename(file_tmp)
        category = os.path.basename(os.path.dirname(file_tmp))
        path_new = os.path.join(new_root, "val", category)
        if not os.path.exists(path_new):
            os.mkdir(path_new)
        file_new = os.path.join(path_new, file_name)
        shutil.copyfile(file_tmp, file_new)
        print(i)
