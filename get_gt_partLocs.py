import numpy as np


image_list_path = './CUB_200_2011/CUB_200_2011/images.txt'
class_list_path = './CUB_200_2011/CUB_200_2011/image_class_labels.txt'
tr_te_split_list_path = './CUB_200_2011/CUB_200_2011/train_test_split.txt'
part_locs_list_path = './CUB_200_2011/CUB_200_2011/parts/part_locs.txt'


imlist = []
with open(image_list_path, 'r') as rf:
    for line in rf.readlines():
        imindex, impath = line.strip().split()
        imlist.append(impath)

classlist = []
with open(class_list_path, 'r') as rf:
    for line in rf.readlines():
        imindex, label = line.strip().split()
        classlist.append(str(int(label)-1))

tr_te_list = []
with open(tr_te_split_list_path, 'r') as rf:
    for line in rf.readlines():
        imindex, tr_te = line.strip().split()
        tr_te_list.append(tr_te)

part_id_list = []
part_locsX_list = []
part_locsY_list = []
with open(part_locs_list_path, 'r') as rf:
    for line in rf.readlines():
        imindex, part_idx, x_cord, y_cord, uncertainty = line.strip().split()
        part_id_list.append(part_idx)
        part_locsX_list.append(x_cord)
        part_locsY_list.append(y_cord)

save_train_list = './CUB_200_2011/CUB_200_2011/CUB200_gt_tr.txt'
save_test_list = './CUB_200_2011/CUB_200_2011/CUB200_gt_te.txt'
save_partLocs_train_list = './CUB_200_2011/CUB_200_2011/CUB200_partLocs_gt_tr.txt'
save_partLocs_test_list = './CUB_200_2011/CUB_200_2011/CUB200_partLocs_gt_te.txt'

#
# num_tr = 0
# fl = open(save_train_list, 'w')
# for i in range(len(imlist)):
#     if tr_te_list[i] == '1':
#         example_info = './CUB_200_2011/CUB_200_2011/images/' + imlist[i] + " " + classlist[i] + " " + str(num_tr)
#         fl.write(example_info)
#         fl.write("\n")
#         num_tr = num_tr + 1
#
# fl.close()
#
# num_te = 0
# fl = open(save_test_list, 'w')
# for i in range(len(imlist)):
#     if tr_te_list[i] == '0':
#         example_info = './CUB_200_2011/CUB_200_2011/images/' + imlist[i] + " " + classlist[i] + " " + str(num_te)
#         fl.write(example_info)
#         fl.write("\n")
#         num_te = num_te + 1
#
# fl.close()


# save part Locations, <part1 x> <part1 y> <part2 x> <part2 y>,...,<part15 x> <part15 y>, <index>

num_tr = 0
fl = open(save_partLocs_train_list, 'w')
for i in range(len(imlist)):
    if tr_te_list[i] == '1':
        example_info = ""
        for i_part in range(15):
            if int(part_id_list[i*15+i_part]) != i_part+1:
                print('error')
            example_info = example_info + part_locsX_list[i * 15 + i_part] + " " + part_locsY_list[i * 15 + i_part] + " "
        example_info = example_info + " " + str(num_tr)
        fl.write(example_info)
        fl.write("\n")
        num_tr = num_tr + 1
fl.close()

num_te = 0
fl = open(save_partLocs_test_list, 'w')
for i in range(len(imlist)):
    if tr_te_list[i] == '0':
        example_info = ""
        for i_part in range(15):
            if int(part_id_list[i * 15 + i_part]) != i_part + 1:
                print('error')
            example_info = example_info + part_locsX_list[i * 15 + i_part] + " " + part_locsY_list[i * 15 + i_part] + " "
        example_info = example_info + " " + str(num_te)
        fl.write(example_info)
        fl.write("\n")
        num_te = num_te + 1
fl.close()
