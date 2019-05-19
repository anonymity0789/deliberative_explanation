image_list_path = './CUB_200_2011/CUB_200_2011/images.txt'
class_list_path = './CUB_200_2011/CUB_200_2011/image_class_labels.txt'
tr_te_split_list_path = './CUB_200_2011/CUB_200_2011/train_test_split.txt'


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


save_train_list = './CUB_200_2011/CUB_200_2011/CUB200_gt_tr.txt'
save_test_list = './CUB_200_2011/CUB_200_2011/CUB200_gt_te.txt'



num_tr = 0
fl = open(save_train_list, 'w')
for i in range(len(imlist)):
    if tr_te_list[i] == '1':
        example_info = '/data6/peiwang/datasets/CUB_200_2011/images/' + imlist[i] + " " + classlist[i] + " " + str(num_tr)
        fl.write(example_info)
        fl.write("\n")
        num_tr = num_tr + 1

fl.close()

num_te = 0
fl = open(save_test_list, 'w')
for i in range(len(imlist)):
    if tr_te_list[i] == '0':
        example_info = '/data6/peiwang/datasets/CUB_200_2011/images/' + imlist[i] + " " + classlist[i] + " " + str(num_te)
        fl.write(example_info)
        fl.write("\n")
        num_te = num_te + 1

fl.close()
