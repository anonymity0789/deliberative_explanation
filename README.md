# deliberative_explanation


## Requirements

1. The project was implemented and tested in Python 3.5 and Pytorch 0.4. The higher versions should work after minor modification.
2. Other common modules like numpy, pandas and seaborn for visualization.
3. NVIDIA GPU and cuDNN are required to have fast speeds. For now, CUDA 8.0 with cuDNN 6.0.20 has been tested. The other versions should be working.


## Datasets

[CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [ADE20K](http://sceneparsing.csail.mit.edu/) are used. Please organize them as below after download,

cub200

|_ CUB_200_2011

ade

|_ ADEChallengeData2016
 

## Implementation details

### data preparation

build train/validation/test sets,

```
make_cub_list.py
make_ade_list.py
```

compute similarity for parts on CUB200 and objects on ADE20K,

```
create_similarityMatrix_cls_part.py
create_similarityMatrix_cls_object.py
```

prepare attribute location data on CUB200

```
get_gt_partLocs.py
```

### training

Two types of models need to be trained, the standard CNN classifier and [Hardness predictor](http://openaccess.thecvf.com/content_ECCV_2018/html/Pei_Wang_Towards_Realistic_Predictors_ECCV_2018_paper.html). Three most popular architectures were tested. For reproducing each result individually, we separately wrote the code for each experiment. For the classifier,
```
train_cub_alexnet.py
train_cub_vgg.py
train_cub_res.py
train_ade_alexnet.py
train_ade_vgg.py
train_ade_res.py
```
for the hardness predictor,
```
train_hp_cub_alexnet.py
train_hp_cub_vgg.py
train_hp_cub_res.py
train_hp_ade_alexnet.py
train_hp_ade_vgg.py
train_hp_ade_res.py
```

### visualization

Three types of attribution methods are compared, baseline [gradient based](https://ieeexplore.ieee.org/document/8237336), state-of-the-art [integrated gradient based](https://dl.acm.org/citation.cfm?id=3306024) and ours.

In order to reproduce experiment results,

1. for comparison of different scores,

2. for comparison of different hidden layers,

3. for comparison of different attribution maps,

4. for comparison of different architectures,


### results presenting

plot the precision-recall curves on CUB200,

```
plot_recall_precision_curve_std.py
```

show average IoU precision on ADE20K,

```
output_IOU_threshold_std.py
```

### pretrained models

The pre-trained models for all experiments are availiable.



