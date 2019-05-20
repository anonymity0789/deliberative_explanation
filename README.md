# deliberative_explanation


## Requirements

1. The project was implemented and tested in Python 3.5 and Pytorch 0.4. The higher versions should work after minor modification.
2. Other common modules like numpy, pandas and seaborn for visualization.
3. NVIDIA GPU and cuDNN are required to have fast speeds. For now, CUDA 8.0 with cuDNN 6.0.20 has been tested. The other versions should be working.


## Datasets

[CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [ADE20K](http://sceneparsing.csail.mit.edu/) are used. Please organize them as below after download,

cub200

|_ CUB_200_2011
  |_ attributes
  |_ images
  |_ parts
  |_ train_test_split.txt
  |_ ...


VOCdevkit
|_ VOC2007
  |_ JPEGImages
  |  |_ <000001>.jpg
  |  |_ ...
  |  |_ <009963>.jpg
  |_ Annotations
  |  |_ <000001>.xml
  |  |_ ...
  |  |_ <009963>.xml
  |_ ...


ade

|_ ADEChallengeData2016
  |_ annotations
  |_ images
  |_ objectInfo 150.txt
  |_ sceneCategories.txt
 

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

Three types of attribution methods are compared, baseline [gradient based](https://ieeexplore.ieee.org/document/8237336), state-of-the-art [integrated gradient (IG) based](https://dl.acm.org/citation.cfm?id=3306024) and ours (gradient-Hessian(2ndG)).

In order to reproduce experiment results,

1. for comparison of different scores,
```
get_cs_cub_insecurity_vgg.py
get_entropy_cub_insecurity_vgg.py
get_hp_cub_insecurity_vgg.py
get_cs_ade_insecurity_vgg.py
get_entropy_ade_insecurity_vgg.py
get_hp_ade_insecurity_vgg.py
```

2. for comparison of different hidden layers,
```
get_hp_cub_insecurity_vgg.py
get_hp_ade_insecurity_vgg.py
```
Additionally, the layer number need to be changed for different hidden layers, 12, 22, 32, 42 w.r.t. conv2_2, conv3_3, conv4_3, conv5_3.


3. for comparison of different attribution maps,
```
get_hp_cub_insecurity_vgg.py
get_hp_cub_insecurity_vgg_IG.py
get_hp_cub_insecurity_vgg_2ndG.py
get_hp_ade_insecurity_vgg.py
get_hp_ade_insecurity_vgg_IG.py
get_hp_ade_insecurity_vgg_2ndG.py
```

4. for comparison of different architectures,

```
get_hp_cub_insecurity_alexnet.py
get_hp_cub_insecurity_vgg.py
get_hp_cub_insecurity_res.py
```

### results presenting

1. insecurity extraction,
```
get_hp_cub_insecurity_vgg_show.py
get_hp_ade_insecurity_vgg_sho.py
```

2. plot the precision-recall curves on CUB200,

```
plot_recall_precision_curve_std.py
```

3. show average IoU precision on ADE20K,

```
output_IOU_threshold_std.py
```

### pretrained models

The pre-trained models for all experiments are availiable.

## References

[1] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra.  Grad-cam:  Visual explanations from deep networks via gradient-based localization.  In Proceedings of the IEEE International Conference on Computer Vision, pages 618–626, 2017.

[2] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 3319–3328. JMLR. org,4662017.

[3] Pei Wang and Nuno Vasconcelos. Towards realistic predictors. In The European Conference on Computer Vision, 2018.

[4] Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical report, Citeseer, 2009.

[5] Bolei  Zhou,  Hang  Zhao,  Xavier  Puig,  Sanja  Fidler,  Adela  Barriuso,  and  Antonio  Torralba.   Scene parsing through ade20k dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.


