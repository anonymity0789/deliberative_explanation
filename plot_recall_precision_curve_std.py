import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy import interpolate
import seaborn as sns
sns.set()

Comp_hardness_score = 0
Comp_contribution_map = 1
Comp_architecture = 0
Comp_hiddenLayer = 0

if Comp_hardness_score == 1:


    recall_hp_vgg1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_recall_multi1.npy')
    precision_hp_vgg1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_multi1.npy')
    recall_cs_vgg1 = np.load('./cub200/confidence_score_vgg16_layer42_recall_multi1.npy')
    precision_cs_vgg1 = np.load('./cub200/confidence_score_vgg16_layer42_precision_multi1.npy')
    recall_entropy_vgg1 = np.load('./cub200/entropy_vgg16_layer42_recall_multi1.npy')
    precision_entropy_vgg1 = np.load('./cub200/entropy_vgg16_layer42_precision_multi1.npy')

    recall_hp_vgg2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_recall_multi2.npy')
    precision_hp_vgg2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_multi2.npy')
    recall_cs_vgg2 = np.load('./cub200/confidence_score_vgg16_layer42_recall_multi2.npy')
    precision_cs_vgg2 = np.load('./cub200/confidence_score_vgg16_layer42_precision_multi2.npy')
    recall_entropy_vgg2 = np.load('./cub200/entropy_vgg16_layer42_recall_multi2.npy')
    precision_entropy_vgg2 = np.load('./cub200/entropy_vgg16_layer42_precision_multi2.npy')

    recall_hp_vgg3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_recall_multi3.npy')
    precision_hp_vgg3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_multi3.npy')
    recall_cs_vgg3 = np.load('./cub200/confidence_score_vgg16_layer42_recall_multi3.npy')
    precision_cs_vgg3 = np.load('./cub200/confidence_score_vgg16_layer42_precision_multi3.npy')
    recall_entropy_vgg3 = np.load('./cub200/entropy_vgg16_layer42_recall_multi3.npy')
    precision_entropy_vgg3 = np.load('./cub200/entropy_vgg16_layer42_precision_multi3.npy')


    recall = np.arange(0.1, 0.45, 0.01)
    f_linear_hp_vgg1 = interpolate.interp1d(recall_hp_vgg1, precision_hp_vgg1)
    precision_hp_vgg1 = f_linear_hp_vgg1(recall)
    f_linear_cs_vgg1 = interpolate.interp1d(recall_cs_vgg1, precision_cs_vgg1)
    precision_cs_vgg1 = f_linear_cs_vgg1(recall)
    f_linear_entropy_vgg1 = interpolate.interp1d(recall_entropy_vgg1, precision_entropy_vgg1)
    precision_entropy_vgg1 = f_linear_entropy_vgg1(recall)

    f_linear_hp_vgg2 = interpolate.interp1d(recall_hp_vgg2, precision_hp_vgg2)
    precision_hp_vgg2 = f_linear_hp_vgg2(recall)
    f_linear_cs_vgg2 = interpolate.interp1d(recall_cs_vgg2, precision_cs_vgg2)
    precision_cs_vgg2 = f_linear_cs_vgg2(recall)
    f_linear_entropy_vgg2 = interpolate.interp1d(recall_entropy_vgg2, precision_entropy_vgg2)
    precision_entropy_vgg2 = f_linear_entropy_vgg2(recall)

    f_linear_hp_vgg3 = interpolate.interp1d(recall_hp_vgg3, precision_hp_vgg3)
    precision_hp_vgg3 = f_linear_hp_vgg3(recall)
    f_linear_cs_vgg3 = interpolate.interp1d(recall_cs_vgg3, precision_cs_vgg3)
    precision_cs_vgg3 = f_linear_cs_vgg3(recall)
    f_linear_entropy_vgg3 = interpolate.interp1d(recall_entropy_vgg3, precision_entropy_vgg3)
    precision_entropy_vgg3 = f_linear_entropy_vgg3(recall)

    precision = []
    precision.extend(precision_cs_vgg1.tolist())
    precision.extend(precision_entropy_vgg1.tolist())
    precision.extend(precision_hp_vgg1.tolist())
    precision.extend(precision_cs_vgg2.tolist())
    precision.extend(precision_entropy_vgg2.tolist())
    precision.extend(precision_hp_vgg2.tolist())
    precision.extend(precision_cs_vgg3.tolist())
    precision.extend(precision_entropy_vgg3.tolist())
    precision.extend(precision_hp_vgg3.tolist())

    scores = ['hesitancy score', 'entropy score', 'hardness score']
    condition = []
    for k in range(3):
        for i in range(3):
            for j in range(np.size(recall)):
                condition.append(scores[i])

    recall_list = []
    for k in range(3):
        recall_list.extend(recall.tolist())
        recall_list.extend(recall.tolist())
        recall_list.extend(recall.tolist())

    fig = plt.figure()

    all_data = OrderedDict([('recall', recall_list),
                            ('precision', precision),
                            ('scores', condition)])
    df = pd.DataFrame.from_dict(all_data)

    # sns.tsplot(data=df, time="recall", condition="scores", value="precision")
    sns.lineplot(x="recall", y="precision", hue="scores", data=df)

    plt.xlabel('recall', fontsize=15)
    plt.ylabel('precision', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #
    # # plt.legend()
    plt.legend(prop={'size': 15})

    # plt.show()
    plt.savefig('precision_recall_curve_hardnessComp_std.pdf')

if Comp_contribution_map == 1:

    recall_gradient1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_recall_multi1.npy')
    precision_gradient1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_multi1.npy')
    recall_IG1 = np.load('./cub200/hardness_predictor_vgg16_layer42_IG_recall2_multi1.npy')
    precision_IG1 = np.load('./cub200/hardness_predictor_vgg16_layer42_IG_precision2_multi1.npy')
    recall_2ndgradient1 = np.load('./cub200/hardness_predictor_vgg16_layer42_2ndG_recall_multi1.npy')
    precision_2ndgradient1 = np.load('./cub200/hardness_predictor_vgg16_layer42_2ndG_precision_multi1.npy')
    recall_cls1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_recall_cls_multi1.npy')
    precision_cls1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_cls_multi1.npy')

    recall_gradient2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_recall_multi2.npy')
    precision_gradient2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_multi2.npy')
    recall_IG2 = np.load('./cub200/hardness_predictor_vgg16_layer42_IG_recall2_multi2.npy')
    precision_IG2 = np.load('./cub200/hardness_predictor_vgg16_layer42_IG_precision2_multi2.npy')
    recall_2ndgradient2 = np.load('./cub200/hardness_predictor_vgg16_layer42_2ndG_recall_multi2.npy')
    precision_2ndgradient2 = np.load('./cub200/hardness_predictor_vgg16_layer42_2ndG_precision_multi2.npy')
    recall_cls2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_recall_cls_multi2.npy')
    precision_cls2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_cls_multi2.npy')

    recall_gradient3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_recall_multi3.npy')
    precision_gradient3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_multi3.npy')
    recall_IG3 = np.load('./cub200/hardness_predictor_vgg16_layer42_IG_recall2_multi3.npy')
    precision_IG3 = np.load('./cub200/hardness_predictor_vgg16_layer42_IG_precision2_multi3.npy')
    recall_2ndgradient3 = np.load('./cub200/hardness_predictor_vgg16_layer42_2ndG_recall_multi3.npy')
    precision_2ndgradient3 = np.load('./cub200/hardness_predictor_vgg16_layer42_2ndG_precision_multi3.npy')
    recall_cls3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_recall_cls_multi3.npy')
    precision_cls3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_cls_multi3.npy')

    recall = np.arange(0.1, 0.45, 0.01)
    f_linear_gradient1 = interpolate.interp1d(recall_gradient1, precision_gradient1)
    precision_gradient1 = f_linear_gradient1(recall)
    f_linear_IG1 = interpolate.interp1d(recall_IG1, precision_IG1)
    precision_IG1 = f_linear_IG1(recall)
    f_linear_2ndgradient1 = interpolate.interp1d(recall_2ndgradient1, precision_2ndgradient1)
    precision_2ndgradient1 = f_linear_2ndgradient1(recall)
    f_linear_cls1 = interpolate.interp1d(recall_cls1, precision_cls1)
    precision_cls1 = f_linear_cls1(recall)

    f_linear_gradient2 = interpolate.interp1d(recall_gradient2, precision_gradient2)
    precision_gradient2 = f_linear_gradient2(recall)
    f_linear_IG2 = interpolate.interp1d(recall_IG2, precision_IG2)
    precision_IG2 = f_linear_IG2(recall)
    f_linear_2ndgradient2 = interpolate.interp1d(recall_2ndgradient2, precision_2ndgradient2)
    precision_2ndgradient2 = f_linear_2ndgradient2(recall)
    f_linear_cls2 = interpolate.interp1d(recall_cls2, precision_cls2)
    precision_cls2 = f_linear_cls2(recall)

    f_linear_gradient3 = interpolate.interp1d(recall_gradient3, precision_gradient3)
    precision_gradient3 = f_linear_gradient3(recall)
    f_linear_IG3 = interpolate.interp1d(recall_IG3, precision_IG3)
    precision_IG3 = f_linear_IG3(recall)
    f_linear_2ndgradient3 = interpolate.interp1d(recall_2ndgradient3, precision_2ndgradient3)
    precision_2ndgradient3 = f_linear_2ndgradient3(recall)
    f_linear_cls3 = interpolate.interp1d(recall_cls3, precision_cls3)
    precision_cls3 = f_linear_cls3(recall)

    precision = []
    precision.extend(precision_gradient1.tolist())
    precision.extend(precision_cls1.tolist())
    precision.extend(precision_IG1.tolist())
    precision.extend(precision_2ndgradient1.tolist())

    precision.extend(precision_gradient2.tolist())
    precision.extend(precision_cls2.tolist())
    precision.extend(precision_IG2.tolist())
    precision.extend(precision_2ndgradient2.tolist())

    precision.extend(precision_gradient3.tolist())
    precision.extend(precision_cls3.tolist())
    precision.extend(precision_IG3.tolist())
    precision.extend(precision_2ndgradient3.tolist())

    scores = ['gradient', 'gradient w/o $m^s$', 'integrated gradient', 'gradient-Hessian']
    condition = []
    for k in range(3):
        for i in range(4):
            for j in range(np.size(recall)):
                condition.append(scores[i])

    recall_list = []
    for k in range(3):
        recall_list.extend(recall.tolist())
        recall_list.extend(recall.tolist())
        recall_list.extend(recall.tolist())
        recall_list.extend(recall.tolist())

    fig = plt.figure()

    all_data = OrderedDict([('recall', recall_list),
                            ('precision', precision),
                            ('attribution map', condition)])
    df = pd.DataFrame.from_dict(all_data)

    # sns.tsplot(data=df, time="recall", condition="scores", value="precision")
    sns.lineplot(x="recall", y="precision", hue="attribution map", data=df)

    plt.xlabel('recall', fontsize=15)
    plt.ylabel('precision', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #
    # # plt.legend()
    plt.legend(prop={'size': 15})

    # plt.show()
    plt.savefig('precision_recall_curve_contributionmapComp_std_new.pdf')


if Comp_architecture == 1:

    recall_vgg1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_recall_multi1.npy')
    precision_vgg1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_multi1.npy')
    recall_resnet1 = np.load('./cub200/hardness_predictor_res50_lastCovlayer_ComAttr02_recall_multi1.npy')
    precision_resnet1 = np.load('./cub200/hardness_predictor_res50_lastCovlayer_ComAttr02_precision_multi1.npy')
    recall_alexnet1 = np.load('./cub200/hardness_predictor_alexnet_lastConv_recall_multi1.npy')
    precision_alexnet1 = np.load('./cub200/hardness_predictor_alexnet_lastConv_precision_multi1.npy')

    recall_vgg2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_recall_multi2.npy')
    precision_vgg2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_multi2.npy')
    recall_resnet2 = np.load('./cub200/hardness_predictor_res50_lastCovlayer_ComAttr02_recall_multi2.npy')
    precision_resnet2 = np.load('./cub200/hardness_predictor_res50_lastCovlayer_ComAttr02_precision_multi2.npy')
    recall_alexnet2 = np.load('./cub200/hardness_predictor_alexnet_lastConv_recall_multi2.npy')
    precision_alexnet2 = np.load('./cub200/hardness_predictor_alexnet_lastConv_precision_multi2.npy')

    recall_vgg3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_recall_multi3.npy')
    precision_vgg3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_multi3.npy')
    recall_resnet3 = np.load('./cub200/hardness_predictor_res50_lastCovlayer_ComAttr02_recall_multi3.npy')
    precision_resnet3 = np.load('./cub200/hardness_predictor_res50_lastCovlayer_ComAttr02_precision_multi3.npy')
    recall_alexnet3 = np.load('./cub200/hardness_predictor_alexnet_lastConv_recall_multi3.npy')
    precision_alexnet3 = np.load('./cub200/hardness_predictor_alexnet_lastConv_precision_multi3.npy')

    recall = np.arange(0.1, 0.45, 0.01)
    f_linear_vgg1 = interpolate.interp1d(recall_vgg1, precision_vgg1)
    precision_vgg1 = f_linear_vgg1(recall)
    f_linear_resnet1 = interpolate.interp1d(recall_resnet1, precision_resnet1)
    precision_resnet1 = f_linear_resnet1(recall)
    f_linear_alexnet1 = interpolate.interp1d(recall_alexnet1, precision_alexnet1)
    precision_alexnet1 = f_linear_alexnet1(recall)

    f_linear_vgg2 = interpolate.interp1d(recall_vgg2, precision_vgg2)
    precision_vgg2 = f_linear_vgg2(recall)
    f_linear_resnet2 = interpolate.interp1d(recall_resnet2, precision_resnet2)
    precision_resnet2 = f_linear_resnet2(recall)
    f_linear_alexnet2 = interpolate.interp1d(recall_alexnet2, precision_alexnet2)
    precision_alexnet2 = f_linear_alexnet2(recall)

    f_linear_vgg3 = interpolate.interp1d(recall_vgg3, precision_vgg3)
    precision_vgg3 = f_linear_vgg3(recall)
    f_linear_resnet3 = interpolate.interp1d(recall_resnet3, precision_resnet3)
    precision_resnet3 = f_linear_resnet3(recall)
    f_linear_alexnet3 = interpolate.interp1d(recall_alexnet3, precision_alexnet3)
    precision_alexnet3 = f_linear_alexnet3(recall)

    precision = []
    precision.extend(precision_alexnet1.tolist())
    precision.extend(precision_vgg1.tolist())
    precision.extend(precision_resnet1.tolist())
    precision.extend(precision_alexnet2.tolist())
    precision.extend(precision_vgg2.tolist())
    precision.extend(precision_resnet2.tolist())
    precision.extend(precision_alexnet3.tolist())
    precision.extend(precision_vgg3.tolist())
    precision.extend(precision_resnet3.tolist())

    scores = ['AlexNet', 'VGG16', 'ResNet50']
    condition = []
    for k in range(3):
        for i in range(3):
            for j in range(np.size(recall)):
                condition.append(scores[i])

    recall_list = []
    for k in range(3):
        recall_list.extend(recall.tolist())
        recall_list.extend(recall.tolist())
        recall_list.extend(recall.tolist())

    fig = plt.figure()

    all_data = OrderedDict([('recall', recall_list),
                            ('precision', precision),
                            ('architecture', condition)])
    df = pd.DataFrame.from_dict(all_data)

    # sns.tsplot(data=df, time="recall", condition="scores", value="precision")
    sns.lineplot(x="recall", y="precision", hue="architecture", data=df)

    plt.xlabel('recall', fontsize=15)
    plt.ylabel('precision', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #
    # # plt.legend()
    plt.legend(prop={'size': 15})

    # plt.show()
    plt.savefig('precision_recall_curve_architectureComp_std.pdf')

if Comp_hiddenLayer == 1:

    recall_vgg_12_1 = np.load('./cub200/hardness_predictor_vgg16bn_layer12_recall_multi1.npy')
    precision_vgg_12_1 = np.load('./cub200/hardness_predictor_vgg16bn_layer12_precision_multi1.npy')
    recall_vgg_22_1 = np.load('./cub200/hardness_predictor_vgg16bn_layer22_recall_multi1.npy')
    precision_vgg_22_1 = np.load('./cub200/hardness_predictor_vgg16bn_layer22_precision_multi1.npy')
    recall_vgg_32_1 = np.load('./cub200/hardness_predictor_vgg16bn_layer32_recall_multi1.npy')
    precision_vgg_32_1 = np.load('./cub200/hardness_predictor_vgg16bn_layer32_precision_multi1.npy')
    recall_vgg_42_1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_recall_multi1.npy')
    precision_vgg_42_1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_multi1.npy')

    recall_vgg_12_2 = np.load('./cub200/hardness_predictor_vgg16bn_layer12_recall_multi2.npy')
    precision_vgg_12_2 = np.load('./cub200/hardness_predictor_vgg16bn_layer12_precision_multi2.npy')
    recall_vgg_22_2 = np.load('./cub200/hardness_predictor_vgg16bn_layer22_recall_multi2.npy')
    precision_vgg_22_2 = np.load('./cub200/hardness_predictor_vgg16bn_layer22_precision_multi2.npy')
    recall_vgg_32_2 = np.load('./cub200/hardness_predictor_vgg16bn_layer32_recall_multi2.npy')
    precision_vgg_32_2 = np.load('./cub200/hardness_predictor_vgg16bn_layer32_precision_multi2.npy')
    recall_vgg_42_2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_recall_multi2.npy')
    precision_vgg_42_2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_multi2.npy')

    recall_vgg_12_3 = np.load('./cub200/hardness_predictor_vgg16bn_layer12_recall_multi3.npy')
    precision_vgg_12_3 = np.load('./cub200/hardness_predictor_vgg16bn_layer12_precision_multi3.npy')
    recall_vgg_22_3 = np.load('./cub200/hardness_predictor_vgg16bn_layer22_recall_multi3.npy')
    precision_vgg_22_3 = np.load('./cub200/hardness_predictor_vgg16bn_layer22_precision_multi3.npy')
    recall_vgg_32_3 = np.load('./cub200/hardness_predictor_vgg16bn_layer32_recall_multi3.npy')
    precision_vgg_32_3 = np.load('./cub200/hardness_predictor_vgg16bn_layer32_precision_multi3.npy')
    recall_vgg_42_3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_recall_multi3.npy')
    precision_vgg_42_3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_multi3.npy')

    recall = np.arange(0.11, 0.45, 0.01)
    f_linear_vgg_12_1 = interpolate.interp1d(recall_vgg_12_1, precision_vgg_12_1)
    precision_vgg_12_1 = f_linear_vgg_12_1(recall)
    f_linear_vgg_22_1 = interpolate.interp1d(recall_vgg_22_1, precision_vgg_22_1)
    precision_vgg_22_1 = f_linear_vgg_22_1(recall)
    f_linear_vgg_32_1 = interpolate.interp1d(recall_vgg_32_1, precision_vgg_32_1)
    precision_vgg_32_1 = f_linear_vgg_32_1(recall)
    f_linear_vgg_42_1 = interpolate.interp1d(recall_vgg_42_1, precision_vgg_42_1)
    precision_vgg_42_1 = f_linear_vgg_42_1(recall)

    f_linear_vgg_12_2 = interpolate.interp1d(recall_vgg_12_2, precision_vgg_12_2)
    precision_vgg_12_2 = f_linear_vgg_12_2(recall)
    f_linear_vgg_22_2 = interpolate.interp1d(recall_vgg_22_2, precision_vgg_22_2)
    precision_vgg_22_2 = f_linear_vgg_22_2(recall)
    f_linear_vgg_32_2 = interpolate.interp1d(recall_vgg_32_2, precision_vgg_32_2)
    precision_vgg_32_2 = f_linear_vgg_32_2(recall)
    f_linear_vgg_42_2 = interpolate.interp1d(recall_vgg_42_2, precision_vgg_42_2)
    precision_vgg_42_2 = f_linear_vgg_42_2(recall)

    f_linear_vgg_12_3 = interpolate.interp1d(recall_vgg_12_3, precision_vgg_12_3)
    precision_vgg_12_3 = f_linear_vgg_12_3(recall)
    f_linear_vgg_22_3 = interpolate.interp1d(recall_vgg_22_3, precision_vgg_22_3)
    precision_vgg_22_3 = f_linear_vgg_22_3(recall)
    f_linear_vgg_32_3 = interpolate.interp1d(recall_vgg_32_3, precision_vgg_32_3)
    precision_vgg_32_3 = f_linear_vgg_32_3(recall)
    f_linear_vgg_42_3 = interpolate.interp1d(recall_vgg_42_3, precision_vgg_42_3)
    precision_vgg_42_3 = f_linear_vgg_42_3(recall)


    precision = []
    precision.extend(precision_vgg_12_1.tolist())
    precision.extend(precision_vgg_22_1.tolist())
    precision.extend(precision_vgg_32_1.tolist())
    precision.extend(precision_vgg_42_1.tolist())
    precision.extend(precision_vgg_12_2.tolist())
    precision.extend(precision_vgg_22_2.tolist())
    precision.extend(precision_vgg_32_2.tolist())
    precision.extend(precision_vgg_42_2.tolist())
    precision.extend(precision_vgg_12_3.tolist())
    precision.extend(precision_vgg_22_3.tolist())
    precision.extend(precision_vgg_32_3.tolist())
    precision.extend(precision_vgg_42_3.tolist())

    scores = ['conv2_2', 'conv3_3', 'conv4_3','conv5_3']
    condition = []
    for k in range(3):
        for i in range(4):
            for j in range(np.size(recall)):
                condition.append(scores[i])

    recall_list = []
    for k in range(3):
        recall_list.extend(recall.tolist())
        recall_list.extend(recall.tolist())
        recall_list.extend(recall.tolist())
        recall_list.extend(recall.tolist())

    fig = plt.figure()

    all_data = OrderedDict([('recall', recall_list),
                            ('precision', precision),
                            ('hidden layer', condition)])
    df = pd.DataFrame.from_dict(all_data)

    # sns.tsplot(data=df, time="recall", condition="scores", value="precision")
    sns.lineplot(x="recall", y="precision", hue="hidden layer", data=df)

    plt.xlabel('recall', fontsize=15)
    plt.ylabel('precision', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #
    # # plt.legend()
    plt.legend(prop={'size': 15})

    # plt.show()
    plt.savefig('precision_recall_curve_hiddenLayerComp_std.pdf')