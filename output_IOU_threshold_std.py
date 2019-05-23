import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

Comp_hardness_score = 0
Comp_contribution_map = 1
Comp_architecture = 0
Comp_hiddenLayer = 0


if Comp_hardness_score == 1:

    IOU_hp_vgg1 = np.load('./ade/hardness_predictor_vgg16bn_layer42_IOU_multi1.npy')
    IOU_cs_vgg1 = np.load('./ade/confidence_score_vgg16_layer42_IOU_multi1.npy')
    IOU_entropy_vgg1 = np.load('./ade/entropy_vgg16_layer42_IOU_multi1.npy')

    IOU_hp_vgg2 = np.load('./ade/hardness_predictor_vgg16bn_layer42_IOU_multi2.npy')
    IOU_cs_vgg2 = np.load('./ade/confidence_score_vgg16_layer42_IOU_multi2.npy')
    IOU_entropy_vgg2 = np.load('./ade/entropy_vgg16_layer42_IOU_multi2.npy')

    IOU_hp_vgg3 = np.load('./ade/hardness_predictor_vgg16bn_layer42_IOU_multi3.npy')
    IOU_cs_vgg3 = np.load('./ade/confidence_score_vgg16_layer42_IOU_multi3.npy')
    IOU_entropy_vgg3 = np.load('./ade/entropy_vgg16_layer42_IOU_multi3.npy')

    IOU_hp_vgg = np.array([IOU_hp_vgg1,IOU_hp_vgg2,IOU_hp_vgg3])
    IOU_hp_vgg_mean = np.mean(IOU_hp_vgg, axis=0)
    IOU_hp_vgg_std = np.std(IOU_hp_vgg, axis=0)

    IOU_cs_vgg = np.array([IOU_cs_vgg1,IOU_cs_vgg2,IOU_cs_vgg3])
    IOU_cs_vgg_mean = np.mean(IOU_cs_vgg, axis=0)
    IOU_cs_vgg_std = np.std(IOU_cs_vgg, axis=0)

    IOU_entropy_vgg = np.array([IOU_entropy_vgg1,IOU_entropy_vgg2,IOU_entropy_vgg3])
    IOU_entropy_vgg_mean = np.mean(IOU_entropy_vgg, axis=0)
    IOU_entropy_vgg_std = np.std(IOU_entropy_vgg, axis=0)

    print('-----mean---------')
    print('hardness score')
    print(100*IOU_hp_vgg_mean[9:90:20], 100*np.mean(IOU_hp_vgg_mean[9:90:20]))
    print('confidence score')
    print(100*IOU_cs_vgg_mean[9:90:20], 100*np.mean(IOU_cs_vgg_mean[9:90:20]))
    print('entropy score')
    print(100*IOU_entropy_vgg_mean[9:90:20], 100*np.mean(IOU_entropy_vgg_mean[9:90:20]))
    print('-------std----------')
    print('hardness score')
    print(100*IOU_hp_vgg_std[9:90:20], 100*np.std(np.array([np.mean(IOU_hp_vgg1[9:90:20]), np.mean(IOU_hp_vgg2[9:90:20]), np.mean(IOU_hp_vgg3[9:90:20])])))
    print('confidence score')
    print(100*IOU_cs_vgg_std[9:90:20], 100*np.std(np.array([np.mean(IOU_cs_vgg1[9:90:20]), np.mean(IOU_cs_vgg2[9:90:20]), np.mean(IOU_cs_vgg3[9:90:20])])))
    print('entropy score')
    print(100*IOU_entropy_vgg_std[9:90:20], 100*np.std(np.array([np.mean(IOU_entropy_vgg1[9:90:20]), np.mean(IOU_entropy_vgg2[9:90:20]), np.mean(IOU_entropy_vgg3[9:90:20])])))



if Comp_contribution_map == 1:

    IOU_gradient1 = np.load('./ade/hardness_predictor_vgg16bn_layer42_IOU_multi1.npy')
    IOU_IG1 = np.load('./ade/hardness_predictor_vgg16_layer42_IG_IOU_multi1.npy')
    IOU_2ndgradient1 = np.load('./ade/hardness_predictor_vgg16_layer42_2ndG_IOU_multi1.npy')
    IOU_cls1 = np.load('./ade/hardness_predictor_vgg16bn_layer12_IOU_cls_multi1.npy')

    IOU_gradient2 = np.load('./ade/hardness_predictor_vgg16bn_layer42_IOU_multi2.npy')
    IOU_IG2 = np.load('./ade/hardness_predictor_vgg16_layer42_IG_IOU_multi2.npy')
    IOU_2ndgradient2 = np.load('./ade/hardness_predictor_vgg16_layer42_2ndG_IOU_multi2.npy')
    IOU_cls2 = np.load('./ade/hardness_predictor_vgg16bn_layer12_IOU_cls_multi2.npy')

    IOU_gradient3 = np.load('./ade/hardness_predictor_vgg16bn_layer42_IOU_multi3.npy')
    IOU_IG3 = np.load('./ade/hardness_predictor_vgg16_layer42_IG_IOU_multi3.npy')
    IOU_2ndgradient3 = np.load('./ade/hardness_predictor_vgg16_layer42_2ndG_IOU_multi3.npy')
    IOU_cls3 = np.load('./ade/hardness_predictor_vgg16bn_layer12_IOU_cls_multi3.npy')

    IOU_gradient = np.array([IOU_gradient1, IOU_gradient2, IOU_gradient3])
    IOU_gradient_mean = np.mean(IOU_gradient, axis=0)
    IOU_gradient_std = np.std(IOU_gradient, axis=0)

    IOU_IG = np.array([IOU_IG1, IOU_IG2, IOU_IG3])
    IOU_IG_mean = np.mean(IOU_IG, axis=0)
    IOU_IG_std = np.std(IOU_IG, axis=0)

    IOU_2ndgradient = np.array([IOU_2ndgradient1, IOU_2ndgradient2, IOU_2ndgradient3])
    IOU_2ndgradient_mean = np.mean(IOU_2ndgradient, axis=0)
    IOU_2ndgradient_std = np.std(IOU_2ndgradient, axis=0)

    IOU_cls = np.array([IOU_cls1, IOU_cls1, IOU_cls1])
    IOU_cls_mean = np.mean(IOU_cls, axis=0)
    IOU_cls_std = np.std(IOU_cls, axis=0)

    print('-----mean---------')
    print('gradient')
    print(100 * IOU_gradient_mean[9:90:20], 100 * np.mean(IOU_gradient_mean[9:90:20]))
    print('IG')
    print(100 * IOU_IG_mean[9:90:20], 100 * np.mean(IOU_IG_mean[9:90:20]))
    print('2ndgradient')
    print(100 * IOU_2ndgradient_mean[9:90:20], 100 * np.mean(IOU_2ndgradient_mean[9:90:20]))
    print('gradient cls')
    print(100 * IOU_cls_mean[9:90:20], 100 * np.mean(IOU_cls_mean[9:90:20]))
    print('-------std----------')
    print('gradient')
    print(100 * IOU_gradient_std[9:90:20], 100 * np.std(np.array([np.mean(IOU_gradient1[9:90:20]), np.mean(IOU_gradient2[9:90:20]), np.mean(IOU_gradient3[9:90:20])])))
    print('IG')
    print(100 * IOU_IG_std[9:90:20],100 * np.std(np.array([np.mean(IOU_IG1[9:90:20]), np.mean(IOU_IG2[9:90:20]), np.mean(IOU_IG3[9:90:20])])))
    print('2ndgradient')
    print(100 * IOU_2ndgradient_std[9:90:20], 100 * np.std(np.array([np.mean(IOU_2ndgradient1[9:90:20]), np.mean(IOU_2ndgradient2[9:90:20]), np.mean(IOU_2ndgradient3[9:90:20])])))
    print('gradient cls')
    print(100 * IOU_cls_std[9:90:20], 100 * np.std(np.array([np.mean(IOU_cls1[9:90:20]), np.mean(IOU_cls2[9:90:20]), np.mean(IOU_cls3[9:90:20])])))

if Comp_architecture == 1:

    IOU_alexnet1 = np.load('./ade/hardness_predictor_alexnet_lastConv_IOU_multi1.npy')
    IOU_vgg1 = np.load('./ade/hardness_predictor_vgg16bn_layer42_IOU_multi1.npy')
    IOU_resnet1 = np.load('./ade/hardness_predictor_res50_lastCovlayer_IOU_multi1.npy')

    IOU_alexnet2 = np.load('./ade/hardness_predictor_alexnet_lastConv_IOU_multi2.npy')
    IOU_vgg2 = np.load('./ade/hardness_predictor_vgg16bn_layer42_IOU_multi2.npy')
    IOU_resnet2 = np.load('./ade/hardness_predictor_res50_lastCovlayer_IOU_multi2.npy')

    IOU_alexnet3 = np.load('./ade/hardness_predictor_alexnet_lastConv_IOU_multi3.npy')
    IOU_vgg3 = np.load('./ade/hardness_predictor_vgg16bn_layer42_IOU_multi3.npy')
    IOU_resnet3 = np.load('./ade/hardness_predictor_res50_lastCovlayer_IOU_multi3.npy')

    IOU_alexnet = np.array([IOU_alexnet1,IOU_alexnet2,IOU_alexnet3])
    IOU_alexnet_mean = np.mean(IOU_alexnet, axis=0)
    IOU_alexnet_std = np.std(IOU_alexnet, axis=0)

    IOU_vgg = np.array([IOU_vgg1,IOU_vgg2,IOU_vgg3])
    IOU_vgg_mean = np.mean(IOU_vgg, axis=0)
    IOU_vgg_std = np.std(IOU_vgg, axis=0)

    IOU_resnet = np.array([IOU_resnet1,IOU_resnet2,IOU_resnet3])
    IOU_resnet_mean = np.mean(IOU_resnet, axis=0)
    IOU_resnet_std = np.std(IOU_resnet, axis=0)

    print('-----mean---------')
    print('alexnet')
    print(100*IOU_alexnet_mean[9:90:20], 100*np.mean(IOU_alexnet_mean[9:90:20]))
    print('vgg')
    print(100*IOU_vgg_mean[9:90:20], 100*np.mean(IOU_vgg_mean[9:90:20]))
    print('resnet')
    print(100*IOU_resnet_mean[9:90:20], 100*np.mean(IOU_resnet_mean[9:90:20]))
    print('-------std----------')
    print('alexnet')
    print(100*IOU_alexnet_std[9:90:20], 100*np.std(np.array([np.mean(IOU_alexnet1[9:90:20]), np.mean(IOU_alexnet2[9:90:20]), np.mean(IOU_alexnet3[9:90:20])])))
    print('vgg')
    print(100*IOU_vgg_std[9:90:20], 100*np.std(np.array([np.mean(IOU_vgg1[9:90:20]), np.mean(IOU_vgg2[9:90:20]), np.mean(IOU_vgg3[9:90:20])])))
    print('resnet')
    print(100*IOU_resnet_std[9:90:20], 100*np.std(np.array([np.mean(IOU_resnet1[9:90:20]), np.mean(IOU_resnet2[9:90:20]), np.mean(IOU_resnet3[9:90:20])])))



if Comp_hiddenLayer == 1:

    IOU_vgg_12_1 = np.load('./ade/hardness_predictor_vgg16bn_layer12_IOU_multi1.npy')
    IOU_vgg_22_1 = np.load('./ade/hardness_predictor_vgg16bn_layer22_IOU_multi1.npy')
    IOU_vgg_32_1 = np.load('./ade/hardness_predictor_vgg16bn_layer32_IOU_multi1.npy')
    IOU_vgg_42_1 = np.load('./ade/hardness_predictor_vgg16bn_layer42_IOU_multi1.npy')

    IOU_vgg_12_2 = np.load('./ade/hardness_predictor_vgg16bn_layer12_IOU_multi2.npy')
    IOU_vgg_22_2 = np.load('./ade/hardness_predictor_vgg16bn_layer22_IOU_multi2.npy')
    IOU_vgg_32_2 = np.load('./ade/hardness_predictor_vgg16bn_layer32_IOU_multi2.npy')
    IOU_vgg_42_2 = np.load('./ade/hardness_predictor_vgg16bn_layer42_IOU_multi2.npy')

    IOU_vgg_12_3 = np.load('./ade/hardness_predictor_vgg16bn_layer12_IOU_multi3.npy')
    IOU_vgg_22_3 = np.load('./ade/hardness_predictor_vgg16bn_layer22_IOU_multi3.npy')
    IOU_vgg_32_3 = np.load('./ade/hardness_predictor_vgg16bn_layer32_IOU_multi3.npy')
    IOU_vgg_42_3 = np.load('./ade/hardness_predictor_vgg16bn_layer42_IOU_multi3.npy')

    IOU_hp_12 = np.array([IOU_vgg_12_1,IOU_vgg_12_2,IOU_vgg_12_3])
    IOU_hp_12_mean = np.mean(IOU_hp_12, axis=0)
    IOU_hp_12_std = np.std(IOU_hp_12, axis=0)

    IOU_hp_22 = np.array([IOU_vgg_22_1,IOU_vgg_22_2,IOU_vgg_22_3])
    IOU_hp_22_mean = np.mean(IOU_hp_22, axis=0)
    IOU_hp_22_std = np.std(IOU_hp_22, axis=0)

    IOU_hp_32 = np.array([IOU_vgg_32_1,IOU_vgg_32_2,IOU_vgg_32_3])
    IOU_hp_32_mean = np.mean(IOU_hp_32, axis=0)
    IOU_hp_32_std = np.std(IOU_hp_32, axis=0)

    IOU_hp_42 = np.array([IOU_vgg_42_1,IOU_vgg_42_2,IOU_vgg_42_3])
    IOU_hp_42_mean = np.mean(IOU_hp_42, axis=0)
    IOU_hp_42_std = np.std(IOU_hp_42, axis=0)

    print('-----mean---------')
    print('con2_2')
    print(100*IOU_hp_12_mean[9:90:20], 100*np.mean(IOU_hp_12_mean[9:90:20]))
    print('con3_3')
    print(100*IOU_hp_22_mean[9:90:20], 100*np.mean(IOU_hp_22_mean[9:90:20]))
    print('con4_3')
    print(100*IOU_hp_32_mean[9:90:20], 100*np.mean(IOU_hp_32_mean[9:90:20]))
    print('con5_3')
    print(100*IOU_hp_42_mean[9:90:20], 100*np.mean(IOU_hp_42_mean[9:90:20]))


    print('-------std----------')
    print('con2_2')
    print(100*IOU_hp_12_std[9:90:20], 100*np.std(np.array([np.mean(IOU_vgg_12_1[9:90:20]), np.mean(IOU_vgg_12_2[9:90:20]), np.mean(IOU_vgg_12_3[9:90:20])])))
    print('con3_3')
    print(100*IOU_hp_22_std[9:90:20], 100*np.std(np.array([np.mean(IOU_vgg_22_1[9:90:20]), np.mean(IOU_vgg_22_2[9:90:20]), np.mean(IOU_vgg_22_3[9:90:20])])))
    print('con4_3')
    print(100*IOU_hp_32_std[9:90:20], 100*np.std(np.array([np.mean(IOU_vgg_32_1[9:90:20]), np.mean(IOU_vgg_32_2[9:90:20]), np.mean(IOU_vgg_32_3[9:90:20])])))
    print('con5_3')
    print(100*IOU_hp_42_std[9:90:20], 100*np.std(np.array([np.mean(IOU_vgg_42_1[9:90:20]), np.mean(IOU_vgg_42_2[9:90:20]), np.mean(IOU_vgg_42_3[9:90:20])])))
