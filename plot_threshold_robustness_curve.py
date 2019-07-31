import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy import interpolate
import seaborn as sns
sns.set()

IOU_points_entropy1 = np.load('./cub200/entropy_vgg16bn_layer42_robustComp_multi1.npy')
IOU_points_entropy2 = np.load('./cub200/entropy_vgg16bn_layer42_robustComp_multi2.npy')
IOU_points_entropy3 = np.load('./cub200/entropy_vgg16bn_layer42_robustComp_multi3.npy')

IOU_points_cs1 = np.load('./cub200/confidence_score_vgg16bn_layer42_robustComp_multi1.npy')
IOU_points_cs2 = np.load('./cub200/confidence_score_vgg16bn_layer42_robustComp_multi2.npy')
IOU_points_cs3 = np.load('./cub200/confidence_score_vgg16bn_layer42_robustComp_multi3.npy')

IOU_points_hp1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_robustComp_multi1.npy')
IOU_points_hp2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_robustComp_multi2.npy')
IOU_points_hp3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_robustComp_multi3.npy')

IOU_points_IG1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_IG_robustComp_multi1.npy')
IOU_points_IG2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_IG_robustComp_multi2.npy')
IOU_points_IG3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_IG_robustComp_multi3.npy')

IOU_points_2nd1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_2ndG_robustComp_multi1.npy')
IOU_points_2nd2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_2ndG_robustComp_multi2.npy')
IOU_points_2nd3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_2ndG_robustComp_multi3.npy')

threshold = np.arange(1, 91)

IOU_points_entropy1 = IOU_points_entropy1[0:90]
IOU_points_entropy2 = IOU_points_entropy2[0:90]
IOU_points_entropy3 = IOU_points_entropy3[0:90]
IOU_points_cs1 = IOU_points_cs1[0:90]
IOU_points_cs2 = IOU_points_cs2[0:90]
IOU_points_cs3 = IOU_points_cs3[0:90]
IOU_points_hp1 = IOU_points_hp1[0:90]
IOU_points_hp2 = IOU_points_hp2[0:90]
IOU_points_hp3 = IOU_points_hp3[0:90]
IOU_points_IG1 = IOU_points_IG1[0:90]
IOU_points_IG2 = IOU_points_IG2[0:90]
IOU_points_IG3 = IOU_points_IG3[0:90]
IOU_points_2nd1 = IOU_points_2nd1[0:90]
IOU_points_2nd2 = IOU_points_2nd2[0:90]
IOU_points_2nd3 = IOU_points_2nd3[0:90]


precision = []
precision.extend(IOU_points_entropy1.tolist())
precision.extend(IOU_points_cs1.tolist())
precision.extend(IOU_points_hp1.tolist())
precision.extend(IOU_points_IG1.tolist())
precision.extend(IOU_points_2nd1.tolist())
precision.extend(IOU_points_entropy2.tolist())
precision.extend(IOU_points_cs2.tolist())
precision.extend(IOU_points_hp2.tolist())
precision.extend(IOU_points_IG2.tolist())
precision.extend(IOU_points_2nd2.tolist())
precision.extend(IOU_points_entropy3.tolist())
precision.extend(IOU_points_cs3.tolist())
precision.extend(IOU_points_hp3.tolist())
precision.extend(IOU_points_IG3.tolist())
precision.extend(IOU_points_2nd3.tolist())

scores = ['hesitancy score (gradient)', 'entropy score (gradient)', 'hardness score (gradient)', 'hardness score (integrated gradient)', 'hardness score (gradient-Hessian)']

condition = []
for k in range(3):
    for i in range(5):
        for j in range(np.size(threshold)):
            condition.append(scores[i])

threshold_list = []
for k in range(3):
    threshold_list.extend(threshold.tolist())
    threshold_list.extend(threshold.tolist())
    threshold_list.extend(threshold.tolist())
    threshold_list.extend(threshold.tolist())
    threshold_list.extend(threshold.tolist())

fig = plt.figure()

all_data = OrderedDict([('threshold', threshold_list),
                        ('precision', precision),
                        ('score (attribution function)', condition)])
df = pd.DataFrame.from_dict(all_data)

sns.lineplot(x="threshold", y="precision", hue="score (attribution function)", data=df)

plt.xlabel('threshold (%)', fontsize=10)
plt.ylabel('IoU', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.xlim(0, 100)
#
# # plt.legend()
plt.legend(prop={'size': 10})

# plt.show()
plt.savefig('IoU_threshold_ScoreContributionmapComp_std.pdf')
