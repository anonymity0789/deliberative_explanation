import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats.stats import pearsonr

def pearson_computation(score1, pre1, score2, pre2, score3, pre3):
    info = zip(score1, pre1)
    info = sorted(info, key=lambda i: i[0])  # from small to large
    score1, pre1 = [list(l) for l in zip(*info)]
    score1 = np.array(score1)
    pre1 = np.array(pre1)
    r1, p1 = pearsonr(score1, pre1)

    info = zip(score2, pre2)
    info = sorted(info, key=lambda i: i[0])  # from small to large
    score2, pre2 = [list(l) for l in zip(*info)]
    score2 = np.array(score2)
    pre2 = np.array(pre2)
    r2, p2 = pearsonr(score2, pre2)

    info = zip(score3, pre3)
    info = sorted(info, key=lambda i: i[0])  # from small to large
    score3, pre3 = [list(l) for l in zip(*info)]
    score3 = np.array(score3)
    pre3 = np.array(pre3)
    r3, p3 = pearsonr(score3, pre3)

    return np.mean(np.array([r1,r2,r3])), np.std(np.array([r1,r2,r3])), np.mean(np.array([p1,p2,p3])), np.std(np.array([p1,p2,p3]))



# confidence score
insecurity_score_cs1 = np.load('./cub200/confidence_score_vgg16bn_layer42_insecurity_score_multi1.npy')
precision_cs1 = np.load('./cub200/confidence_score_vgg16bn_layer42_precision_multi1.npy')
insecurity_score_cs2 = np.load('./cub200/confidence_score_vgg16bn_layer42_insecurity_score_multi2.npy')
precision_cs2 = np.load('./cub200/confidence_score_vgg16bn_layer42_precision_multi2.npy')
insecurity_score_cs3 = np.load('./cub200/confidence_score_vgg16bn_layer42_insecurity_score_multi3.npy')
precision_cs3 = np.load('./cub200/confidence_score_vgg16bn_layer42_precision_multi3.npy')

# entropy
insecurity_score_entropy1 = np.load('./cub200/entropy_vgg16bn_layer42_insecurity_score_multi1.npy')
precision_entropy1 = np.load('./cub200/entropy_vgg16bn_layer42_precision_multi1.npy')
insecurity_score_entropy2 = np.load('./cub200/entropy_vgg16bn_layer42_insecurity_score_multi2.npy')
precision_entropy2 = np.load('./cub200/entropy_vgg16bn_layer42_precision_multi2.npy')
insecurity_score_entropy3 = np.load('./cub200/entropy_vgg16bn_layer42_insecurity_score_multi3.npy')
precision_entropy3 = np.load('./cub200/entropy_vgg16bn_layer42_precision_multi3.npy')

# hardness score
insecurity_score_hp1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_insecurity_score_multi1.npy')
precision_hp1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_multi1.npy')
insecurity_score_hp2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_insecurity_score_multi2.npy')
precision_hp2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_multi2.npy')
insecurity_score_hp3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_insecurity_score_multi3.npy')
precision_hp3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_precision_multi3.npy')

# hardness IG
insecurity_score_IG1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_IG_insecurity_score_multi1.npy')
precision_IG1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_IG_precision_multi1.npy')
insecurity_score_IG2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_IG_insecurity_score_multi2.npy')
precision_IG2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_IG_precision_multi2.npy')
insecurity_score_IG3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_IG_insecurity_score_multi3.npy')
precision_IG3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_IG_precision_multi3.npy')

# Hessian
insecurity_score_2nd1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_2ndG_insecurity_score_multi1.npy')
precision_2nd1 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_2ndG_precision_multi1.npy')
insecurity_score_2nd2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_2ndG_insecurity_score_multi2.npy')
precision_2nd2 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_2ndG_precision_multi2.npy')
insecurity_score_2nd3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_2ndG_insecurity_score_multi3.npy')
precision_2nd3 = np.load('./cub200/hardness_predictor_vgg16bn_layer42_2ndG_precision_multi3.npy')

print('hesitancy')
r_m, r_s, p_m, p_s = pearson_computation(insecurity_score_cs1, precision_cs1, insecurity_score_cs2, precision_cs2, insecurity_score_cs3, precision_cs3)
print(r_m, r_s, p_m, p_s)
print('entropy')
r_m, r_s, p_m, p_s = pearson_computation(insecurity_score_entropy1, precision_entropy1, insecurity_score_entropy2, precision_entropy2, insecurity_score_entropy3, precision_entropy3)
print(r_m, r_s, p_m, p_s)
print('hardness')
r_m, r_s, p_m, p_s = pearson_computation(insecurity_score_hp1, precision_hp1, insecurity_score_hp2, precision_hp2, insecurity_score_hp3, precision_hp3)
print(r_m, r_s, p_m, p_s)
print('IG')
r_m, r_s, p_m, p_s = pearson_computation(insecurity_score_IG1, precision_IG1, insecurity_score_IG2, precision_IG2, insecurity_score_IG3, precision_IG3)
print(r_m, r_s, p_m, p_s)
print('Hessian')
r_m, r_s, p_m, p_s = pearson_computation(insecurity_score_2nd1, precision_2nd1, insecurity_score_2nd2, precision_2nd2, insecurity_score_2nd3, precision_2nd3)
print(r_m, r_s, p_m, p_s)

