import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import datasets
import models as models
import matplotlib.pyplot as plt
import torchvision.models as torch_models
from extra_setting import *
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import utils
import scipy.io as sio
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import cv2
import seaborn as sns
import operator
from scipy import misc


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch end2end ade Training')
parser.add_argument('-d', '--dataset', default='ade', help='dataset name')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet20)')
parser.add_argument('-c', '--channel', type=int, default=16,
                    help='first conv channel (default: 16)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--gpu', default='0', help='index of gpus to use')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_step', default='5', help='decreasing strategy')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./ade/checkpoint_vgg16bn_hp1.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--first_epochs', default=5, type=int, metavar='N',
                    help='number of first stage epochs to run')

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    # select gpus
    args.gpu = args.gpu.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    # data loader
    assert callable(datasets.__dict__[args.dataset])
    get_dataset = getattr(datasets, args.dataset)
    num_classes = datasets._NUM_CLASSES[args.dataset]
    train_loader, val_loader = get_dataset(
        batch_size=args.batch_size, num_workers=args.workers)

    # create model
    model_main = models.__dict__['vgg16_bn'](pretrained=True)
    model_main.classifier[-1] = nn.Linear(model_main.classifier[-1].in_features, num_classes)
    model_main = torch.nn.DataParallel(model_main, device_ids=range(len(args.gpu))).cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_main.module.load_state_dict(checkpoint['state_dict_m'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    model_ahp_trunk = models.__dict__['vgg16_bn'](pretrained=True)
    model_ahp_trunk.classifier[-1] = nn.Linear(model_ahp_trunk.classifier[-1].in_features, 1000)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_ahp_trunk.load_state_dict(checkpoint['state_dict_ahp_trunk'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    model_ahp_trunk = torch.nn.DataParallel(model_ahp_trunk, device_ids=range(len(args.gpu))).cuda()

    model_ahp_hp = models.__dict__['ahp_net_hp_res50_presigmoid']()
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_ahp_hp.load_state_dict(checkpoint['state_dict_ahp_hp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    model_ahp_hp = torch.nn.DataParallel(model_ahp_hp, device_ids=range(len(args.gpu))).cuda()


    # generate predicted hardness score
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_f = nn.CrossEntropyLoss(reduce=False).cuda()
    prec1, prec5, all_correct_te, all_predicted_te, all_class_dis_te = validate(val_loader, model_main, model_ahp_trunk,
                                                              model_ahp_hp, criterion, criterion_f)
    all_predicted_te = all_predicted_te.astype(int)
    np.save('./ade/all_correct_vgg16_te_multi1.npy', all_correct_te)
    np.save('./ade/all_predicted_vgg16_te_multi1.npy', all_predicted_te)
    np.save('./ade/all_class_dis_vgg16_te_multi1.npy', all_class_dis_te)

    all_correct_te = np.load('./ade/all_correct_vgg16_te_multi1.npy')
    all_predicted_te = np.load('./ade/all_predicted_vgg16_te_multi1.npy')
    all_class_dis_te = np.load('./ade/all_class_dis_vgg16_te_multi1.npy')


    hardness_scores_te, hardness_te_idx_each = save_predicted_hardness(train_loader, val_loader, model_ahp_trunk, model_ahp_hp)
    np.save('./ade/hardness_scores_te_vgg16_multi1.npy', hardness_scores_te)
    np.save('./ade/hardness_te_idx_each_vgg16_multi1.npy', hardness_te_idx_each)

    hardness_scores_te = np.load('./ade/hardness_scores_te_vgg16_multi1.npy')
    hardness_te_idx_each = np.load('./ade/hardness_te_idx_each_vgg16_multi1.npy')

    predicted_hardness_score_te = np.copy(hardness_scores_te)

    # pickup K hardnest examples
    test_info = zip(all_correct_te, hardness_scores_te, hardness_te_idx_each)
    test_info = sorted(test_info, key=lambda test: test[1])  # from small to large
    all_correct_te, hardness_scores_te, hardness_te_idx_each = [list(l) for l in zip(*test_info)]
    all_correct_te = np.array(all_correct_te)
    hardness_scores_te = np.array(hardness_scores_te)
    hardness_te_idx_each = np.array(hardness_te_idx_each)

    # hardness_idx_cor_classified = hardness_te_idx_each * all_correct_te
    # hardness_idx_incor_classified = hardness_te_idx_each * (1 - all_correct_te)
    # hardness_idx_cor_classified = hardness_idx_cor_classified[np.nonzero(hardness_idx_cor_classified)]
    # hardness_idx_incor_classified = hardness_idx_incor_classified[np.nonzero(hardness_idx_incor_classified)]
    #
    # K = 100
    # K_idx_incor_classified = hardness_idx_incor_classified[-K:]
    # K_idx_incor_classified = K_idx_incor_classified.astype(int)  # index K hard examples

    K = 100
    K_idx_incor_classified = hardness_te_idx_each[-K:]
    K_idx_incor_classified = K_idx_incor_classified.astype(int)

    imlist = []
    imclass = []

    with open('./ade/ADEChallengeData2016/ADE_gt_val.txt', 'r') as rf:
        for line in rf.readlines():
            impath, imlabel, imindex = line.strip().split()
            imlist.append(impath)
            imclass.append(imlabel)

    picked_list = []
    picked_class_list = []
    for i in range(K):
        picked_list.append(imlist[K_idx_incor_classified[i]])
        picked_class_list.append(imclass[K_idx_incor_classified[i]])

    secondgrad_hp = SecondGrad_hp(model_ahp_trunk, model_ahp_hp, target_layer_names=["42"], use_cuda=True)
    grad_cam_cls = GradCam_cls(model_main, target_layer_names=["42"], use_cuda=True)

    com_extracted_attributes = np.load('./ade/com_extracted_attributes_001.npy')

    picked_seg_list = []
    for i in range(K):
        img_name = picked_list[i]
        img_name = img_name[:27] + "annotations" + img_name[33:-3] + "png"
        picked_seg_list.append(img_name)

    topK_prob_predicted_classes, _ = largest_indices_each_example(all_class_dis_te, 5)
    picked_topK_prob_predicted_classes = topK_prob_predicted_classes[K_idx_incor_classified, :]

    # save ade hard info
    adehard = './ade/ADEChallengeData2016/ADEhard_gt_val.txt'
    fl = open(adehard, 'w')
    for ii in range(K):
        example_info = picked_list[ii] + " " + picked_class_list[ii] + " " + str(K_idx_incor_classified[ii])
        fl.write(example_info)
        fl.write("\n")
    fl.close()

    # data loader
    assert callable(datasets.__dict__['adehard'])
    get_dataset = getattr(datasets, 'adehard')
    num_classes = datasets._NUM_CLASSES['adehard']
    _, val_hard_loader = get_dataset(
        batch_size=1, num_workers=args.workers)


    remaining_mask_size_pool = np.arange(0.01, 1.0, 0.01)
    IOU = insecurity_extraction(val_hard_loader, secondgrad_hp, grad_cam_cls,
                                                                     picked_list, 3, com_extracted_attributes,
                                                                     picked_seg_list,
                                                                     picked_topK_prob_predicted_classes,
                                                                     remaining_mask_size_pool)


    print(IOU)
    np.save('./ade/hardness_predictor_vgg16_layer42_2ndG_IOU_multi1.npy', IOU)



def validate(val_loader, model_main, model_ahp_trunk, model_ahp_hp, criterion, criterion_f):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model_main.eval()
    model_ahp_trunk.eval()
    model_ahp_hp.eval()
    end = time.time()

    all_correct_te = []
    all_predicted_te = []
    all_class_dis = np.zeros((1, 1040))
    for i, (input, target, index) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda(async=True)

        # compute output
        output = model_main(input)
        class_dis = F.softmax(output, dim=1)
        class_dis = class_dis.data.cpu().numpy()
        all_class_dis = np.concatenate((all_class_dis, class_dis), axis=0)

        loss = criterion(output, target)
        trunk_output = model_ahp_trunk(input)
        predicted_hardness_scores, _ = model_ahp_hp(trunk_output)

        p_i_m = torch.max(output, dim=1)[1]
        all_predicted_te = np.concatenate((all_predicted_te, p_i_m), axis=0)
        p_i_m = p_i_m.long()
        p_i_m[p_i_m - target == 0] = -1
        p_i_m[p_i_m > -1] = 0
        p_i_m[p_i_m == -1] = 1
        correct = p_i_m.float()
        all_correct_te = np.concatenate((all_correct_te, correct), axis=0)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    # print(' * Testing Prec@1 {top1.avg:.3f}'.format(top1=top1))
    all_class_dis = all_class_dis[1:, :]
    return top1.avg, top5.avg, all_correct_te, all_predicted_te, all_class_dis


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def largest_indices_each_example(all_response, topK):
    topK_maxIndex = np.zeros((np.size(all_response, 0), topK), dtype=np.int16)
    topK_maxValue = np.zeros((np.size(all_response, 0), topK))
    for i in range(np.size(topK_maxIndex, 0)):
        arr = all_response[i, :]
        topK_maxIndex[i, :] = np.argsort(arr)[-topK:][::-1]
        topK_maxValue[i, :] = np.sort(arr)[-topK:][::-1]
    return topK_maxIndex, topK_maxValue


def save_predicted_hardness(train_loader, val_loader, model_ahp_trunk, model_ahp_hp):
    model_ahp_trunk.eval()
    model_ahp_hp.eval()
    # hardness_scores_tr = []
    # hardness_scores_idx_tr = []
    # for i, (input, target, index) in enumerate(train_loader):
    #     input = input.cuda()
    #     predicted_hardness_scores =  model_ahp_hp(model_ahp_trunk(input)).squeeze()
    #     scores = predicted_hardness_scores.data.cpu().numpy()
    #     hardness_scores_tr = np.concatenate((hardness_scores_tr, scores), axis=0)
    #     index = index.numpy()
    #     hardness_scores_idx_tr = np.concatenate((hardness_scores_idx_tr, index), axis=0)

    hardness_scores_val = []
    hardness_scores_idx_val = []
    for i, (input, target, index) in enumerate(val_loader):
        input = input.cuda()
        trunk_output = model_ahp_trunk(input)
        predicted_hardness_scores, _ = model_ahp_hp(trunk_output)
        scores = predicted_hardness_scores.data.cpu().numpy().squeeze()
        hardness_scores_val = np.concatenate((hardness_scores_val, scores), axis=0)
        index = index.numpy()
        hardness_scores_idx_val = np.concatenate((hardness_scores_idx_val, index), axis=0)

    return hardness_scores_val, hardness_scores_idx_val


def save_checkpoint(state, filename='checkpoint_res.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res





class FeatureExtractor_hp():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules['module']._modules['features']._modules.items():
            x = module(x)  # forward one layer each time
            if name in self.target_layers:  # store the gradient of target layer
                x.register_hook(self.save_gradient)
                outputs += [x]  # after last feature map, nn.MaxPool2d(kernel_size=2, stride=2)] follows

        x = x.view(x.size(0), -1)
        x = self.model._modules['module'].classifier(x)
        return outputs, x


class FeatureExtractor_cls():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules['module']._modules['features']._modules.items():
            x = module(x)  # forward one layer each time
            if name in self.target_layers:  # store the gradient of target layer
                x.register_hook(self.save_gradient)
                outputs += [x]  # after last feature map, nn.MaxPool2d(kernel_size=2, stride=2)] follows
        return outputs, x

class ModelOutputs_hp():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model_hp_trunk, model_hp_head, target_layers):
        self.model_hp_trunk = model_hp_trunk
        self.model_hp_head = model_hp_head
        self.feature_extractor = FeatureExtractor_hp(self.model_hp_trunk, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        output, _ = self.model_hp_head(output)
        return target_activations, output


class ModelOutputs_cls():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor_cls(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model._modules['module'].classifier(output)  # travel many fc layers
        return target_activations, output


def preprocess_image(img):
    # means=[0.485, 0.456, 0.406]
    # stds=[0.229, 0.224, 0.225]
    means = [0.4706145, 0.46000465, 0.45479808]
    stds = [0.26668432, 0.26578658, 0.2706199]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam

def show_segment_on_image(img, mask, mark_locs=None, is_cls=True):
    img = np.float32(img)
    # if is_cls == False:
    #     threshold = np.sort(mask.flatten())[-int(0.05*224*224)]
    #     mask[mask < threshold] = 0
    #     mask[mask > 0] = 1
    mask = np.concatenate((mask[:, :, np.newaxis], mask[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
    img = np.uint8(255 * mask * img)
    if is_cls == False:
        if np.sum(mark_locs) > 0:
            x, y = np.where(mark_locs == 1)
            for i in range(np.size(x)):
                cv2.circle(img, (y[i], x[i]), 2, (0,0,255))
    return img



class SecondGrad_hp:
    def __init__(self, model_hp_trunk, model_hp_head, target_layer_names, use_cuda):
        self.model_hp_trunk = model_hp_trunk
        self.model_hp_head = model_hp_head
        self.model_hp_trunk.eval()
        self.model_hp_head.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model_hp_trunk = model_hp_trunk.cuda()
            self.model_hp_head = model_hp_head.cuda()

        self.extractor = ModelOutputs_hp(self.model_hp_trunk, self.model_hp_head, target_layer_names)

    def forward(self, input):
        return self.model_hp_head(self.model_hp_trunk(input))

    def __call__(self, input):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        grad_feature = torch.autograd.grad(output, features, create_graph=True)
        grad_feature = grad_feature[0].squeeze()
        grad2_fearure = np.zeros((14,14,512,512))


        for i_W in range(14):
            for i_H in range(14):
                for i_D in range(512):
                    cur_grad_feature = torch.autograd.grad(grad_feature[i_D,i_W,i_H], features, create_graph=True)
                    cur_grad_feature = cur_grad_feature[0].squeeze()
                    cur_grad_feature = cur_grad_feature.cpu().data.numpy()
                    grad2_fearure[i_W, i_H, i_D, :] = cur_grad_feature[:, i_W, i_H]


        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        secondresponse = np.zeros((14, 14))
        for i_W in range(14):
            for i_H in range(14):
                firstTwoMatrices = target[:, i_W, i_H].squeeze() @ grad2_fearure[i_W, i_H, :, :].squeeze()
                secondresponse[i_W, i_H] = firstTwoMatrices @ target[:, i_W, i_H].squeeze()

        secondresponse = np.abs(secondresponse)

        grads_val = grad_feature.cpu().data.numpy()

        mask_positive = np.copy(grads_val)
        mask_positive[mask_positive < 0.0] = 0.0
        mask_positive = mask_positive.squeeze()

        cam_positive = target * mask_positive
        cam_positive = np.sum(cam_positive, axis=0)

        cam_positive = cam_positive + secondresponse / 2.0

        return cam_positive


class GradCam_cls:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs_cls(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, TopKclass = 5, topK_prob_predicted_classes=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        classifier_heatmaps = np.zeros((np.size(target,2), np.size(target,2), np.size(topK_prob_predicted_classes)))
        for i_cls in range(np.size(topK_prob_predicted_classes)):
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][topK_prob_predicted_classes[i_cls]] = 1
            one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
            if self.cuda:
                one_hot = torch.sum(one_hot.cuda() * output)
            else:
                one_hot = torch.sum(one_hot * output)
            self.model.zero_grad()
            one_hot.backward(retain_graph=True)
            grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
            grads_val = grads_val.squeeze()
            cam_positive = target * grads_val
            cam_positive = np.sum(cam_positive, axis=0)
            classifier_heatmaps[:, :, i_cls] = cam_positive

        return classifier_heatmaps



def insecurity_extraction(val_loader, secondgrad_hp, grad_cam_cls, imglist, topKcls, com_extracted_attributes, seg_list, topK_prob_predicted_classes, remaining_mask_size_pool):

    IOU = np.zeros((len(imglist), np.size(remaining_mask_size_pool)))

    for i, (input, target, index) in enumerate(val_loader):
        print('processing sample', i)

        img = cv2.imread(imglist[i])
        img_X_max = np.size(img, axis=0)
        img_Y_max = np.size(img, axis=1)
        hardness_heatmaps = secondgrad_hp(input)
        hardness_mask = np.copy(hardness_heatmaps)
        hardness_mask[hardness_mask > 0] = 1
        classifier_heatmaps = grad_cam_cls(input, 1040, topK_prob_predicted_classes[i, :])
        classifier_heatmaps[classifier_heatmaps < 0] = 1e-7

        seg_img = misc.imread(seg_list[i])
        seg_img = np.resize(seg_img, (224, 224))

        confusion_classes = np.argsort(classifier_heatmaps, axis=2)[:, :, -topKcls:]
        confusion_classes = np.sort(confusion_classes, axis=2)
        confusion_classes = confusion_classes * np.repeat(hardness_mask[:,:,np.newaxis], topKcls, axis=2)

        misclass_pairs = np.zeros((1, 2))
        for i_cls in range(topKcls):
            for j_cls in range(i_cls+1, topKcls):
                cur_misclass_pairs = np.concatenate((np.reshape(confusion_classes[:,:,i_cls].squeeze(), (-1, 1)), np.reshape(confusion_classes[:,:,j_cls].squeeze(), (-1, 1))), axis=1)
                misclass_pairs = np.concatenate((misclass_pairs, cur_misclass_pairs), axis=0)

        misclass_pairs = np.unique(misclass_pairs, axis=0)
        misclass_pairs = misclass_pairs[~np.all(misclass_pairs == 0, axis=1)]
        misclass_pairs = misclass_pairs.astype(int)

        atom_num = np.size(misclass_pairs, axis=0)
        atom_mask = np.zeros(np.size(hardness_mask))

        for i_remain in range(np.size(remaining_mask_size_pool)):
            remaining_mask_size = remaining_mask_size_pool[i_remain]
            noeffect_atom = 0
            total_IOU_i = 0
            for i_atom in range(atom_num):
                cur_atom_mask1 = np.zeros((np.size(classifier_heatmaps, 0), np.size(classifier_heatmaps, 0), topKcls))
                cur_atom_mask1[np.where(confusion_classes[:, :, :] == misclass_pairs[i_atom, 0])] = 1
                cur_atom_mask1 = np.sum(cur_atom_mask1, axis=-1)
                cur_atom_mask2 = np.zeros((np.size(classifier_heatmaps, 0), np.size(classifier_heatmaps, 0), topKcls))
                cur_atom_mask2[np.where(confusion_classes[:, :, :] == misclass_pairs[i_atom, 1])] = 1
                cur_atom_mask2 = np.sum(cur_atom_mask2, axis=-1)
                atom_mask = cur_atom_mask1 * cur_atom_mask2

                hardness_atom = classifier_heatmaps[:, :, misclass_pairs[i_atom, 0]].squeeze() * classifier_heatmaps[:, :, misclass_pairs[i_atom, 1]].squeeze() * hardness_heatmaps * atom_mask
                hardness_atom = cv2.resize(hardness_atom, (224, 224))
                hardness_atom_mask = np.copy(hardness_atom)

                threshold = np.sort(hardness_atom_mask.flatten())[int(-remaining_mask_size * 224 * 224)]
                hardness_atom_mask[hardness_atom_mask > threshold] = 1
                hardness_atom_mask[hardness_atom_mask < 1] = 0

                com_attributes = com_extracted_attributes[topK_prob_predicted_classes[i, misclass_pairs[i_atom, 0]], topK_prob_predicted_classes[i, misclass_pairs[i_atom, 1]]]

                if len(com_attributes) == 0:
                    noeffect_atom = noeffect_atom + 1
                    continue
                com_attributes = np.array(com_attributes)

                commom_seg_img = np.zeros((224, 224))
                for i_com in range(np.size(com_attributes)):
                    commom_seg_img[seg_img == com_attributes[i_com]] = 1
                IOU_i = np.sum(hardness_atom_mask * commom_seg_img) / np.sum(hardness_atom_mask + commom_seg_img - hardness_atom_mask * commom_seg_img)
                total_IOU_i = total_IOU_i + IOU_i
            IOU[i, i_remain] = total_IOU_i / (atom_num - noeffect_atom)

    return np.mean(IOU, axis=0)



def mask_IOU(A, B):
    return np.sum(A * B) / (np.sum(A+B) - np.sum(A * B))

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def get_second_order_grad(grads, xs):
    start = time.time()
    grads2 = []
    for j, (grad, x) in enumerate(zip(grads, xs)):
        print('2nd order on ', j, 'th layer')
        print(x.size())
        grad = torch.reshape(grad, [-1])
        grads2_tmp = []
        for count, g in enumerate(grad):
            g2 = torch.autograd.grad(g, x, retain_graph=True)[0]
            g2 = torch.reshape(g2, [-1])
            grads2_tmp.append(g2[count].data.cpu().numpy())
        grads2.append(torch.from_numpy(np.reshape(grads2_tmp, x.size())).to(DEVICE_IDS[0]))
        print('Time used is ', time.time() - start)

if __name__ == '__main__':
    main()



