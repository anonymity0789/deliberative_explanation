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
import scipy.io as sio
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge



model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch end2end cub200 Training')
parser.add_argument('-d', '--dataset', default='cub200', help='dataset name')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet20)')
parser.add_argument('-c', '--channel', type=int, default=16,
                    help='first conv channel (default: 16)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--gpu', default='0,1,2,3', help='index of gpus to use')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
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
parser.add_argument('--resume', default='./cub200/checkpoint_pretrain_res50.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--first_epochs', default=5, type=int, metavar='N',
                    help='number of first stage epochs to run')

best_prec1 = 0
REMOVE_STEP = 99
REPEAT_NUM = 10

def main():
    global args, best_prec1
    args = parser.parse_args()

    # training multiple times for obtaining acc mean and variance
    remain_bool_te = np.zeros([REPEAT_NUM, 5794])
    hardness_tr = np.zeros([REPEAT_NUM, 5994])
    hardness_tr_idx = np.zeros([REPEAT_NUM, 5994])
    hardness_te = np.zeros([REPEAT_NUM, 5794])
    hardness_te_idx = np.zeros([REPEAT_NUM, 5794])
    all_accuracy = np.zeros([REPEAT_NUM, REMOVE_STEP])
    remain_num_te = np.zeros([REPEAT_NUM, REMOVE_STEP])

    # select gpus
    args.gpu = args.gpu.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    # data loader
    assert callable(datasets.__dict__[args.dataset])
    get_dataset = getattr(datasets, args.dataset)
    num_classes = datasets._NUM_CLASSES[args.dataset]
    train_loader, val_loader = get_dataset(
        batch_size=args.batch_size, num_workers=args.workers)

    threshold = 7.5
    for j in range(REPEAT_NUM):

        # create model
        model_main = torch_models.resnet50(pretrained=True)
        model_main.fc = nn.Linear(512 * 4, num_classes)
        model_main = torch.nn.DataParallel(model_main, device_ids=range(len(args.gpu))).cuda()
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                # args.start_epoch = checkpoint['epoch']
                model_main.module.load_state_dict(checkpoint['state_dict_m'])
                # optimizer_m1.load_state_dict(checkpoint['optimizer_m'])
                # optimizer_ahp_hp.load_state_dict(checkpoint['optimizer_ahp_hp'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        model_ahp_trunk = torch_models.resnet50(pretrained=True)
        model_ahp_trunk.fc = nn.Linear(512 * 4, num_classes)
        # model_ahp_trunk = torch.nn.DataParallel(model_ahp_trunk, device_ids=range(len(args.gpu))).cuda()
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                # args.start_epoch = checkpoint['epoch']
                model_ahp_trunk.load_state_dict(checkpoint['state_dict_m'])
                # optimizer_m1.load_state_dict(checkpoint['optimizer_m'])
                # optimizer_ahp_hp.load_state_dict(checkpoint['optimizer_ahp_hp'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        model_ahp_trunk.fc = nn.Linear(512 * 4, 1000)
        model_ahp_trunk = torch.nn.DataParallel(model_ahp_trunk, device_ids=range(len(args.gpu))).cuda()
        model_ahp_hp = models.__dict__['ahp_net_hp_res50']()
        model_ahp_hp = torch.nn.DataParallel(model_ahp_hp, device_ids=range(len(args.gpu))).cuda()

        criterion = nn.CrossEntropyLoss().cuda()
        criterion_f = nn.CrossEntropyLoss(reduce=False).cuda()

        optimizer_m = torch.optim.SGD(model_main.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        optimizer_ahp_trunk = torch.optim.Adam(model_ahp_trunk.parameters(), lr=0.00001, weight_decay=1e-3)
        optimizer_ahp_hp = torch.optim.Adam(model_ahp_hp.parameters(), lr=0.00001, weight_decay=1e-3)

        cudnn.benchmark = True

        all_acc_cor = np.zeros(args.epochs)
        all_acc_incor = np.zeros(args.epochs)

        # train nn in order to get the feature vector for each sample
        lr_step = np.arange(args.start_epoch + 1, args.epochs).tolist()
        AUC = 0.0
        for epoch in range(args.start_epoch, args.epochs):
            if epoch in lr_step:
                for param_group in optimizer_m.param_groups:
                    param_group['lr'] *= 0.95

            train_ap(train_loader, model_main, model_ahp_trunk, model_ahp_hp, optimizer_m, optimizer_ahp_trunk, optimizer_ahp_hp, epoch, criterion_f)

            # evaluate on validation set
            prec1, prec5, all_correct_te, all_p_i_c = validate(val_loader, model_main, criterion, criterion_f)

            hardness_scores_te, hardness_te_idx_each = save_predicted_hardness(train_loader, val_loader,
                                                                               model_ahp_trunk, model_ahp_hp)

            remain_bool_te[j, :] = all_correct_te
            hardness_te[j, :] = hardness_scores_te

            idx_matrix1 = np.zeros(len(val_loader.dataset))
            idx_matrix2 = np.zeros(len(val_loader.dataset))
            idx_matrix3 = np.zeros(len(val_loader.dataset))
            idx_matrix4 = np.zeros(len(val_loader.dataset))

            idx_matrix1[all_p_i_c < 0.5] = 1.0
            idx_matrix2[hardness_scores_te > 0.5] = 1.0
            idx_matrix3[all_p_i_c > 0.5] = 1.0
            idx_matrix4[hardness_scores_te < 0.5] = 1.0
            total_cor_num_cor = np.sum(idx_matrix3 * idx_matrix4)
            total_cor_num_incor = np.sum(idx_matrix1 * idx_matrix2)

            all_acc_cor[epoch] = total_cor_num_cor / np.sum(all_correct_te)
            all_acc_incor[epoch] = total_cor_num_incor / (5794 - np.sum(all_correct_te))


            # compute acc under different removing percentage
            all_correct_te = all_correct_te.tolist()
            hardness_scores_te = hardness_scores_te.tolist()
            test_info = zip(all_correct_te, hardness_scores_te)
            test_info = sorted(test_info, key=lambda test: test[1])  # from small to large
            all_correct_te, hardness_scores_te = [list(l) for l in zip(*test_info)]
            all_correct_te = np.array(all_correct_te)
            hardness_scores_te = np.array(hardness_scores_te)
            all_accuracy_each = np.zeros(REMOVE_STEP)
            remain_num_te_each = np.zeros(REMOVE_STEP)
            for i in range(1, REMOVE_STEP + 1):
                removed_num = int(0.01 * i * len(val_loader.dataset))
                current_acc = np.sum(all_correct_te[:-removed_num]) / (len(val_loader.dataset) - removed_num)
                all_accuracy_each[i - 1] = current_acc
                remain_num_te_each[i - 1] = len(val_loader.dataset) - removed_num

            cur_AUC = np.trapz(all_accuracy_each - all_accuracy_each[0])

            if cur_AUC > AUC:
                AUC = cur_AUC
            print(cur_AUC)
            print(AUC)
            if AUC > threshold:
                threshold = threshold + 0.02
                all_accuracy[j, :] = all_accuracy_each
                remain_num_te[j, :] = remain_num_te_each

                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict_m': model_main.state_dict(),
                    'state_dict_ahp_trunk': model_ahp_trunk.state_dict(),
                    'state_dict_ahp_hp': model_ahp_hp.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer_m1': optimizer_m.state_dict(),
                    'optimizer_ahp_trunk': optimizer_ahp_trunk.state_dict(),
                }, filename='./cub200/checkpoint_res50_hp.pth.tar')



def train(train_loader, model_main, optimizer_m, epoch, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_m = AverageMeter()
    losses_a = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model_main.train()

    end = time.time()
    for i, (input, target, index) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # input and target
        input = input.cuda()
        target = target.cuda(async=True)

        # compute output
        predicted_labels = model_main(input)

        loss_m = criterion(predicted_labels, target)
        prec1, prec5 = accuracy(predicted_labels, target, topk=(1, 5))
        losses_m.update(loss_m.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer_m.zero_grad()
        loss_m.backward()
        optimizer_m.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            curr_lr_m = optimizer_m.param_groups[0]['lr']
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: [{4}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_m {loss_m.val:.4f} ({loss_m.avg:.4f})\t'
                  'Loss_a {loss_a.val:.4f} ({loss_a.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, args.epochs, i, len(train_loader), curr_lr_m,
                batch_time=batch_time, data_time=data_time, loss_m=losses_m, loss_a=losses_a, top1=top1, top5=top5))


def train_ap(train_loader, model_main, model_ahp_trunk, model_ahp_hp, optimizer_m, optimizer_ahp_trunk, optimizer_ahp_hp, epoch, criterion_f):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_m = AverageMeter()
    losses_a = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model_main.train()
    model_ahp_trunk.train()
    model_ahp_hp.train()

    end = time.time()
    for i, (input, target, index) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # # input and target
        # input = input.cuda()
        # target = target.cuda(async=True)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        predicted_labels = model_main(input_var)
        loss_m = criterion_f(predicted_labels, target_var).squeeze()
        predicted_hardness_scores = model_ahp_hp(model_ahp_trunk(input_var)).squeeze()
        loss_m = torch.mean(loss_m * predicted_hardness_scores)
        loss_a, _, _ = opposite_loss(predicted_labels, predicted_hardness_scores, target, criterion_f)

        prec1, prec5 = accuracy(predicted_labels, target, topk=(1, 5))
        losses_m.update(loss_m.item(), input.size(0))
        losses_a.update(loss_a.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer_m.zero_grad()
        loss_m.backward(retain_graph=True)
        optimizer_m.step()

        optimizer_ahp_hp.zero_grad()
        loss_a.backward(retain_graph=True)
        optimizer_ahp_hp.step()

        optimizer_ahp_trunk.zero_grad()
        loss_a.backward()
        optimizer_ahp_trunk.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            curr_lr_m = optimizer_m.param_groups[0]['lr']
            curr_lr_a = optimizer_ahp_trunk.param_groups[0]['lr']
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: [{4}][{5}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_m {loss_m.val:.4f} ({loss_m.avg:.4f})\t'
                  'Loss_a {loss_a.val:.4f} ({loss_a.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, args.epochs, i, len(train_loader), curr_lr_m, curr_lr_a,
                batch_time=batch_time, data_time=data_time, loss_m=losses_m, loss_a=losses_a, top1=top1, top5=top5))


def validate(val_loader, model_main, criterion, criterion_f):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model_main.eval()
    end = time.time()

    all_correct_te = []
    all_p_i_c = []

    for i, (input, target, index) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda(async=True)

        # compute output
        output = model_main(input)
        loss = criterion(output, target)
        p_i_c = getting_pic(output, target, criterion_f)
        p_i_c = p_i_c.data.cpu().numpy()
        all_p_i_c = np.concatenate((all_p_i_c, p_i_c), axis=0)

        p_i_m = torch.max(output, dim=1)[1]
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

    return top1.avg, top5.avg, all_correct_te, all_p_i_c


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
        predicted_hardness_scores = model_ahp_hp(model_ahp_trunk(input)).squeeze()
        scores = predicted_hardness_scores.data.cpu().numpy()
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


def opposite_loss(predicted_labels, predicted_hardness_scores, target, criterion):

    bonus = 0.55
    cross_entropy_loss = criterion(predicted_labels, target).squeeze()
    cross_entropy_loss = (-1) * cross_entropy_loss
    p_i_c = torch.exp(cross_entropy_loss)

    p_i_m = torch.max(predicted_labels, dim=1)[1]
    p_i_m = p_i_m.long()
    p_i_m[p_i_m - target == 0] = -1
    p_i_m[p_i_m > -1] = 0
    p_i_m[p_i_m == -1] = 1
    error = p_i_m
    error = error.float()
    error_loss = 1.0 - error  # incorrect = 1, otherwise = 0

    # adjust p_i_c
    p_i_c = torch.squeeze(p_i_c)
    p_i_c_tmp = p_i_c.clone()
    p_i_c_tmp[p_i_c < 0.5] = -1
    p_i_c_tmp[error_loss < 1.0] = bonus
    p_i_c = torch.max(p_i_c_tmp, p_i_c)

    term1 = (1 - p_i_c) * predicted_hardness_scores
    term2 = (1 - predicted_hardness_scores) * p_i_c
    final_loss = 1 - term1 - term2

    tmp1 = p_i_c<0.5
    tmp1 = tmp1.float()
    tmp2 = p_i_c>0.5
    tmp2 = tmp2.float()
    weight1 = torch.sum(tmp1) / p_i_c.size(0)
    weight2 = torch.sum(tmp2) / p_i_c.size(0)

    weights = torch.zeros(p_i_c.size(0))
    weights = weights.cuda()
    weights[p_i_c > 0.5] = weight1
    weights[p_i_c < 0.5] = weight2
    final_loss = final_loss * weights * 2.0

    return torch.mean(final_loss), p_i_c, final_loss


if __name__ == '__main__':
    main()
