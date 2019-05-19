import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import os
from PIL import Image



_NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'mnist': 10,
    'flowers': 17,
    'imagenet': 1000,
    'indoor': 67,
    'inaturalist': 1405,
    'inaturalist_supercategory': 14,
    'inaturalist_family': 1120,
    'inaturalist_genus': 4412,
    'inaturalist_phylum': 25,
    'inaturalist_kingdom': 6,
    'inaturalist_class': 57,
    'inaturalist_order': 273,
    'cars196': 196,
    'cub200': 200,
    'hard_mnist': 10,
    'cub200hard': 200,
    'ade': 1040,
    'adehard': 1040,
}


# def mnist(batch_size, data_root='./mnist', train=True, val=True, **kwargs):
#     data_root = os.path.expanduser(os.path.join(data_root, 'mnist-data'))
#     num_workers = kwargs.setdefault('num_workers', 1)
#     kwargs.pop('input_size', None)
#     print("Building MNIST data loader with {} workers".format(num_workers))
#     ds = []
#     if train:
#         train_loader = torch.utils.data.DataLoader(
#             datasets.MNIST(
#                 root=data_root, train=True, download=True,
#                 transform=transforms.Compose([
#                     transforms.Pad(4),
#                     transforms.RandomCrop(28),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                 ])),
#             batch_size=batch_size, shuffle=True, **kwargs)
#         print("MNIST training data size: {}".format(len(train_loader.dataset.train_data)))
#         ds.append(train_loader)
#     if val:
#         test_loader = torch.utils.data.DataLoader(
#             datasets.MNIST(
#                 root=data_root, train=False, download=True,
#                 transform=transforms.Compose([
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                 ])),
#             batch_size=batch_size, shuffle=False, **kwargs)
#         print("MNIST testing data size: {}".format(len(test_loader.dataset.test_data)))
#         ds.append(test_loader)
#     ds = ds[0] if len(ds) == 1 else ds
#     return ds


# def mnist(batch_size, train=True, val=True, **kwargs):
#
#     train_list = '/data6/peiwang/datasets/mnist_png/mnist_gt_tr.txt'
#     val_list = '/data6/peiwang/datasets/mnist_png/mnist_gt_te.txt'
#     # train_list = '/data6/peiwang/datasets/mnist_png/mnist_gt_79_tr.txt'
#     # val_list = '/data6/peiwang/datasets/mnist_png/mnist_gt_79_te.txt'
#     # train_list = '/data6/peiwang/datasets/mnist_png/mnist_gt_7_tr.txt'
#     # val_list = '/data6/peiwang/datasets/mnist_png/mnist_gt_7_te.txt'
#
#
#     num_workers = kwargs.setdefault('num_workers', 1)
#     kwargs.pop('input_size', None)
#     print("Building data loader with {} workers".format(num_workers))
#     ds = []
#
#     if train:
#         train_loader = torch.utils.data.DataLoader(
#             ImageFilelist_MNIST(
#                 flist=train_list,
#                 transform=transforms.Compose(
#                     [transforms.Scale(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
#             ),
#             batch_size=batch_size, shuffle=True, **kwargs)
#         print("Training data size: {}".format(len(train_loader.dataset)))
#         ds.append(train_loader)
#
#     if val:
#         test_loader = torch.utils.data.DataLoader(
#             ImageFilelist_MNIST(
#                 flist=val_list,
#                 transform=transforms.Compose(
#                     [transforms.Scale(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
#             ),
#             batch_size=batch_size, shuffle=False, **kwargs)
#         print("Testing data size: {}".format(len(test_loader.dataset)))
#         ds.append(test_loader)
#     ds = ds[0] if len(ds) == 1 else ds
#     return ds


def mnist(batch_size, train=True, val=True, **kwargs):
    train_list = '/data6/peiwang/datasets/mnist_png/mnist_gt_tr.txt'
    val_list = '/data6/peiwang/datasets/mnist_png/mnist_gt_te.txt'
    # val_list = '/data6/peiwang/datasets/hard_mnist_gt_tr.txt'

    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist_MNIST(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(28),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist_MNIST(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds



def hard_mnist(batch_size, train=True, val=True, **kwargs):

    standard_train_list = '/data6/peiwang/datasets/mnist_png/mnist_gt_tr.txt'
    standard_val_list = '/data6/peiwang/datasets/mnist_png/mnist_gt_te.txt'
    hard_train_list = '/data6/peiwang/datasets/hard_mnist_gt_tr.txt'
    whole_hard_train_list = '/data6/peiwang/datasets/whole_hard_mnist_gt_tr.txt'

    train_list = whole_hard_train_list
    val_list = standard_val_list

    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist_MNIST(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(28),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist_MNIST(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def cifar10(batch_size, data_root='./cifar10', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            CIFAR10(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("CIFAR-10 training data size: {}".format(len(train_loader.dataset.train_data)))
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("CIFAR-10 testing data size: {}".format(len(test_loader.dataset.test_data)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def cifar100(batch_size, data_root='./cifar100', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            CIFAR100(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("CIFAR-100 training data size: {}".format(len(train_loader.dataset.data)))
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            CIFAR100(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("CIFAR-100 testing data size: {}".format(len(test_loader.dataset.data)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def flowers(batch_size, data_root='./flowers', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(data_root)
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=os.path.join(data_root, 'train'),
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=os.path.join(data_root, 'val'),
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


# def imagenet(batch_size, data_root='./data/imgDB/DB/ILSVRC/2012/', train=True, val=True, **kwargs):
#     data_root = os.path.expanduser(data_root)
#     num_workers = kwargs.setdefault('num_workers', 1)
#     kwargs.pop('input_size', None)
#     print("Building data loader with {} workers".format(num_workers))
#     ds = []
#     if train:
#         train_loader = torch.utils.data.DataLoader(
#             datasets.ImageFolder(
#                 root=os.path.join(data_root, 'train'),
#                 transform=transforms.Compose([
#                     transforms.RandomResizedCrop(224),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                 ])),
#             batch_size=batch_size, shuffle=True, **kwargs)
#         print("Training data size: {}".format(len(train_loader.dataset)))
#         ds.append(train_loader)
#     if val:
#         test_loader = torch.utils.data.DataLoader(
#             datasets.ImageFolder(
#                 root=os.path.join('/data6/peiwang/datasets/imagenet', 'val'),
#                 transform=transforms.Compose([
#                     transforms.Resize(256),
#                     transforms.CenterCrop(224),
#                     transforms.ToTensor(),
#                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                 ])),
#             batch_size=batch_size, shuffle=False, **kwargs)
#         print("Testing data size: {}".format(len(test_loader.dataset)))
#         ds.append(test_loader)
#     ds = ds[0] if len(ds) == 1 else ds
#     return ds

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_loader_mnist(path):
    return Image.open(path).convert('L')

# def default_loader(path):
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel, imindex = line.strip().split()
            imlist.append((impath, int(imlabel), int(imindex)))

    return imlist


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# def default_flist_reader(flist):
#     """
#     flist format: impath label\nimpath label\n ...(same to caffe's filelist)
#     """
#     imlist = []
#     with open(flist, 'r') as rf:
#         for line in rf.readlines():
#             impath, imlabel = line.strip().split()
#             imlist.append((impath, int(imlabel)))
#
#     return imlist

# class ImageFilelist(data.Dataset):
#     def __init__(self, flist, transform=None, target_transform=None,
#                  flist_reader=default_flist_reader, loader=default_loader):
#         self.imlist = flist_reader(flist)
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = loader
#
#     def __getitem__(self, index):
#         impath, target = self.imlist[index]
#         img = self.loader(impath)
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target
#
#     def __len__(self):
#         return len(self.imlist)

class ImageFilelist(data.Dataset):
    def __init__(self, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target, index = self.imlist[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imlist)


class ImageFilelist_MNIST(data.Dataset):
    def __init__(self, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader_mnist):
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target, index = self.imlist[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imlist)


def imagenet(batch_size, train=True, val=True, **kwargs):

    train_list = '/data6/peiwang/projects/end2end_realistic_predictors/img_label_with_idx_tr.txt'
    val_list = '/data6/peiwang/projects/end2end_realistic_predictors/img_label_with_idx_val.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def indoor(batch_size, train=True, val=True, **kwargs):

    train_list = './indoor/gt_with_idx_tr.txt'
    val_list = './indoor/gt_with_idx_val.txt'
    # val_list = './indoor/gt_with_idx_val_debug.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def cars196(batch_size, train=True, val=True, **kwargs):

    train_list = '/data6/peiwang/datasets/cars196/cars196_gt_tr.txt'
    val_list = '/data6/peiwang/datasets/cars196/cars196_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def ade(batch_size, train=True, val=True, **kwargs):

    train_list = '/data6/peiwang/datasets/ADEChallengeData2016/ADE_gt_tr.txt'
    val_list = '/data6/peiwang/datasets/ADEChallengeData2016/ADE_gt_val.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)


    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def adehard(batch_size, train=True, val=True, **kwargs):

    train_list = '/data6/peiwang/datasets/ADEChallengeData2016/ADE_gt_tr.txt'
    val_list = '/data6/peiwang/datasets/ADEChallengeData2016/ADEhard_gt_val.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)


    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds



def cub200(batch_size, train=True, val=True, **kwargs):

    train_list = '/data6/peiwang/datasets/CUB_200_2011/CUB200_gt_tr.txt'
    val_list = '/data6/peiwang/datasets/CUB_200_2011/CUB200_gt_te.txt'
    # val_list = '/data6/peiwang/datasets/CUB_200_2011/multibirds_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def cub200hard(batch_size, train=True, val=True, **kwargs):

    train_list = '/data6/peiwang/datasets/CUB_200_2011/CUB200hard_gt_te.txt'
    val_list = '/data6/peiwang/datasets/CUB_200_2011/CUB200hard_gt_te.txt'
    # val_list = '/data6/peiwang/datasets/CUB_200_2011/multibirds_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds




def inaturalist(batch_size, train=True, val=True, **kwargs):

    train_list = './inaturalist/gt_with_idx_tr2.txt'
    val_list = './inaturalist/gt_with_idx_val2.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def inaturalist_supercategory(batch_size, train=True, val=True, **kwargs):

    train_list = './inaturalist/gt_with_idx_tr_supercategory.txt'
    val_list = './inaturalist/gt_with_idx_val_supercategory.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def inaturalist_kingdom(batch_size, train=True, val=True, **kwargs):

    train_list = './inaturalist/gt_with_idx_tr_kingdom.txt'
    val_list = './inaturalist/gt_with_idx_val_kingdom.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def inaturalist_phylum(batch_size, train=True, val=True, **kwargs):

    train_list = './inaturalist/gt_with_idx_tr_phylum.txt'
    val_list = './inaturalist/gt_with_idx_val_phylum.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def inaturalist_class(batch_size, train=True, val=True, **kwargs):

    train_list = './inaturalist/gt_with_idx_tr_class.txt'
    val_list = './inaturalist/gt_with_idx_val_class.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def inaturalist_order(batch_size, train=True, val=True, **kwargs):

    train_list = './inaturalist/gt_with_idx_tr_order.txt'
    val_list = './inaturalist/gt_with_idx_val_order.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def inaturalist_family(batch_size, train=True, val=True, **kwargs):

    train_list = './inaturalist/gt_with_idx_tr_family.txt'
    val_list = './inaturalist/gt_with_idx_val_family.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def inaturalist_genus(batch_size, train=True, val=True, **kwargs):

    train_list = './inaturalist/gt_with_idx_tr_genus.txt'
    val_list = './inaturalist/gt_with_idx_val_genus.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def inaturalist_species(batch_size, train=True, val=True, **kwargs):

    train_list = './inaturalist/gt_with_idx_tr_species.txt'
    val_list = './inaturalist/gt_with_idx_val_species.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds
