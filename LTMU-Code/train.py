import random
import sys
import argparse
from torch.backends import cudnn
from utils import util
from utils.util import *
from model import resnet
from imbalance_data import cifar10Imbanlance, cifar100Imbanlance
import logging
from utils.Trainer import Trainer
best_acc1 = 0


def get_model(args):
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'resnet50':
        net = resnet.resnet50(num_class=args.num_classes)
    elif args.arch == 'resnet18':
        net = resnet.resnet18(num_class=args.num_classes)
    elif args.arch == 'resnet34':
        net = resnet.resnet34(num_class=args.num_classes)
    return net


def get_dataset(args):
    transform_train, transform_val = util.get_transform(args.dataset)
    if args.dataset == 'cifar10' and args.forget_class is not None:
        trainset = cifar10Imbanlance.Cifar10Imbanlance_forget(transform=transform_train,
                                                              imbanlance_rate=args.imbanlance_rate, train=True,
                                                              file_path=args.root, forget_class=args.forget_class)
        forgetset = cifar10Imbanlance.Cifar10Imbanlance_forget(transform=transform_train,
                                                               forget=True,
                                                               imbanlance_rate=args.imbanlance_rate, train=True,
                                                               file_path=args.root, forget_class=args.forget_class)
        testset = cifar10Imbanlance.Cifar10Imbanlance_forget(imbanlance_rate=args.imbanlance_rate, train=False,
                                                             transform=transform_val, file_path=args.root,
                                                             forget_class=args.forget_class)
        forgettest = cifar10Imbanlance.Cifar10Imbanlance_forget(imbanlance_rate=args.imbanlance_rate, train=False,
                                                                transform=transform_val, file_path=args.root,
                                                                forget_class=args.forget_class, forget=True)
        print("load cifar10")
        return trainset, testset, forgetset, forgettest

    if args.dataset == 'cifar100' and args.forget_class is not None:
        trainset = cifar100Imbanlance.Cifar100Imbanlance(transform=transform_train,
                                                         imbanlance_rate=args.imbanlance_rate, train=True,
                                                         file_path=os.path.join(args.root, 'cifar-100-python/'))
        forgetset = cifar100Imbanlance.Cifar100Imbanlance_forget(transform=transform_train,
                                                                 imbanlance_rate=args.imbanlance_rate, train=True,
                                                                 file_path=os.path.join(args.root, 'cifar-100-python/'),
                                                                 forget_class=args.forget_class)
        testset = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False,
                                                        transform=transform_val,
                                                        file_path=os.path.join(args.root, 'cifar-100-python/'))
        forgettest = cifar100Imbanlance.Cifar100Imbanlance_forget(imbanlance_rate=args.imbanlance_rate, train=False,
                                                                  transform=transform_val,
                                                                  file_path=os.path.join(args.root,
                                                                                         'cifar-100-python/'),
                                                                  forget_class=args.forget_class)
        print("load cifar100")
        return trainset, testset, forgetset, forgettest

    if args.dataset == 'cifar10':
        trainset = cifar10Imbanlance.Cifar10Imbanlance(transform=util.TwoCropTransform(transform_train),
                                                       imbanlance_rate=args.imbanlance_rate, train=True,
                                                       file_path=args.root)
        testset = cifar10Imbanlance.Cifar10Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False,
                                                      transform=transform_val, file_path=args.root)
        print("load cifar10")
        return trainset, testset

    if args.dataset == 'cifar100':
        trainset = cifar100Imbanlance.Cifar100Imbanlance(transform=util.TwoCropTransform(transform_train),
                                                         imbanlance_rate=args.imbanlance_rate, train=True,
                                                         file_path=os.path.join(args.root, 'cifar-100-python/'))
        testset = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False,
                                                        transform=transform_val,
                                                        file_path=os.path.join(args.root, 'cifar-100-python/'))
        print("load cifar100")
        return trainset, testset


def main():
    args = parser.parse_args()
    print(args)
    args.store_name = '#'.join([args.dataset, args.arch, "imbanlance_rate" + str(args.imbanlance_rate)])
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_acc1
    global train_cls_num_list
    global cls_num_list_cuda

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # create model
    num_classes = args.num_classes
    model = get_model(args)
    _ = print_model_param_nums(model=model)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.root_log + args.store_name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)

    # Data loading code
    if args.forget_class is None:
        train_dataset, val_dataset = get_dataset(args)
    else:
        train_dataset, val_dataset, forget_dataset, forget_val_dataset = get_dataset(args)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers,
                                               persistent_workers=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, persistent_workers=True, pin_memory=True)

    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    if args.forget_class != None:
        cls_num_list[args.forget_class - 1] = 1
    train_cls_num_list = np.array(cls_num_list)

    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()
    start_time = time.time()
    print("Training started!")
    trainer = Trainer(args, model=model, train_loader=train_loader, val_loader=val_loader,
                          per_class_num=train_cls_num_list, log=logging)
    trainer.train()
    end_time = time.time()
    print("It took {} to execute the program".format(hms_string(end_time - start_time)))


if __name__ == '__main__':
    # train set
    parser = argparse.ArgumentParser(description="Deep Long-tailed Unlearning Training Stage")
    parser.add_argument('--dataset', type=str, default='cifar100', help="cifar10,cifar100")
    parser.add_argument('--root', type=str, default='./data/', help="dataset setting")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                        choices=('resnet18', 'resnet34', 'resnet50'))
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes ')
    parser.add_argument('--imbanlance_rate', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate',
                        dest='lr')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=5e-3, type=float, metavar='W',
                        help='weight decay (default: 5e-3、2e-4、1e-4)', dest='weight_decay')
    parser.add_argument('--seed', default=3407, type=int, help='seed for initializing training. ')
    parser.add_argument('-p', '--print_freq', default=1000, type=int, metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--root_log', type=str, default='./output/')
    parser.add_argument('--root_model', type=str, default='./output/')
    parser.add_argument('--store_name', type=str, default='./output/')
    parser.add_argument('--forget_class', default=None, type=int, help='Class to forget.')
    main()
