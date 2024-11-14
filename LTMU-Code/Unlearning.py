import copy
import random
import sys
import argparse
from torch.backends import cudnn

from unlearn_contrast import select_method
from utils import util, Trainer
from utils.util import *
from model import resnet
from imbalance_data import cifar10Imbanlance, cifar100Imbanlance
import logging
from LTMU import LTMU_input, find_center, find_mid_center, LTMU

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
        forgetset = cifar10Imbanlance.Cifar10Imbanlance_forget(transform=transform_train, forget=True,
                                                               imbanlance_rate=args.imbanlance_rate, train=True,
                                                               file_path=args.root, forget_class=args.forget_class,
                                                               input_aug=True if args.unlearn_method == 'GA_IA' or 'BS_IA' else False)
        testset = cifar10Imbanlance.Cifar10Imbanlance_forget(imbanlance_rate=args.imbanlance_rate, train=False,
                                                             transform=transform_val, file_path=args.root,
                                                             forget_class=args.forget_class)
        forgettest = cifar10Imbanlance.Cifar10Imbanlance_forget(imbanlance_rate=args.imbanlance_rate, train=False,
                                                                transform=transform_val, file_path=args.root,
                                                                forget_class=args.forget_class, forget=True)
        print("load cifar10")
        return trainset, testset, forgetset, forgettest

    if args.dataset == 'cifar100' and args.forget_class is not None:
        trainset = cifar100Imbanlance.Cifar100Imbanlance_forget(transform=transform_train,
                                                         imbanlance_rate=args.imbanlance_rate, train=True,
                                                         file_path=os.path.join(args.root, 'cifar-100-python/'),
                                                                forget_class=args.forget_class)
        forgetset = cifar100Imbanlance.Cifar100Imbanlance_forget(transform=transform_train, forget=True,
                                                                 imbanlance_rate=args.imbanlance_rate, train=True,
                                                                 file_path=os.path.join(args.root, 'cifar-100-python/'),
                                                                 forget_class=args.forget_class,
                                                                 input_aug=True if args.unlearn_method == 'GA_IA' or 'BS_IA' else False)
        testset = cifar100Imbanlance.Cifar100Imbanlance_forget(imbanlance_rate=args.imbanlance_rate, train=False,
                                                        transform=transform_val,
                                                        file_path=os.path.join(args.root, 'cifar-100-python/'),
                                                               forget_class=args.forget_class)
        forgettest = cifar100Imbanlance.Cifar100Imbanlance_forget(imbanlance_rate=args.imbanlance_rate, train=False,
                                                                  transform=transform_val, forget=True,
                                                                  file_path=os.path.join(args.root,
                                                                                         'cifar-100-python/'),
                                                                  forget_class=args.forget_class)
        print("load cifar100")
        return trainset, testset, forgetset, forgettest


def main():
    args = parser.parse_args()
    if args.ori_path:
        args.ori_path = os.path.join(args.ori_path, 'ckpt.best.pth.tar')
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

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join('./output/logs/', 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)

    model = get_model(args)
    _ = print_model_param_nums(model=model)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    logger.info("forget class: {}".format(args.forget_class))
    logger.info('args.seed =  %d' % args.seed)
    if args.ori_path:
        print("=> loading checkpoint '{}'".format(args.store_name))
        checkpoint = torch.load(args.ori_path, map_location='cuda:0')
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.ori_path, checkpoint['epoch']))

    # Data loading code
    train_dataset, val_dataset, forget_dataset, forgettest_dataset = get_dataset(args)
    num_classes = len(np.unique(train_dataset.targets)) + len(np.unique(forget_dataset.targets))
    assert num_classes == args.num_classes

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers,
                                               persistent_workers=True, pin_memory=True)
    forget_loader = torch.utils.data.DataLoader(forget_dataset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.workers,
                                                persistent_workers=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, persistent_workers=True, pin_memory=True)
    forgettest_loader = torch.utils.data.DataLoader(forgettest_dataset, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.workers, persistent_workers=True, pin_memory=True)


    unlearn_model = copy.deepcopy(model).cuda()
    start_time = time.time()
    print("Training started!")
    if args.unlearn_method == 'LTMU':
        find_mid_center(unlearn_model, train_loader, forget_loader, num_classes, args)
        logger.info('data augment and center augment time: {}'.format(time.time() - start_time))
        unlearn_model = LTMU(unlearn_model, forget_loader, args.k, logger, args)
    elif args.unlearn_method == 'LTMU_input':
        find_center(unlearn_model, train_loader, forget_loader, num_classes, args)
        unlearn_model = LTMU_input(unlearn_model, forget_loader, args.k, logger, args)
    elif args.unlearn_method == 'BT':
        unlearn_teacher = get_model(args).cuda()
        unlearn_model = select_method(args.unlearn_method, unlearn_model, teacher=unlearn_teacher,
                                      forget_loader=forget_loader, train_loader=train_loader,)
    else:
        unlearn_model = select_method(args.unlearn_method, unlearn_model, forget_loader=forget_loader,
                                      train_loader=train_loader, forget_cls=args.forget_class,
                                      num_classes=args.num_classes)
    end_time = time.time()
    logger.info("It took {} to execute the program".format(hms_string(end_time - start_time)))
    Trainer.validate(unlearn_model, train_loader, logger)
    Trainer.validate(unlearn_model, forget_loader, logger)
    Trainer.validate(unlearn_model, val_loader, logger)
    Trainer.validate(unlearn_model, forgettest_loader, logger)


if __name__ == '__main__':
    # train set
    parser = argparse.ArgumentParser(description="Deep Long-tailed Unlearning Unlearning Stage")
    parser.add_argument('--dataset', type=str, default='cifar100', help="cifar10,cifar100")
    parser.add_argument('--root', type=str, default='./data/', help="dataset setting")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                        choices=('resnet18', 'resnet34', 'resnet50'))
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes ')
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--seed', default=3407, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--ori_path', type=str, metavar='PATH',
                        help='path to original model (default: none)')
    parser.add_argument('--root_log', type=str, default='./output/')
    parser.add_argument('--root_model', type=str, default='./output/')
    parser.add_argument('--store_name', type=str, default='./output/')
    parser.add_argument('--unlearn_method', type=str, default='LTMU',
                        choices=['LTMU', 'LTMU_input', 'BS', 'GA', 'RL', 'BT', 'FT', 'L1', 'GA_IA', 'GA_RFA', 'BS_IA',
                                 'BS_RFA'])
    parser.add_argument('--forget_class', default=None, type=int, help='Class to forget.')
    parser.add_argument('--k', default=8, type=int, help='Number of selected class.')
    main()
