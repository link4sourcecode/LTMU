import torch
import torch.nn as nn
from utils.util import *
import math
from sklearn.metrics import confusion_matrix


class Trainer(object):
    def __init__(self, args, model=None,train_loader=None, val_loader=None,per_class_num=[],log=None):
        self.args = args
        self.device = args.gpu
        self.print_freq = args.print_freq
        self.lr = args.lr
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.use_cuda = True
        self.num_classes = args.num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cls_num_list = per_class_num
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=self.lr,weight_decay=args.weight_decay)
        self.train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.log = log
        self.beta = args.beta

    def train(self):
        best_acc1 = 0
        for epoch in range(self.start_epoch, self.epochs):
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            criterion = nn.CrossEntropyLoss()
            # switch to train mode
            self.model.train()
            end = time.time()

            for i, (inputs, targets) in enumerate(self.train_loader):
                output = self.model(inputs.cuda())

                loss = criterion(output, targets.cuda())

                losses.update(loss.item(), inputs[0].size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % self.print_freq == 0:
                    output = ('Epoch: [{0}/{1}][{2}/{3}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        epoch + 1, self.epochs, i, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))  # TODO
                    print(output)
                    # evaluate on validation set
            acc1 = self.validate(epoch=epoch)
            if self.args.dataset == 'ImageNet-LT' or self.args.dataset == 'iNaturelist2018':
                self.paco_adjust_learning_rate(epoch)
            else:
                self.train_scheduler.step()
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1,  best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_best)
            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc1':  best_acc1,
            }, is_best, epoch + 1)

    def validate(self,epoch=None):
        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        eps = np.finfo(np.float64).eps

        # switch to evaluate mode
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(self.val_loader):
                input = input.cuda()
                target = target.cuda()

                # compute output
                output = self.model(input)

                # measure accuracy
                acc1 = accuracy(output, target, topk=(1,))
                top1.update(acc1.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                if i % self.print_freq == 0:
                    output = ('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                        i, len(self.val_loader), batch_time=batch_time, top1=top1))
                    print(output)
            cf = confusion_matrix(all_targets, all_preds).astype(float)
            cls_cnt = cf.sum(axis=1)
            cls_hit = np.diag(cf)
            cls_acc = cls_hit / cls_cnt
            output = ('EPOCH: {epoch} {flag} Results: Prec@1 {top1.avg:.3f}'.format(epoch=epoch + 1 , flag='val', top1=top1))
            self.log.info(output)
            if len(cls_acc) != len(self.cls_num_list):
                diff = self.cls_num_list.shape[0] - cls_acc.shape[0]
                if diff > 0:
                    padding = np.zeros(diff)
                    cls_acc = np.concatenate((cls_acc, padding))
            many_shot = self.cls_num_list > 100
            medium_shot = (self.cls_num_list <= 100) & (self.cls_num_list > 20)
            few_shot = self.cls_num_list <= 20
            print("many avg, med avg, few avg",
                  float(sum(cls_acc[many_shot]) * 100 / (sum(many_shot) + eps)),
                  float(sum(cls_acc[medium_shot]) * 100 / (sum(medium_shot) + eps)),
                  float(sum(cls_acc[few_shot]) * 100 / (sum(few_shot) + eps))
                  )
        return top1.avg

    def paco_adjust_learning_rate(self, epoch):
        warmup_epochs = 10
        lr = self.lr
        if epoch <= warmup_epochs:
            lr = self.lr / warmup_epochs * (epoch + 1)
        else:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs + 1) / (self.epochs - warmup_epochs + 1)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def validate(self,epoch=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    eps = np.finfo(np.float64).eps

    # switch to evaluate mode
    self.model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(self.val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = self.model(input)

            # measure accuracy
            acc1 = accuracy(output, target, topk=(1,))
            top1.update(acc1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % self.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    i, len(self.val_loader), batch_time=batch_time, top1=top1))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('EPOCH: {epoch} {flag} Results: Prec@1 {top1.avg:.3f}'.format(epoch=epoch + 1 , flag='val', top1=top1))
        self.log.info(output)
        if len(cls_acc) != len(self.cls_num_list):
            diff = self.cls_num_list.shape[0] - cls_acc.shape[0]
            if diff > 0:
                padding = np.zeros(diff)
                cls_acc = np.concatenate((cls_acc, padding))
        many_shot = self.cls_num_list > 100
        medium_shot = (self.cls_num_list <= 100) & (self.cls_num_list > 20)
        few_shot = self.cls_num_list <= 20
        print("many avg, med avg, few avg",
              float(sum(cls_acc[many_shot]) * 100 / (sum(many_shot) + eps)),
              float(sum(cls_acc[medium_shot]) * 100 / (sum(medium_shot) + eps)),
              float(sum(cls_acc[few_shot]) * 100 / (sum(few_shot) + eps))
              )
    return top1.avg
