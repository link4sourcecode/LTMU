import copy
import time
import tqdm
from torch.utils.data import DataLoader, Dataset
from unlearn_contrast.adv_generator import LinfPGD, inf_generator, FGSM
import pruner
import torch
import random
from torch.nn import functional as F
import utils.util as utils
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
norm = False

def select_method(method, model, forget_loader=None, train_loader=None, forget_cls=None, num_classes=None, epoch=10,
                  teacher=None, lr=0.0001):
    if method == 'GA' or 'GA_IA':
        for ep in range(epoch):
            model = gradient_ascent(model, forget_loader)
        return model
    elif method == 'RL':
        for ep in range(epoch):
            model = random_label(model, forget_loader, forget_cls, num_classes)
        return model
    elif method == 'L1':
        for ep in range(epoch):
            model = FT_prune(train_loader, model, ep)
        return model
    elif method == 'FT':
        for ep in range(epoch):
            model = FT_clean(train_loader, model, ep)
        return model
    elif method == 'BS' or 'BS_IA':
        return boundary_shrink(model, forget_loader)
    elif method == 'BT':
        return blindspot_unlearner(model=model, unlearning_teacher=teacher, unlearning_loader=forget_loader, lr=lr,
                                   retain_loader=train_loader, epochs=epoch)
    elif method == 'GA_RFA':
        aug_feat, _, _ = feature_augment_single_class(forget_loader, model,
                                                                         forget_cls)
        return GA_feat(model, aug_feat, forget_cls)
    elif method == 'BS_RFA':
        aug_feat, _, g_label = feature_augment_single_class(forget_loader, model,
                                                                         forget_cls)
        return unlearn_feat(model, aug_feat, g_label)
    else:
        raise ValueError('Invalid method')

def feature_augment_single_class(loader, model, cls, feat_num_aug=False):
    embedding_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            y = y.cuda()
            feat = model(x, decode=True)
            embedding_list.append(feat.cpu())
            label_list.append(y.cpu())
    cov_list = list()
    for feat_list in embedding_list:
        mean = torch.mean(feat_list, dim=(2, 3))
        cov_list.append(torch.tensor(np.cov(mean.numpy().T)))
    cov_list = torch.stack(cov_list, dim=0).mean(0)

    if len(embedding_list) == 1:
        feat_exp = embedding_list[0].repeat((64//len(embedding_list[0]) + 1), 1, 1, 1)[:64]
        embedding_list[0] = feat_exp[: 32]
        embedding_list.append(feat_exp[32:])
    elif len(embedding_list) == 2:
        feat_exp = embedding_list[1].repeat((32//len(embedding_list[1]) + 1), 1, 1, 1)[: 32]
        embedding_list[1] = feat_exp
    else:
        embedding_list = embedding_list[: 2]

    lamb = 0.01
    eye = np.eye(256) * lamb
    aug_feat = []
    aug_final_feat = []
    g_label = []
    for feat_list in embedding_list:
        length = len(feat_list)

        alpha = torch.tensor(np.random.multivariate_normal(np.ones(256), eye, (length, 8, 8))).permute(0, 3, 1, 2)
        beta = torch.tensor(np.random.multivariate_normal(np.zeros(256), eye, (length, 8, 8))).permute(0, 3, 1, 2)
        eps = torch.tensor(np.random.multivariate_normal(np.zeros(256), cov_list * lamb, (length, 8, 8))).permute(0, 3, 1,
                                                                                                                2)
        feat_list = torch.mul(alpha, feat_list) + beta
        feat_list = feat_list + eps
        x = model.conv5_x(feat_list.float().cuda())
        x = model.avg_pool(x)
        x = x.view(x.size(0), -1)
        score = model.fc(x)
        argsort = torch.argsort(score, dim=1, descending=True)
        idx = torch.argmax((argsort != cls).float(), dim=1)
        label = argsort[torch.arange(argsort.size(0)), idx]
        aug_feat.append(feat_list.float())
        aug_final_feat.append(x)
        g_label.append(label.tolist())
    return aug_feat, aug_final_feat, g_label


def unlearn_feat(model, feat_list, label_list):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)
    for _ in range(10):
        for x, y in zip(feat_list, label_list):
            y = torch.tensor(y).cuda()
            x = model.avg_pool(model.conv5_x(x.cuda()))
            x = x.view(x.size(0), -1)
            score = model.fc(x)
            loss = F.cross_entropy(score, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def GA_feat(model, feat_list, forget_class):
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)
    for _ in range(10):
        for x in feat_list:
            y = torch.tensor([forget_class] * len(x)).cuda()
            y = torch.tensor(y).cuda()
            x = model.avg_pool(model.conv5_x(x.cuda()))
            x = x.view(x.size(0), -1)
            score = model.fc(x)
            loss = - F.cross_entropy(score, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def boundary_shrink(ori_model, train_forget_loader, bound=1.0, step=1/255, iter=10, epoch=10, loss_func=torch.nn.CrossEntropyLoss(), mode='FGSM'):
    poison_epoch = epoch
    test_model = copy.deepcopy(ori_model).cuda()
    unlearn_model = copy.deepcopy(ori_model).cuda()
    random_start = False if mode != "PGD" else True
    if mode == 'PGD':
        adv = LinfPGD(test_model, bound, step, iter, norm, random_start,
                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        adv = FGSM(test_model, bound, norm, random_start,
                   device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    forget_data_gen = inf_generator(train_forget_loader)
    batches_per_epoch = len(train_forget_loader)

    criterion = loss_func
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=0.0005, momentum=0.9)

    num_hits = 0
    num_sum = 0
    nearest_label = []

    start = time.perf_counter()
    cnt = 0
    for itr in tqdm.tqdm(range(poison_epoch * batches_per_epoch)):

        x, y = forget_data_gen.__next__()
        x = x.cuda()
        y = y.cuda()
        test_model.eval()
        x_adv = adv.perturb(x, y, target_y=None, model=test_model)
        adv_logits = test_model(x_adv)

        pred_label = torch.argmax(adv_logits, dim=1)
        if itr >= (poison_epoch - 1) * batches_per_epoch:
            nearest_label.append(pred_label.tolist())
        num_hits += (y != pred_label).float().sum()
        num_sum += y.shape[0]

        # adv_train
        unlearn_model.train()
        unlearn_model.zero_grad()
        optimizer.zero_grad()

        ori_logits = unlearn_model(x)
        ori_loss = criterion(ori_logits, pred_label)
        loss = ori_loss  # - KL_div
        loss.backward()
        optimizer.step()
    end = time.perf_counter()
    tm = end - start
    print('Time Consuming:', tm, 'secs')
    print('attack success ratio:', (num_hits / num_sum).float())
    return unlearn_model

def gradient_ascent(model, forget_loader):
    model.train()
    criterion = F.nll_loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005, momentum=0.9)
    for batch_idx, (data, target) in enumerate(forget_loader):
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss = -loss
        loss.backward()
        optimizer.step()
    return model

def random_label(model, forget_loader, forget_cls, num_classes):
    model.train()
    criterion = F.nll_loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
    random_label_list = list(range(num_classes))
    if isinstance(forget_cls, list):
        for cls in forget_cls:
            random_label_list.remove(cls)
    else:
        random_label_list.remove(forget_cls)
    for batch_idx, (data, _) in enumerate(forget_loader):
        data = data.cuda()
        random_label = random.choices(random_label_list, k=data.size(0))
        target = torch.tensor(random_label).cuda()

        optimizer.zero_grad()
        output= model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return model


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def FineTune(
    data_loaders, model, criterion, optimizer, epoch, with_l1=False
):
    alpha = 0.005
    losses = utils.AverageMeter("FineTune")
    top1 = utils.AverageMeter("FineTune")

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(data_loaders):
        image = image.cuda()
        target = target.cuda()
        if epoch-1 < 10:
            current_alpha = alpha * (
                1 - (epoch-1) / 10
            )
        else:
            current_alpha = 0
        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)
        if with_l1:
            loss += current_alpha * l1_regularization(model)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % 20 == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, len(data_loaders), end - start, loss=losses, top1=top1
                )
            )
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return model


def FT_prune(data_loaders, model, epoch):
    optimizer = torch.optim.SGD(model.parameters(), 0.00005,
                                momentum=0.9,
                                weight_decay=5e-8)
    criterion = torch.nn.CrossEntropyLoss()
    # save checkpoint
    initialization = copy.deepcopy(model.state_dict())

    # unlearn
    model = FineTune(data_loaders, model, criterion, optimizer, epoch, with_l1=True)

    # val
    pruner.check_sparsity(model)

    return model

def FT_clean(data_loaders, model, epoch):
    optimizer = torch.optim.SGD(model.parameters(), 0.001,
                                momentum=0.9,
                                weight_decay=5e-6)
    criterion = torch.nn.CrossEntropyLoss()
    # unlearn
    model = FineTune(data_loaders, model, criterion, optimizer, epoch)
    return model


class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if (index < self.forget_len):
            x = self.forget_data[index][0]
            y = 1
            return x, y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x, y


def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    labels = torch.unsqueeze(labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)
    # label 1 means forget sample
    # label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out)


def unlearning_step(model, unlearning_teacher, full_trained_teacher, unlearn_data_loader, optimizer, KL_temperature):
    losses = []
    for batch in unlearn_data_loader:
        x, y = batch
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        model.train()
        output = model(x)
        optimizer.zero_grad()
        loss = UnlearnerLoss(output=output, labels=y, full_teacher_logits=full_teacher_logits,
                             unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=KL_temperature)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())


def blindspot_unlearner(model, unlearning_teacher, unlearning_loader, retain_loader, epochs=10,
                        optimizer='adam', lr=0.01, KL_temperature=1):
    unlearning_teacher.eval()
    full_trained_teacher = copy.deepcopy(model).cuda()
    full_trained_teacher.eval()
    optimizer = optimizer
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        # if optimizer is not a valid string, then assuming it as a function to return optimizer
        optimizer = optimizer  # (model.parameters())

    data_loader_length = len(retain_loader.dataset)
    # 计算需要选择的样本数量，这里选择 0.3 倍的长度
    sample_size = min(int(0.3 * data_loader_length), len(unlearning_loader.dataset) * 10)
    all_indices = list(range(data_loader_length))
    random_indices = random.sample(all_indices, sample_size)
    # 构建一个新的 DataLoader，只包含随机选择的样本
    random_subset_loader = DataLoader(retain_loader.dataset, batch_size=retain_loader.batch_size,
                                      sampler=random_indices)
    unlearn_data = UnLearningData(forget_data=unlearning_loader.dataset, retain_data=random_subset_loader.dataset)
    unlearn_loader = DataLoader(unlearn_data, batch_size=unlearning_loader.batch_size, shuffle=True)

    for epoch in range(epochs):
        unlearning_step(model=model, unlearning_teacher=unlearning_teacher,
                       full_trained_teacher=full_trained_teacher, unlearn_data_loader=unlearn_loader,
                       optimizer=optimizer, KL_temperature=KL_temperature)
    return model
