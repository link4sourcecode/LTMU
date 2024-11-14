import os
import torch
import copy
import time
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F


def binary_search(model, x_0, x_random, adv_label, tol=0.00003, feat=False):
    adv = x_random
    cln = x_0

    while True:

        mid = (cln + adv) / 2.0

        if not is_adversarial(model, mid, adv_label, feat):
            adv = mid
        else:
            cln = mid

        norm = torch.norm(adv - cln).cpu().numpy()
        if norm < tol:
            break

    return cln


def binary_search_mid(model, x_0, x_random, adv_label, tol=0.00003, arch=32):
    adv = x_random
    cln = x_0

    while True:
        mid = (cln + adv) / 2.0
        if not is_adversarial_mid(model, mid, adv_label, arch=arch):
            adv = mid
        else:
            cln = mid
        norm = torch.norm(adv - cln).cpu().numpy()
        if norm < tol:
            break
    return cln


def is_adversarial(model, perturbed, adv_label, feat=False):
    perturbed = perturbed.unsqueeze(0)
    if (feat):
        predict_label = torch.argmax(model.fc(perturbed))
    else:
        predict_label = torch.argmax(model(perturbed))
    target_label = torch.tensor(adv_label).cuda()
    return predict_label == target_label


def is_adversarial_mid(model, perturbed, adv_label, arch=32):
    perturbed = perturbed.unsqueeze(0)
    if arch == 32:
        x = model.layer3(perturbed.cuda())
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
    else:
        x = model.conv5_x(perturbed.cuda())
        x = model.avg_pool(x)
        x = x.view(x.size(0), -1)
    score = model.fc(x)
    predict_label = torch.argmax(score)
    target_label = torch.tensor(adv_label).cuda()
    return predict_label == target_label


def compute_distance(logit, center):
    logit = logit.unsqueeze(1)
    center = center.unsqueeze(0)
    return torch.sqrt(torch.sum((logit - center) ** 2, dim=2))


def compute_distance_mid(tensor_data, centroids):
    # 计算每个张量到每个质心的距离
    num_tensors = tensor_data.size(0)
    num_centroids = centroids.size(0)

    distance_matrix = torch.zeros((num_tensors, num_centroids))

    for i in range(num_tensors):
        tensor_i = tensor_data[i]
        for j in range(num_centroids):
            centroid_j = centroids[j]
            distance = torch.sqrt(((tensor_i - centroid_j) ** 2).sum())
            distance_matrix[i, j] = distance
    return distance_matrix


def LTMU_input(ori_model, train_forget_loader, k, logger, cfg):
    test_model = copy.deepcopy(ori_model).cuda()
    unlearn_model = copy.deepcopy(ori_model).cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, unlearn_model.parameters()), lr=0.01, momentum=0.9)
    centroid = torch.load(f'centroid_{cfg.dataset}.pt').cuda()
    start = time.perf_counter()
    result_indices = []
    feat_x = torch.tensor([]).cuda()
    with torch.no_grad():
        for x, y in train_forget_loader:
            x = x.cuda()
            y = y.cuda()
            test_model.eval()
            logits = test_model(x, feat=True)
            feat_x = torch.cat((feat_x, logits), 0)
            distance_matrix = compute_distance(logits, centroid)
            # min_indices = torch.argmin(distance_matrix, dim=1)
            min_indices = torch.argsort(distance_matrix, dim=1)
            # 遍历每行
            for i in range(min_indices.size(0)):
                # 获取当前行的索引和对应的值
                row_indices = min_indices[i]
                y_value = y[i]

                # 找到当前行第一个不是 y 的元素的索引
                non_y_indices = torch.nonzero(row_indices != y_value).squeeze()
                result_indices.append(row_indices[non_y_indices[:k]])
    result_indices = torch.stack(result_indices, 0).cuda()
    feat_time = time.perf_counter()
    logger.info('feat extract time {}'.format(feat_time - start))
    # feat_x = torch.stack(feat_x).squeeze(0)
    # feat_x = feat_x.view(-1, 2048)
    loader = generate_loader(test_model, feat_x, result_indices, centroid, n=5)
    logger.info('feat generate time {}'.format(time.perf_counter() - feat_time))
    for x, y in loader:
        x = x.cuda()
        y = y.cuda()
        unlearn_model.train()
        unlearn_model.zero_grad()
        optimizer.zero_grad()
        ori_logits = unlearn_model.fc_cb(x)
        ori_loss = criterion(ori_logits, y)
        loss = ori_loss
        loss.backward()
        optimizer.step()
    end = time.perf_counter()
    tm = end - start
    logger.info('BU Time Consuming: {} secs'.format(tm))

    return unlearn_model


def LTMU(ori_model, train_forget_loader, k, logger, cfg):
    test_model = copy.deepcopy(ori_model).cuda()
    unlearn_model = copy.deepcopy(ori_model).cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=0.0005)
    centroid = torch.load(f'centroid_mid_{cfg.dataset}.pt').cuda()
    start = time.perf_counter()
    result_indices = []
    feat_x = torch.tensor([]).cuda()
    with torch.no_grad():
        for x, y in train_forget_loader:
            x = x.cuda()
            y = y.cuda()
            test_model.eval()
            logits = test_model(x, decode=True)
            feat_x = torch.cat((feat_x, logits), 0)
            distance_matrix = compute_distance_mid(logits, centroid)
            min_indices = torch.argsort(distance_matrix, dim=1)
            for i in range(min_indices.size(0)):
                row_indices = min_indices[i]
                y_value = y[i].cpu()
                non_y_indices = torch.nonzero(row_indices != y_value).squeeze()
                result_indices.append(row_indices[non_y_indices[:k]])
            if len(result_indices) > 150:
                break
    result_indices = torch.stack(result_indices, 0).cuda()
    feat_time = time.perf_counter()
    logger.info('feat extract time {}'.format(feat_time - start))
    loader = generate_loader_mid(test_model, feat_x, result_indices, centroid, n=5, arch=32 if cfg.arch=='resnet32' else 18)
    logger.info('feat generate time {}'.format(time.perf_counter() - feat_time))
    for _ in range(10):
        for x, y in loader:
            x = x.cuda()
            y = y.cuda()
            unlearn_model.train()
            unlearn_model.zero_grad()
            optimizer.zero_grad()
            x = unlearn_model.conv5_x(x.cuda())
            x = unlearn_model.avg_pool(x)
            x = x.view(x.size(0), -1)
            score = unlearn_model.fc(x)
            ori_loss = criterion(score, y)
            loss = ori_loss
            loss.backward()
            optimizer.step()
    end = time.perf_counter()
    tm = end - start
    logger.info('BU Time Consuming: {} secs'.format(tm))

    return unlearn_model

eye = np.eye(256) * 0.001
beta = torch.tensor(np.random.multivariate_normal(np.zeros(256), eye, )).float().cuda()

def generate_loader(model, feat_x, top_k, centroid, n=5):
    dataset_x = []
    dataset_y = []
    assert len(feat_x) == len(top_k)
    step = 1 / n
    for i in range(len(feat_x)):
        x = feat_x[i]
        k = top_k[i]
        for cls in k:
            boundary_sample = binary_search(model, x, centroid[cls], cls, feat=True)
            # norm = torch.norm(x - boundary_sample).cpu().numpy()
            for lamb in range(n):
                new_x = x + lamb * step * (boundary_sample - x) + beta
                dataset_x.append(new_x)
                dataset_y.append(cls.item())
    loader = torch.utils.data.DataLoader(DatasetXY(dataset_x, dataset_y),
                                         batch_size=16, shuffle=True, drop_last=True)
    return loader


def generate_loader_mid(model, feat_x, top_k, centroid, n=5, arch=32):
    dataset_x = []
    dataset_y = []
    assert len(feat_x) == len(top_k)
    step = 1 / n
    for i in range(len(feat_x)):
        x = feat_x[i]
        k = top_k[i]
        for cls in k:
            boundary_sample = binary_search_mid(model, x, centroid[cls], cls, arch=arch)
            for lamb in range(n):
                new_x = x + lamb * step * (boundary_sample - x)
                dataset_x.append(new_x)
                dataset_y.append(cls.item())
    loader = torch.utils.data.DataLoader(DatasetXY(dataset_x, dataset_y),
                                         batch_size=16, shuffle=False, drop_last=True)
    return loader


def find_center(model, loader, forget_loader, num_classes, cfg):
    centroid_path = f'centroid_{cfg.dataset}.pt'
    if os.path.exists(centroid_path):
        center = torch.load(centroid_path)
        return center
    feat_sum = [[] for _ in range(num_classes)]
    # model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(loader):
            data = torch.tensor(data).cuda()
            output = model(data, decode=True)
            feat = output
            for i, (f, pid) in enumerate(zip(feat, target)):
                f = f.cpu()
                # feat_sum[pid].add_(f)
                feat_sum[pid].append(f)
    with torch.no_grad():
        for _, (data, target) in enumerate(forget_loader):
            data = torch.tensor(data).cuda()
            output = model(data, decode=True)
            feat = output
            for i, (f, pid) in enumerate(zip(feat, target)):
                f = f.cpu()
                # feat_sum[pid].add_(f)
                feat_sum[pid].append(f)
    center = torch.zeros([num_classes, feat_sum[0][0].size(0)])
    for i in range(num_classes):
        center[i] = torch.sum(torch.stack(feat_sum[i]), dim=0) / len(feat_sum[i])
    # forget_center = center[forget_class].unsqueeze(0)
    torch.save(center, centroid_path)
    return center


def find_mid_center(model, loader, forget_loader, num_classes, cfg):
    centroid_path = f'centroid_mid_{cfg.dataset}.pt'
    if os.path.exists(centroid_path):
        center = torch.load(centroid_path)
        return center
    feat_sum = [[] for _ in range(num_classes)]
    model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(loader):
            data = data.cuda()
            output = model(data, decode=True)
            feat = output
            for i, (f, pid) in enumerate(zip(feat, target)):
                f = f.cpu()
                # feat_sum[pid].add_(f)
                feat_sum[pid].append(f)
    with torch.no_grad():
        for _, (data, target) in enumerate(forget_loader):
            data = data.cuda()
            output = model(data, decode=True)
            feat = output
            for i, (f, pid) in enumerate(zip(feat, target)):
                f = f.cpu()
                # feat_sum[pid].add_(f)
                feat_sum[pid].append(f)
    center = torch.zeros([num_classes, feat_sum[0][0].size(0), feat_sum[0][0].size(1), feat_sum[0][0].size(2)])
    for i in range(num_classes):
        center[i] = torch.sum(torch.stack(feat_sum[i]), dim=0)/len(feat_sum[i])
    # forget_center = center[forget_class].unsqueeze(0)
    torch.save(center, centroid_path)
    return center


class DatasetXY(Dataset):
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._len = len(x)

    def __getitem__(self, item):  # 每次循环的时候返回的值
        return self._x[item], self._y[item]

    def __len__(self):
        return self._len

