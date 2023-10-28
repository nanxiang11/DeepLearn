import hashlib
import os
import tarfile
import zipfile

import d2l.torch
import requests

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from d2l import torch as d2l

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(name, cache_dir=os.path.join("..", 'data')):
    """
    下载一个DATA_HUB中的文件，返回本地文件名
    :param name:
    :param cache_dir:
    :return:
    """
    assert name in DATA_HUB, f"{name}不存在于{DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'正在从{url}下载{fname}')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    """
    下载并且解压zip,tar文件
    :param name:
    :param folder:
    :return:
    """
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件开源被压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
    """
    下载DATA_HUB中的所有文件
    :return:
    """
    for name in DATA_HUB:
        download(name)


DATA_HUB['kaggle_house_train'] = (  # @save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  # @save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

# 获取数据
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))


# print(train_data.shape, test_data.shape)
# (1460, 81) (1459, 80)


# 数据处理并转化为张量
def data_processing(train_data, test_data):
    # 收集所有的特征，统一处理
    all_feature = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    # 对数字数据进行标准化
    number_features = all_feature.dtypes[all_feature.dtypes != 'object'].index
    all_feature[number_features] = all_feature[number_features].apply(
        lambda x: (x - x.mean()) / (x.std())
    )
    # 将缺失值设置为0
    all_feature[number_features] = all_feature[number_features].fillna(0)
    # 热独编码
    all_feature = pd.get_dummies(all_feature, dummy_na=True)
    n_train = train_data.shape[0]

    train_features = torch.tensor(all_feature[:n_train].values.astype(float), dtype=torch.float32)
    test_features = torch.tensor(all_feature[n_train:].values.astype(float), dtype=torch.float32)
    train_labels = torch.tensor(
        train_data.SalePrice.values.astype(float).reshape(-1, 1), dtype=torch.float32
    )
    return train_features, test_features, train_labels


train_features, test_features, train_labels = data_processing(train_data, test_data)


# print(train_features[:20, :20])


def log_rmse(net, features, labels):
    """
    用相对误差，更好体现价格上的关系|y-y1/y|
    :param net:
    :param features:
    :param labels:
    :return:
    """
    loss = nn.MSELoss()
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def test_SGD(num_epoch):
    loss = nn.MSELoss()
    in_feature = train_features.shape[1]
    net = nn.Sequential(nn.Linear(in_feature, 1))
    net.apply(init_weights)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls = []
    pre = []
    dataSet = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(dataSet, 128, shuffle=True)

    for epoch in range(num_epoch):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
    d2l.plt.figure(figsize=(10, 6))
    d2l.plot(list(range(1, len(train_ls) + 1)), train_ls, 'epochs', 'rmse',
             legend=['train_loss'])
    d2l.plt.title("SGD")
    d2l.plt.show()

    # d2l.plt.figure(figsize=(10, 6))
    # pre = net(train_features).detach().numpy()
    # d2l.plt.scatter(range(1, 1461), train_labels, label="true")
    # d2l.plt.scatter(range(1, 1461), pre, label="pre")
    # d2l.plt.xlabel("sample")
    # d2l.plt.ylabel("price")
    # d2l.plt.legend()
    # d2l.plt.show()


def test_GD(num_epoch, learning_rate):
    loss = nn.MSELoss()
    in_feature = train_features.shape[1]
    net = nn.Sequential(nn.Linear(in_feature, 1))
    net.apply(init_weights)
    train_ls = []
    pre = []
    dataSet = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(dataSet, 128, shuffle=True)

    for epoch in range(num_epoch):
        for X, y in train_iter:
            net.zero_grad()
            l = loss(net(X), y)
            l.backward()
            # 更新参数
            with torch.no_grad():
                for param in net.parameters():
                    param -= learning_rate * param.grad

        train_ls.append(log_rmse(net, train_features, train_labels))
    d2l.plt.figure(figsize=(10, 6))
    d2l.plot(list(range(1, len(train_ls) + 1)), train_ls, 'epochs', 'rmse',
             legend=['train_loss'])
    d2l.plt.title("GD")
    d2l.plt.show()

# test_SGD(100)
# test_GD(100, 0.01)


def test_MLP(num_epoch):
    loss = nn.MSELoss()
    in_feature = train_features.shape[1]
    net = nn.Sequential(
        nn.Linear(in_feature, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls = []
    dataSet = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(dataSet, 64, shuffle=True)

    # 设置梯度裁剪的阈值
    max_grad_norm = 1.0

    for epoch in range(num_epoch):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()

            # 执行梯度裁剪
            clip_grad_norm_(net.parameters(), max_grad_norm)

            trainer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
    d2l.plt.figure(figsize=(10, 6))
    d2l.plot(list(range(1, len(train_ls) + 1)), train_ls, 'epochs', 'rmse',
             legend=['train_loss'])
    d2l.plt.title("MLP")
    d2l.plt.show()

# test_MLP(100)


def test_Adam(num_epoch):
    loss = nn.MSELoss()
    in_feature = train_features.shape[1]
    net = nn.Sequential(nn.Linear(in_feature, 1))
    trainer = torch.optim.Adam(
        net.parameters(),
        lr=0.01,
        weight_decay=0
    )
    train_ls = []
    pre = []
    dataSet = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(dataSet, 64, shuffle=True)

    for epoch in range(num_epoch):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
    d2l.plt.figure(figsize=(10, 6))
    d2l.plot(list(range(1, len(train_ls) + 1)), train_ls, 'epochs', 'rmse',
             legend=['train_loss'])
    d2l.plt.title("Adam")
    d2l.plt.show()


# test_Adam(100)


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    loss = nn.MSELoss()
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    # 设置梯度裁剪的阈值
    max_grad_norm = 1.0
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            # 执行梯度裁剪
            clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size, title):
    train_l_sum, valid_l_sum = 0, 0
    in_feature = train_features.shape[1]
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = nn.Sequential(
            nn.Linear(in_feature, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        net.apply(init_weights)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == k - 1:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
            d2l.plt.title(title)
            d2l.plt.show()
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


def train_k_v():
    k, num_epochs, lr, weight_decay, batch_size = 6, 100, 0.005, 0, 64
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                              weight_decay, batch_size, "adam_1MLP")
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
          f'平均验证log rmse: {float(valid_l):f}')


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    in_feature = train_features.shape[1]
    net = nn.Sequential(
        nn.Linear(in_feature, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    net.apply(init_weights)
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    d2l.plt.show()
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


num_epochs, lr, weight_decay, batch_size = 100, 0.005, 0, 64
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
