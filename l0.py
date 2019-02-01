# coding: utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use("seaborn-white")

import random
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm

from data import PermutedMNIST
from utils import train_l0, l0_train_trans, get_l0_scores_and_model_params, test

epochs = 20
lr = 0.1
batch_size = 128
sample_size = 200
hidden_size = 200
num_task = 3

from models import L0LeNet5
import torchvision
from torchvision import datasets
import numpy as np

import pdb

# rng_permute = np.random.RandomState(92916)
# 
# def get_permute_mnist():
#     train_loader = {}
#     test_loader = {}
#     for i in range(num_task):
#         idx_permute = torch.from_numpy(rng_permute.permutation(784) )
#         transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#                       torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1, 28, 28) ),
#                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
#                       ])
#         train_loader[i] = torch.utils.data.DataLoader(datasets.MNIST(root="~/.torch/data/mnist", train=True, transform=transform),
#                                                       batch_size=batch_size,
#                                                       num_workers=4)
#         test_loader[i] = torch.utils.data.DataLoader(datasets.MNIST(root="~/.torch/data/mnist", train=False, transform=transform),
#                                                      batch_size=batch_size)
#     return train_loader, test_loader

def get_permute_mnist():
    train_loader = {}
    test_loader = {}
    idx = list(range(28 * 28))
    for i in range(num_task):
        train_loader[i] = torch.utils.data.DataLoader(PermutedMNIST(train=True, permute_idx=idx),
                                                      batch_size=batch_size,
                                                      num_workers=4)
        test_loader[i] = torch.utils.data.DataLoader(PermutedMNIST(train=False, permute_idx=idx),
                                                     batch_size=batch_size)
        random.shuffle(idx)
    return train_loader, test_loader

train_loader, test_loader = get_permute_mnist()

def l0_process(epochs, importance, use_cuda=True, weight=None):
    model = L0LeNet5(10)
    if torch.cuda.is_available() and use_cuda:
        model.cuda()
    optimizer = optim.SGD(params=model.parameters(), lr=lr)

    loss, acc, ewc = {}, {}, {}
    for task in range(num_task):
        loss[task] = []
        acc[task] = []

        if task == 0:
            if weight:
                model.load_state_dict(weight)
            else:
                for epoch in tqdm(range(epochs)):
                    print("Epoch {}".format(epoch))
                    loss[task].append(train_l0(model, optimizer, train_loader[task]))
                    task_test_acc = test(model, test_loader[task])
                    print('Test acc for task {}, epoch {}, acc {}'.format(task, epoch, task_test_acc))
                    acc[task].append(task_test_acc)
        else:
            old_tasks = []
            for sub_task in range(task):
                old_tasks = old_tasks + train_loader[sub_task].dataset.get_sample(sample_size)
            old_tasks = random.sample(old_tasks, k=sample_size)
            train_l0(model, optimizer, old_tasks)
            l0_scores, model_params = get_l0_scores_and_model_params(model)
            for _ in tqdm(range(epochs)):
                loss[task].append(l0_train_trans(model, optimizer, train_loader[task], l0_scores, model_params, importance))
                for sub_task in range(task + 1):
                    task_test_acc = test(model, test_loader[sub_task])
                    print('Test acc for task {}, epoch {}, acc {}'.format(task, epoch, task_test_acc))
                    acc[sub_task].append(task_test_acc)

    print(loss)
    print(acc)
    return loss, acc

def loss_plot(x):
    for t, v in x.items():
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v)

def accuracy_plot(x):
    for t, v in x.items():
        plt.plot(list(range(t * epochs, num_task * epochs)), v)
    plt.ylim(0, 1)

loss_ewc, acc_ewc = l0_process(epochs, importance=1000 )
loss_plot(loss_ewc)
accuracy_plot(acc_ewc)

plt.plot(acc_ewc[0], label="ewc")
plt.legend()
plt.savefig('L0_transfer.png')

