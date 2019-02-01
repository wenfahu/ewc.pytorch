from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from l0_layers import L0Dense, L0Conv2d


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input, _ in self.dataset:
            self.model.zero_grad()
            input = input.cuda()
            if input.dim() == 3:
                input = input.unsqueeze_(0)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target) + importance * ewc.penalty(model)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


def test(model: nn.Module, data_loader: torch.utils.data.DataLoader):
    with torch.no_grad():
        model.eval()
        correct = 0
        for input, target in data_loader:
            input, target = variable(input), variable(target)
            output = model(input)
            correct += (F.softmax(output, dim=1).argmax(1) == target).data.sum()
    acc = correct.float() / len(data_loader.dataset)
    return acc

def train_l0(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        if input.dim() == 3:
            input.unsqueeze_(0)
            target.unsqueeze_(0)
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        xent_loss = F.cross_entropy(output, target)
        l0_loss = model.regularization()
        loss = xent_loss + l0_loss
        print("Training of l0, xent loss {}, l0 loss {}".format(xent_loss.item(),
            l0_loss.item()))
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)

def get_l0_scores_and_model_params(model: nn.Module):
    l0_scores = {}
    model_params = {}
    with torch.no_grad():
        model.eval()
        for n, m in model.named_modules():
            if isinstance(m, L0Dense) or isinstance(m, L0Conv2d):
                weight_key = '{}.weight'.format(n)
                l0_scores[weight_key] = m.sample_info()
        for n, p in model.named_parameters():
            model_params[n] = p.detach()
    return l0_scores, model_params

def l0_weighted_penalty(model: nn.Module, params: dict, l0_scores: dict):
    penalty = 0
    for n, p in model.named_parameters():
        if n in l0_scores:
            scores = l0_scores[n]
            if scores.dim() == 1:
                scores.unsqueeze_(1)
            penalty += (scores * ( p - params[n]).pow(2)).sum()
    return penalty


def l0_train_trans(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
        l0_scores: dict, params: dict, importance: float):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        xent_loss = F.cross_entropy(output, target)
        l0_trans_loss = importance * l0_weighted_penalty(model,
                params, l0_scores)
        l0_reg_loss = model.regularization()
        loss = xent_loss + l0_trans_loss + l0_reg_loss
        print("xent loss {}, l0 trans loss {}, l0 reg loss {}".format(
            xent_loss.item(), l0_trans_loss.item(), l0_reg_loss.item()))
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)
