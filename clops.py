import copy

import torch
import torch.nn as nn
from models import SeqSleepNet
from models import TinySeqSleepNet
from models import SeqSleepNetClops
from models import init_weight


def correct(y_hat, y):
    y_hat = y_hat.argmax(axis=1).to(torch.int64)
    cmp = (y == y_hat)
    return float(cmp.sum().item())


def evaluate(net, dataloader, device):
    net.to(device)
    net.eval()
    total, true = 0, 0
    with torch.no_grad():
        for X_org, y_org in dataloader:
            X, y = X_org.to(device), y_org.to(device)
            y_hat = net(X)
            y = y.view(-1)
            true += correct(y_hat, y)
            total += y.shape[0]
    return true / total


standard_network = SeqSleepNet()
standard_network.apply(init_weight)
tiny_network = TinySeqSleepNet()
tiny_network.apply(init_weight)


class CLNetworkClops:
    def __init__(self, args):
        self.args = args
        if args.model_volume == 'standard':
            print('using standard network.')
            self.net = SeqSleepNetClops(copy.deepcopy(standard_network), args.batch_size)
        elif args.model_volume == 'tiny':
            print('using tiny network')
            self.net = SeqSleepNetClops(copy.deepcopy(tiny_network), args.batch_size)
        self.buffer_size = args.buffer_size
        self.memory_buffer_data = []
        self.memory_buffer_label = []
        self.sample_num = 0
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.best_train_loss, self.best_train_acc, self.best_valid_acc = 0, 0, 0
        self.train_loss, self.train_acc, self.num_samples = 0, 0, 0
        self.best_net = copy.deepcopy(self.net)
        self.best_net_memory = []
        self.device = torch.device(f'cuda:{args.cuda_idx}')
        self.net.to(self.device)
        self.epoch = 0

    def start_task(self):
        self.epoch = 0
        self.best_net = copy.deepcopy(self.net)
        self.best_train_loss, self.best_train_acc, self.best_valid_acc = 0, 0, 0

    def start_epoch(self):
        self.train_loss, self.train_acc, self.num_samples = 0, 0, 0
        self.net.train()

    def observe(self, X_org, y_org, first_time=False):
        X, y = X_org.to(self.device), y_org.to(self.device)
        self.optimizer.zero_grad()
        y_hat = self.net(X)
        y = y.view(-1)
        L_current = torch.sum(self.loss(y_hat, y))
        L = L_current / X.shape[0]
        L.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=20, norm_type=2)
        self.optimizer.step()
        self.train_loss += L
        self.train_acc += correct(y_hat, y)
        self.num_samples += y.shape[0]

    def end_epoch(self, valid_iter):
        self.train_loss /= self.num_samples
        self.train_acc /= self.num_samples
        valid_acc = evaluate(self.net, valid_iter, self.device)
        if valid_acc > self.best_valid_acc:
            self.best_train_loss = self.train_loss
            self.best_train_acc = self.train_acc
            self.best_valid_acc = valid_acc
            self.best_net = copy.deepcopy(self.net)
        self.epoch += 1
        print(f'epoch: {self.epoch}, train loss: {self.train_loss:.3f}, '
              f'train acc: {self.train_acc:.3f}, valid acc: {valid_acc:.3f}')

    def end_task(self):
        self.best_net_memory.append(self.best_net)

    def test(self, test_iter):
        return evaluate(self.best_net, test_iter, self.device)


if __name__ == '__main__':
    pass
