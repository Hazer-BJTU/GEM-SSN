import copy
import torch
import torch.nn as nn
from models import SeqSleepNet
from models import TinySeqSleepNet
from models import SeqSleepNetClops
from models import init_weight
import numpy as np


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
        if args.using_saved_network:
            print('using saved network')
            seqsleepnet = SeqSleepNet()
            seqsleepnet.load_state_dict(torch.load('saved_network.pt'))
            self.net = SeqSleepNetClops(seqsleepnet, args.batch_size)
        elif args.model_volume == 'standard':
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
        '''
        CLOPS
        '''
        self.mc_epochs = args.mc_epochs
        self.task_memory = []

    def start_task(self):
        self.epoch = 0
        self.best_net = copy.deepcopy(self.net)
        self.best_train_loss, self.best_train_acc, self.best_valid_acc = 0, 0, 0
        '''
        CLOPS
        '''
        self.task_memory = []

    def start_epoch(self):
        self.train_loss, self.train_acc, self.num_samples = 0, 0, 0
        self.net.train()

    def observe(self, X_org, y_org, first_time=False):
        X, y = X_org.to(self.device), y_org.to(self.device)
        self.net.clear_beta()
        S = torch.zeros(X.shape[0], requires_grad=False, device=self.device)
        '''
        print('initial beta:')
        print(self.net.get_beta())
        '''
        for mc_epoch in range(self.mc_epochs):
            self.optimizer.zero_grad()
            y_hat = self.net(X)
            y = y.view(-1)
            L_current = self.loss(y_hat, y)
            L_current = torch.sum(L_current.view(X.shape[0], -1), dim=1)
            beta = self.net.get_beta()
            L = torch.sum(beta[0:X.shape[0]] * L_current)
            if self.args.replay_mode == 'clops':
                B = 0
                for item in self.memory_buffer_data:
                    B += item.shape[0]
                print(f'replay on {len(self.memory_buffer_data)} tasks and {B} examples...')
                for item0, item1 in zip(self.memory_buffer_data, self.memory_buffer_label):
                    Xr, yr = item0.to(self.device), item1.to(self.device)
                    yr_hat = self.net(Xr)
                    yr = yr.view(-1)
                    L_replay = torch.sum(self.loss(yr_hat, yr))
                    L = L + L_replay / B
            L.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=20, norm_type=2)
            self.optimizer.step()
            self.train_loss += L
            self.train_acc += correct(y_hat, y)
            self.num_samples += y.shape[0]
            S = S + self.net.beta.detach()[0:X.shape[0]]
        '''
        print('beta after mc epochs:')
        print(self.net.get_beta())
        '''
        if first_time:
            lst = []
            for i in range(S.shape[0]):
                lst.append((S[i].item(), i))
            lst.sort(key=lambda x: -x[0])
            for i in range(min(len(lst), self.args.clops_ratio)):
                self.task_memory.append([0, torch.unsqueeze(X_org[lst[i][1]].clone(), dim=0),
                                         torch.unsqueeze(y_org[lst[i][1]].clone(), dim=0)])
            print(f'storing temporary examples, currently {len(self.task_memory)}...')

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
        '''
        print(self.memory_buffer_data)
        print(self.memory_buffer_label)
        '''
        softmax = nn.Softmax(dim=1)
        print('calculating example perplexity...')
        with torch.no_grad():
            for i in range(len(self.task_memory)):
                self.net.train()
                H1 = 0
                y_sum = torch.zeros((10, 5), requires_grad=False, device=self.device)
                for mc_epoch in range(self.args.mc_epochs):
                    X = self.task_memory[i][1].to(self.device)
                    y_hat = softmax(self.net(X))
                    y_sum += y_hat
                    y_h = np.array(y_hat.cpu())
                    H1 += -np.sum(y_h * np.log2(y_h))
                H1 /= (10 * self.args.mc_epochs)
                y_sum = torch.sum(y_sum, dim=0) / (10 * self.args.mc_epochs)
                H0 = np.array(y_sum.cpu())
                H0 = -np.sum(H0 * np.log2(H0))
                self.task_memory[i][0] = H0 - H1
            self.task_memory.sort(key=lambda x: -x[0])
            '''
            for item in self.task_memory:
                print(item[0])
            '''
        print('storing examples for replay...')
        memory_data, memory_label = None, None
        for i in range(min(self.buffer_size, len(self.task_memory))):
            if i == 0:
                memory_data = self.task_memory[i][1]
                memory_label = self.task_memory[i][2]
            else:
                memory_data = torch.cat((memory_data, self.task_memory[i][1]), dim=0)
                memory_label = torch.cat((memory_label, self.task_memory[i][2]), dim=0)
        print(f'{memory_data.shape[0]} examples stored...')
        print(memory_data.shape)
        print(memory_label.shape)
        self.memory_buffer_data.append(memory_data)
        self.memory_buffer_label.append(memory_label)

    def test(self, test_iter):
        return evaluate(self.best_net, test_iter, self.device)


if __name__ == '__main__':
    pass
