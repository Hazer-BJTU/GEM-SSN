import torch
from models import SeqSleepNet
from models import init_weight
from load_data import load_data_isrucs3
from load_data import Continuum
import argparse
import torch.nn as nn
import copy
import random
import sys


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


def train_procedure(net, train_iter, valid_iter, num_epochs, lr, weight_decay, device):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, num_epochs // 6, 0.6)
    loss = nn.CrossEntropyLoss()
    best_train_loss, best_train_acc, best_valid_acc, best_net = 0, 0, 0, None
    for epoch in range(num_epochs):
        train_loss, train_acc, valid_acc, total = 0, 0, 0, 0
        net.train()
        for X_org, y_org in train_iter:
            X, y = X_org.to(device), y_org.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            y = y.view(-1)
            L = loss(y_hat, y)
            L.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            train_loss += L
            train_acc += correct(y_hat, y)
            total += y.shape[0]
        train_loss /= total
        train_acc /= total
        valid_acc = evaluate(net, valid_iter, device)
        scheduler.step()
        print(f'epoch: {epoch + 1}, train loss: {train_loss:.3f}, '
              f'train acc: {train_acc:.3f}, valid acc: {valid_acc:.3f}')
        if valid_acc > best_valid_acc:
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_valid_acc = valid_acc
            best_net = copy.deepcopy(net)
    return best_train_loss, best_train_acc, best_valid_acc, best_net


def k_fold_train(args):
    mapping_list = [i for i in range(1, 11)]
    random.shuffle(mapping_list)
    results = []
    for i in range(5):
        train_list = [j for j in range(1, 11)]
        valid_idx = mapping_list[2 * i]
        test_idx = mapping_list[2 * i + 1]
        train_list.remove(valid_idx)
        train_list.remove(test_idx)
        print(f'train idx: {train_list}, valid idx: {valid_idx}, test idx: {test_idx}')
        train_iter = load_data_isrucs3(train_list, args.channel, args.batch_size, args.window_size, True, True)
        valid_iter = load_data_isrucs3([valid_idx], args.channel, 1, args.window_size, False, False)
        test_iter = load_data_isrucs3([test_idx], args.channel, 1, args.window_size, False, False)
        net = SeqSleepNet()
        net.apply(init_weight)
        device = torch.device(f'cuda:{args.cuda_idx}')
        train_loss, train_acc, valid_acc, best_net = train_procedure(net, train_iter, valid_iter, args.num_epochs,
                                                                     args.lr, args.weight_decay, device)
        test_acc = evaluate(best_net, test_iter, device)
        results.append((train_loss, train_acc, valid_acc, test_acc))
    return results


def write_results(results):
    with open('output_record.txt', 'w') as file:
        original_stdout = sys.stdout
        sys.stdout = file
        print(f'(train loss, train acc, valid acc, test acc)')
        cnt = 1
        train_loss, train_acc, valid_acc, test_acc = 0, 0, 0, 0
        for item in results:
            print(f'case{cnt}: ({item[0]:.3f}, {item[1]:.3f}, {item[2]:.3f}, {item[3]:.3f})')
            train_loss += item[0]
            train_acc += item[1]
            valid_acc += item[2]
            test_acc += item[3]
            cnt += 1
        print(f'average loss: {train_loss / 5:.3f}, train acc: {train_acc / 5:.3f}, '
              f'valid acc: {valid_acc / 5:.3f}, test acc: {test_acc / 5:.3f}')
        sys.stdout = original_stdout


standard_network = SeqSleepNet()
standard_network.apply(init_weight)


class CLNetwork:
    def __init__(self, args):
        self.net = copy.deepcopy(standard_network)
        self.net.apply(init_weight)
        self.memorys = []
        self.buffer_size = args.buffer_size
        self.memory_buffer_data = torch.zeros((args.buffer_size, args.window_size, 129, 48), dtype=torch.float32)
        self.memory_buffer_label = torch.zeros((args.buffer_size, args.window_size), dtype=torch.int64)
        self.sample_num = 0
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.best_train_loss, self.best_train_acc, self.best_valid_acc = 0, 0, 0
        self.train_loss, self.train_acc, self.num_samples = 0, 0, 0
        self.best_net = copy.deepcopy(self.net)
        self.device = torch.device(f'cuda:{args.cuda_idx}')
        self.net.to(self.device)
        self.epoch = 0

    def reservoir_sampling(self, X_org, y_org):
        for i in range(X_org.shape[0]):
            if self.sample_num < self.buffer_size:
                self.memory_buffer_data[self.sample_num].copy_(X_org[i])
                self.memory_buffer_label[self.sample_num].copy_(y_org[i])
                self.sample_num += 1
            elif random.random() <= self.buffer_size / (self.sample_num + 1):
                idx = random.randint(0, self.buffer_size - 1)
                self.memory_buffer_data[idx].copy_(X_org[i])
                self.memory_buffer_label[idx].copy_(y_org[i])
                self.sample_num += 1

    def start_task(self):
        self.epoch = 0
        self.best_train_loss, self.best_train_acc, self.best_valid_acc = 0, 0, 0
        self.sample_num = 0

    def start_epoch(self):
        self.train_loss, self.train_acc, self.num_samples = 0, 0, 0
        self.net.train()

    def observe(self, X_org, y_org, args, first_time=False):
        if first_time:
            self.reservoir_sampling(X_org, y_org)
        X, y = X_org.to(self.device), y_org.to(self.device)
        self.optimizer.zero_grad()
        y_hat = self.net(X)
        y = y.view(-1)
        L_current = torch.sum(self.loss(y_hat, y))
        L = L_current / X.shape[0]
        if args.replay_mode == 'naive':
            B = 0
            for item in self.memorys:
                B += item[0].shape[0]
            print(f'naively replay on {len(self.memorys)} tasks and {B} examples...')
            for item in self.memorys:
                Xr, yr = item[0].to(self.device), item[1].to(self.device)
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
        self.memorys.append((self.memory_buffer_data.clone(), self.memory_buffer_label.clone()))

    def test(self, test_iter):
        return evaluate(self.best_net, test_iter, self.device)


def train_cl(args, continuum):
    clnetwork = CLNetwork(args)
    test_iters = []
    for i in range(continuum.tasks_num):
        test_iters.append(
            load_data_isrucs3(continuum.get_test(), continuum[i], 1, args.window_size, False, False)
        )
    R = torch.zeros(continuum.tasks_num + 1, continuum.tasks_num)
    for j in range(continuum.tasks_num):
        R[0][j] = clnetwork.test(test_iters[j])
    for i in range(continuum.tasks_num):
        train_iter = load_data_isrucs3(continuum.get_train(), continuum[i], args.batch_size,
                                       args.window_size, True, True)
        valid_iter = load_data_isrucs3(continuum.get_valid(), continuum[i], 1, args.window_size, False, False)
        print(f'start task: {continuum[i]}')
        clnetwork.start_task()
        for epoch in range(args.num_epochs):
            clnetwork.start_epoch()
            for X_org, y_org in train_iter:
                if epoch == 0:
                    clnetwork.observe(X_org, y_org, args, True)
                else:
                    clnetwork.observe(X_org, y_org, args, False)
            clnetwork.end_epoch(valid_iter)
        clnetwork.end_task()
        for j in range(continuum.tasks_num):
            R[i + 1][j] = clnetwork.test(test_iters[j])
    '''
    for item in clnetwork.memorys:
        print(item[0])
        print(item[1])
    '''
    return R


def write_format(R, continuum, filepath='cl_output_record.txt'):
    original_stdout = sys.stdout
    with open(filepath, 'w') as file:
        sys.stdout = file
        print('tasks: ', end='')
        for i in range(continuum.tasks_num):
            print(f'[{continuum[i]}]', end=' ')
        print('')
        print('-' * (8 * continuum.tasks_num + 8))
        for i in range(continuum.tasks_num + 1):
            print(f'task:{i} |', end='')
            for j in range(continuum.tasks_num):
                print(f' {R[i][j]:.3f} ', end='|')
            print('')
        print('-' * (8 * continuum.tasks_num + 8))
        aacc, bwt, fwt = 0, 0, 0
        for j in range(continuum.tasks_num):
            aacc += R[continuum.tasks_num][j]
            if j != continuum.tasks_num - 1:
                bwt += R[continuum.tasks_num][j] - R[j + 1][j]
            if j != 0:
                fwt += R[j][j] - R[0][j]
        aacc, bwt, fwt = aacc / continuum.tasks_num, bwt / (continuum.tasks_num - 1), fwt / (continuum.tasks_num - 1)
        print(f'average acc: {aacc:.3f}')
        print(f'BWT: {bwt:.3f}')
        print(f'FWT: {fwt:.3f}')
    sys.stdout = original_stdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train procedure')
    parser.add_argument('--cuda_idx', type=int, nargs='?', default=0)
    parser.add_argument('--num_epochs', type=int, nargs='?', default=30)
    parser.add_argument('--batch_size', type=int, nargs='?', default=32)
    parser.add_argument('--window_size', type=int, nargs='?', default=10)
    parser.add_argument('--lr', type=float, nargs='?', default=1e-3)
    parser.add_argument('--weight_decay', type=float, nargs='?', default=1e-5)
    parser.add_argument('--channel', type=str, nargs='?', default='F3_A2')
    parser.add_argument('--buffer_size', type=int, nargs='?', default=128)
    parser.add_argument('--phase', type=int, nargs='?', default=1)
    parser.add_argument('--replay_mode', type=str, nargs='?', default='naive')
    args = parser.parse_args()
    if args.phase == 0:
        results = k_fold_train(args)
        write_results(results)
    elif args.phase == 1:
        continuum = Continuum()
        R = train_cl(args, continuum)
        write_format(R, continuum, 'cl_output_replay.txt')
        args.replay_mode = 'none'
        R = train_cl(args, continuum)
        write_format(R, continuum, 'cl_output_none_replay.txt')
    