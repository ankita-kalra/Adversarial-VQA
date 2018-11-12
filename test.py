import sys
import os.path
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import data
import model
import utils


def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_iterations = 0


def run_test(net, loader, optimizer, tracker,  prefix='', epoch=0):
    """ Run an epoch over the given loader """


    net.eval()
    tracker_class, tracker_params = tracker.MeanMonitor, {}
    answ = []
    idxs = []
    accs = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    log_softmax = nn.LogSoftmax().cuda()
    for v, q, a, idx, q_len in tq:
        var_params = {
            'volatile': False,
            'requires_grad': False,
        }
        v = Variable(v.cuda(async=True), **var_params))
        q = Variable(q.cuda(async=True), **var_params))
        a = Variable(a.cuda(async=True), **var_params))
        q_len = Variable(q_len.cuda(async=True), **var_params))

        out = net(v, q, q_len)
        nll = -log_softmax(out)
        loss = (nll * a / 10).sum(dim=1).mean()
        acc = utils.batch_accuracy(out.data, a.data).cpu()

        # store information about evaluation of this minibatch
        _, answer = out.data.cpu().max(dim=1)
        answ.append(answer.view(-1))
        accs.append(acc.view(-1))
        idxs.append(idx.view(-1).clone())

        loss_tracker.append(loss.data[0])
        acc_tracker.append(acc.mean())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

        answ = list(torch.cat(answ, dim=0))
        accs = list(torch.cat(accs, dim=0))
        idxs = list(torch.cat(idxs, dim=0))
        return answ, accs, idxs


def main():

    dataset_name = 'vqa1'
    target_name = os.path.join('logs', '{}_test.pth'.format(dataset_name))
    print('will save to {}'.format(target_name))

    cudnn.benchmark = True
    test_loader = data.get_loader(test=True)

    checkpoint = torch.load('logs/2017-08-04_00:55:19.pth')
    tokens = len(checkpoint['vocab']['question']) + 1
    net = torch.nn.DataParallel(model.Net(tokens))
    net.load_state_dict(checkpoint['weights'])
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
    tracker = []
    tracker.load_state_dict(checkpoint['tracker'])
    config.load_state_dict(checkpoint['config'])

    t = run_test(net, test_loader, optimizer, tracker, prefix='test', epoch=1)

    results = {
        'name': dataset_name,
        'tracker': tracker.to_dict(),
        'weights': net.state_dict(),
        'eval': {
            'answers': t[0],
            'accuracies': t[1],
            'idx': t[2],
        },

    }
    torch.save(results, target_name)


if __name__ == '__main__':
    main()
