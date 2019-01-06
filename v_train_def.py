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

import v_config_def_batch as config
import data
import v_def_model as model
import utils

import pickle
import numpy as np
import pdb

def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_iterations = 0


def run(net, loader, optimizer, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.vqa_net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.vqa_net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    loss_tracker_v = tracker.track('{}_loss_v'.format(prefix), tracker_class(**tracker_params))
    loss_tracker_q = tracker.track('{}_loss_q'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    log_softmax = nn.LogSoftmax().cuda()
    for v, q, a, idx, q_len in tq:
        var_params = {
            'volatile': False,
            'requires_grad': False,
        }
        v = Variable(v.cuda(async=True), **var_params)
        q = Variable(q.cuda(async=True), **var_params)
        a = Variable(a.cuda(async=True), **var_params)
        q_len = Variable(q_len.cuda(async=True), **var_params)

        out, w_att, att, da_dv, da_dq  = net.forward(v, q, q_len)
        nll = -log_softmax(out)
	#pdb.set_trace()
        #loss = (nll * a / 10).sum(dim=1).mean() + config.lambda_v * torch.mean((da_dv / 10000000.0)**2) + config.lambda_q * torch.mean((da_dq/10.0)**2)
	loss = (nll * a / 10).sum(dim=1).mean() + config.lambda_v * torch.mean(da_dv**2)

        acc = utils.batch_accuracy(out.data, a.data).cpu()

        if train:
            global total_iterations
            update_learning_rate(optimizer, total_iterations)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_iterations += 1
        else:
            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())

        loss_tracker.append(loss.data[0])
	loss_tracker_v.append(config.lambda_v * torch.mean(da_dv**2).data.cpu().numpy()[0])
	loss_tracker_q.append(torch.mean((da_dq/10.0)**2).data.cpu().numpy()[0])
        acc_tracker.append(acc.mean())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), loss_v=fmt(loss_tracker_v.mean.value), loss_q=fmt(loss_tracker_q.mean.value), acc=fmt(acc_tracker.mean.value))

    if not train:
        answ = list(torch.cat(answ, dim=0))
        accs = list(torch.cat(accs, dim=0))
        idxs = list(torch.cat(idxs, dim=0))
        return answ, accs, idxs


def main():
    if len(sys.argv) > 1:
        name = ' '.join(sys.argv[1:])
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}.pth'.format(name))
    print('will save to {}'.format(target_name))

    cudnn.benchmark = True

    train_loader = data.get_loader(train=True, batch_size=config.batch_size)
    val_loader = data.get_loader(val=True, batch_size=config.batch_size)

    #log = torch.load(config.vqa_model_path)
    #tokens = len(log['vocab']['question']) + 1

    #net = torch.nn.DataParallel(vqa_model.Net(tokens)).cuda()
    #net = vqa_model.Net(tokens).cuda()
    #net.load_state_dict(log['weights'], strict=False)

    vqa_model = model.VQANet()
    #vocab = vqa_model.get_vocab()

    optimizer = optim.Adam([p for p in vqa_model.vqa_net.parameters() if p.requires_grad])

    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    for i in range(config.epochs):
        _ = run(vqa_model, train_loader, optimizer, tracker, train=True, prefix='train', epoch=i)
        r = run(vqa_model, val_loader, optimizer, tracker, train=False, prefix='val', epoch=i)

	#pdb.set_trace()

        results = {
            'name': name,
            'tracker': tracker.to_dict(),
            'config': config_as_dict,
            'weights': vqa_model.vqa_net.state_dict(),
            'eval': {
                'answers': r[0],
                'accuracies': r[1],
                'idx': r[2],
            },
            'vocab': train_loader.dataset.vocab,
        }
        torch.save(results, target_name)


if __name__ == '__main__':
    main()
