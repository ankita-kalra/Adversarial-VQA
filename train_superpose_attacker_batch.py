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

import config_att_batch as config
import superpose_data as data
import superpose_att_model as model
import utils

#import vqa_model as model
from superpose_attack_batch import Attacker as Attacker
#from attack_batch import CarliniAttacker as CarliniAttacker
import pickle
import numpy as np


def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_iterations = 0

'''
def run(net, loader, optimizer, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    log_softmax = nn.LogSoftmax().cuda()
    for v1, v2, q1, q2, a1, a2, idx, q_len1, q_len2 in tq:
        var_params = {
            'volatile': not train,
            'requires_grad': False,
        }
        v = Variable(v.cuda(async=True), **var_params)
        q = Variable(q.cuda(async=True), **var_params)
        a = Variable(a.cuda(async=True), **var_params)
        q_len = Variable(q_len.cuda(async=True), **var_params)

        out = net(v, q, q_len)
        nll = -log_softmax(out)
        loss = (nll * a / 10).sum(dim=1).mean()
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
        acc_tracker.append(acc.mean())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    if not train:
        answ = list(torch.cat(answ, dim=0))
        accs = list(torch.cat(accs, dim=0))
        idxs = list(torch.cat(idxs, dim=0))
        return answ, accs, idxs
'''


def run(attacker, vqa_model, loader, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        #attacker.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        #attacker.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    noise_tracker = tracker.track('{}_noise'.format(prefix), tracker_class(**tracker_params))
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
    orig_tracker = tracker.track('{}_orig'.format(prefix), tracker_class(**tracker_params))

    origs = np.array([])
    successes = np.array([])
    dec_accs = []
    tot_mean_noise = 0
    i = 0
    for v1, v2, q1, q2, a1, a2, idx, q_len1, q_len2 in tq:
            var_params = {
                'volatile': not train,
                'requires_grad': False,
            }
            v1 = Variable(v1.cuda(async=True), **var_params)
            q1 = Variable(q1.cuda(async=True), **var_params)
            a1 = Variable(a1.cuda(async=True), **var_params)
            q_len1 = Variable(q_len1.cuda(async=True), **var_params)

            v2 = Variable(v2.cuda(async=True), **var_params)
            q2 = Variable(q2.cuda(async=True), **var_params)
            a2 = Variable(a2.cuda(async=True), **var_params)
            q_len2 = Variable(q_len2.cuda(async=True), **var_params)


            if train:
                global total_iterations
                orig, success, img, loss1, loss2, mean_noise = attacker.perform(v1, v2, q1, q2, a1, a2, q_len1, q_len2, total_iterations)

                #Update Learning rate. Use smooth decay. Can replace by decrease_on_plateau scheduler
                total_iterations += 1
                #update_learning_rate(optimizer, total_iterations)
                loss_tracker.append(loss1.data[0] + loss2.data[0])
                noise_tracker.append(loss2.data[0])
                
            else:
                orig, success, img, _, loss2, mean_noise = attacker.perform_validation(v1, v2, q1, q2, a1, a2, q_len1, q_len2, total_iterations)
                loss_tracker.append(loss2.data[0]) #Tracks only the noise. Not the entire loss
                noise_tracker.append(loss2.data[0])
                
                

            origs = np.concatenate((origs, orig))
            successes = np.concatenate((successes, success))
            tot_mean_noise += mean_noise.data[0]
            i += 1   
  	    orig_tracker.append(np.sum(orig) / orig.shape[0])

            acc_tracker.append(np.sum(np.logical_and(orig, ~success)) / orig.shape[0])
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value), noise=fmt(noise_tracker.mean.value), acc=fmt(acc_tracker.mean.value), orig=fmt(orig_tracker.mean.value))

    orig_acc = np.sum(origs) / len(origs) #Accuracy of original VQA model
    att_pred = np.logical_and(origs, ~(successes))
    att_acc = np.sum(att_pred) / len(att_pred) #Accuracy of attacked VQA model
    dec_acc = orig_acc - att_acc #Decrease in VQA accuracy
    asucr = dec_acc / orig_acc
    enr = asucr / (tot_mean_noise / i)

    if train:
        return loss1, loss2, dec_acc, asucr, enr, orig_acc, att_acc

    else:
        return None, loss2, dec_acc, asucr, enr, orig_acc, att_acc

            

def main():
    global total_iterations
    total_iterations = 1
 
    if len(sys.argv) > 1:
        name = 'attacker_' + (' '.join(sys.argv[1:]))
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', 'attacker_{}.pth'.format(name))
    print('will save to {}'.format(target_name))

    cudnn.benchmark = True
    print("Initialize Data Loaders")
    train_loader = data.get_loader(train=True, batch_size=config.batch_size)
    val_loader = data.get_loader(val=True, batch_size=config.batch_size)

    #net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
    # Define and loadup the models
    print("Initialize VQA model")
    vqa_model = model.VQANet()

    # Get question and vocab answer
    vocab = vqa_model.get_vocab()

    # Define attacker
    print("Load Attacker model")
    #Uncomment this for AttendAndAttackNet
    attacker = Attacker(vqa_model)

    #Uncomment this for Carlini
    #attacker = CarliniAttacker(vqa_model)

    #optimizer = optim.Adam([p for p in attacker.attack_model.parameters() if p.requires_grad])
    #scheduler = ReduceLROnPlateau(optimizer, 'min')

    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    print("Begin Training")
    eval_after_epochs = 1 #Run eval after these many epochs
    for i in range(config.epochs):
        #Run a train epoch
        loss1, noise, dec_acc, asucr, enr, orig_acc, att_acc = run(attacker, vqa_model, train_loader, tracker, train=True, prefix='train', epoch=i)
        print("Epoch " + str(i) +" : Training Results: Decrease in VQA acc: "+ str(dec_acc) + " ASUCR: " +str(asucr) + " ENR: "+str(enr))
        if i % eval_after_epochs == 0:
            #Run eval 
            _, noise, dec_acc, asucr, enr, orig_acc, att_acc = run(attacker, vqa_model, val_loader, tracker, train=False, prefix='val', epoch=i)
            print("Epoch " + str(i) +" : Validation Results: Decrease in VQA acc: "+ str(dec_acc) + " ASUCR: " +str(asucr) + " ENR: "+str(enr))
            #Put if condition here to save only if attacker improves it's performance

            results = {
                'name': name,
                'tracker': tracker.to_dict(),
                'config': config_as_dict,
                'weights': attacker.state_dict(),
                'eval': {
                    'dec_acc': dec_acc,
                    'noise': noise,
                    'asucr': asucr,
                    'att_acc': att_acc
                },
                'vocab': train_loader.dataset.vocab,
            }
            torch.save(results, target_name)


if __name__ == '__main__':
    main()

