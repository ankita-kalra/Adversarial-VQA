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

import config_lang_att as config
import data_for_inference as data
import att_model as model
import utils

#import vqa_model as model
from attack_batch import Attacker as Attacker
from attack_batch import CarliniAttacker as CarliniAttacker
import pickle
import numpy as np


def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_iterations = 0


def run(vqa_model, loader, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        #attacker.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        #attacker.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))


    origs = np.array([])
    i = 0
    for v, q, a, idx, q_len in tq:
            var_params = {
                'volatile': not train,
                'requires_grad': False,
            }
            v = Variable(v.cuda(async=True), **var_params)
            q = Variable(q.cuda(async=True), **var_params)
            a = Variable(a.cuda(async=True), **var_params)
            q_len = Variable(q_len.cuda(async=True), **var_params)

            if train:
                global total_iterations
                #orig, success, img, loss1, loss2, mean_noise = attacker.perform(v, q, q_len, a, total_iterations)

                #Update Learning rate. Use smooth decay. Can replace by decrease_on_plateau scheduler
                #total_iterations += 1
                #update_learning_rate(optimizer, total_iterations)
                #loss_tracker.append(loss1.data[0] + loss2.data[0])
                #noise_tracker.append(loss2.data[0])
                
            else:
                #orig, success, img, _, loss2, mean_noise = attacker.perform_validation(v, q, q_len, a, total_iterations)
                #loss_tracker.append(loss2.data[0]) #Tracks only the noise. Not the entire loss
                #noise_tracker.append(loss2.data[0])

                ans_, att_, a_ = vqa_model.forward_pass(v, q, q_len)

                prob_value, ans_index = ans_.data.cpu().max(dim=1)
                #ans stores the target index for the attack or ground truth for untargetted
                _, target_idx = a.data.cpu().max(dim=1)

                orig = (ans_index == target_idx).numpy()
                
                



            acc_tracker.append(np.sum(orig) / orig.shape[0])
            origs = np.concatenate((origs, orig))
            fmt = '{:.4f}'.format
            tq.set_postfix(acc=fmt(acc_tracker.mean.value))

    return (np.sum(origs) / origs.shape[0])

            

def main():
    global total_iterations
    total_iterations = 1
 
    if len(sys.argv) > 1:
        name = 'lang_attacker_' + (' '.join(sys.argv[1:]))
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', 'lang_attacker_{}.pth'.format(name))
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
    #print("Load Attacker model")
    #Uncomment this for AttendAndAttackNet
    #attacker = Attacker(vqa_model)

    #Uncomment this for Carlini
    #attacker = CarliniAttacker(vqa_model)

    #optimizer = optim.Adam([p for p in attacker.attack_model.parameters() if p.requires_grad])
    #scheduler = ReduceLROnPlateau(optimizer, 'min')

    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    print("Begin Inference")
    eval_after_epochs = 1 #Run eval after these many epochs
    for i in range(1): #Only 1 epoch needed for inference. Keep this to later generalize to training to protect against attacks
        #Run a train epoch
        #acc = run(vqa_model, train_loader, tracker, train=True, prefix='train', epoch=i)

        #Run inference
        acc = run(vqa_model, val_loader, tracker, train=False, prefix='val', epoch=i)

        print("Epoch " + str(i) +" : Inference Results: Accuracy: "+ str(acc)) 

        '''
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
        '''


if __name__ == '__main__':
    main()

