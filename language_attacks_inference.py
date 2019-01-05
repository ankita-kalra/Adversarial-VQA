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

import cv2
import torch.nn.functional as F

import config_lang_att as config
import data_for_inference as data
import att_model as model
import utils

#import vqa_model as model
from attack_batch import Attacker as Attacker
from attack_batch import CarliniAttacker as CarliniAttacker
import pickle
import numpy as np
import pdb

#f = open(config.log_path, 'w')
q_dict = np.load('q_dict.npy').item()
a_dict = np.load('a_dict.npy').item()

def sent_from_que(que, max_q_len=28):
    i = 0
    sent = ''
    while i  < max_q_len and que[i] != 247:
	if que[i] == 0:
		if i + 1 < max_q_len and que[i + 1] != 0:
			sent = sent + '<unk> '
		else:
			return sent
	else: 
        	sent = sent + q_dict[que[i]] + ' '
        i += 1
    return sent


def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_iterations = 0

###########################################################################################################
#Functions for visual analysis
##########################################################################################################
unorm = utils.UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def save_class_activation_on_image(img_cv, activation_map, path_to_file=None):
    """
        Saves and returns cam  activation map on the original image
    Args:
        img_cv (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        path_to_file (str): path to store the visualization map to
    """

    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    #path_to_file = os.path.join('../results', file_name+'_Cam_Heatmap.jpg')
    #cv2.imwrite(path_to_file, activation_heatmap)
    # Heatmap on picture
    #img_cv = cv2.resize(img_cv, (448, 448))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(img_cv)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    #path_to_file = os.path.join('../results', file_name+'_Cam_On_Image.jpg')
    final_img = np.uint8(255 * img_with_heatmap)
    if path_to_file != None:
        cv2.imwrite(path_to_file, final_img)

    return final_img

def vis_attention(img, q, ans, att_map, path=None, multiplier=10):
    '''
    Function to visualize the attention maps:  
    img: 3 X 448 X 448 (pytorch variable)
    q: 23 (numpy array)
    ans: 1 (int)
    att_map: 14 X 14 (pytorch variable)
    path: path to save visualization attention map
    multiplier: saliency multiplier for attention maps

    returns: att_map over image, questions in english, answers in english
    '''

    #q_dict = np.load('q_dict.npy').item()
    #a_dict = np.load('a_dict.npy').item()
    #unorm = utils.UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


    sent = sent_from_que(q, q_dict)
    anss = (a_dict[ans])

    #Resize att map to full res
    rsz_att_map = cv2.resize(multiplier * att_map.data.cpu().numpy(), (img.size(2), img.size(2)))    #5 * att values to make maps more salient
    #Convert to 0-255 range
    final_att = np.uint8(255 * rsz_att_map)


    img_np1 = unorm(img.data).cpu().numpy()
    #COnvert Image to PIL format
    img_cv = np.transpose(img_np1,(1,2,0))
    img_cv = cv2.convertScaleAbs(img_cv.reshape(448,448,3)*255)


    att_over_img = save_class_activation_on_image(img_cv, final_att, path)

    return att_over_img, sent, anss


def save_image(image, path=None):
    '''
    image: 3 X 448 X 448 (Pytorch variable directly)

    saves the image as a png image
    '''

    img_np1 = unorm(image.data).cpu().numpy()
    img_cv = np.transpose(img_np1,(1,2,0))
    img_cv = cv2.convertScaleAbs(img_cv.reshape(448,448,3)*255)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    if path != None:
        cv2.imwrite(path, img_cv)
##################################################################################################################################################################




################################################################
#Main workhorse function for language attack inference
#################################################################
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
    bi = 0

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
        	

		a_new = F.softmax(a_.view(a_.size(0), a_.size(1), -1), 2)
        	a_new = a_new.view(a_.size(0), a_.size(1), a_.size(2), a_.size(3))



                prob_value, ans_index = ans_.data.cpu().max(dim=1)
                #ans stores the target index for the attack or ground truth for untargetted
                _, target_idx = a.data.cpu().max(dim=1)

                orig = (ans_index == target_idx).numpy()
                

	    que = q.cpu().data.numpy()
	    ans = a.cpu().data.numpy()
	    max_q_len = que.shape[1]
	    path = None
	    multiplier=10
            if bi == 0:
	    	#pdb.set_trace()
            	att_over_img, sent, anss = vis_attention(v[38], que[38], target_idx[38], a_new[38, 0], path, multiplier)
	    	#pdb.set_trace()

	    	for j in range(que.shape[0]):
			ques = que[j]
			sent = sent_from_que(ques, max_q_len)
			anss = a_dict[target_idx[j]]
			pred_ans = a_dict[ans_index[j]]
#			if ans_index[j] == target_idx[j]:
#				f.write("Correct!!! , "+str(sent) + ' , ' + str(anss) + ' , ' + str(pred_ans) + '\n')      
#			else:
#				f.write("Wrong!!! , "+str(sent) + ' , ' + str(anss) + ' , ' + str(pred_ans) + '\n')          
			print(str(sent) + ' , ' + str(anss) + ' , ' + str(pred_ans) + '\n')

	    #pdb.set_trace()
	    bi += 1
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
    #f.close()

