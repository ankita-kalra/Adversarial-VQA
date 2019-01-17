from torch.autograd import Variable
import torch
from superpose_att_model import AttackNet
import torch.optim as optim
import numpy as np
import torch.nn as nn
import utils
import cv2
import torch.nn.functional as F

import pdb

import config_att_batch as config

max_inter = 1
lr_halflife = config.lr_halflife
initial_lr = config.init_lr
lambda1_multiplier = 1
lambda2_multiplier = 10

def update_learning_rate(optimizer, iteration):
    lr = initial_lr * 0.5**(float(iteration) / lr_halflife)
    for param_group in optimizer.param_groups:
        #print ("LR = ", lr)
        param_group['lr'] = lr

def step_update_learning_rate(optimizer, iteration):
    #lr = initial_lr * 0.5**(float(iteration) / lr_halflife)
    if iteration % lr_halflife == 0:
    	for param_group in optimizer.param_groups:
        	#print ("LR = ", lr)
        	param_group['lr'] /= 2


class Attacker:
    def __init__(self, VQA_model, targetted=False):

        # save a globle vqa model
        self.VQA_model = VQA_model
        # define an attacker net
        self.attack_model = AttackNet()
        # putting it into train mode
        self.attack_model.train() 
        # transfer to gpus
        self.attack_model = nn.DataParallel(self.attack_model).cuda()
        # get all the learnable parameters from it
        self.optimizer = optim.Adam([p for p in 
            self.attack_model.parameters() if p.requires_grad])
        # Define softmax
        self.log_softmax = nn.LogSoftmax().cuda()
        self.scaller_const = Variable(torch.Tensor([1]).float()).cuda()

        # Define unnormalizer
        self.unorm = utils.UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        print("Using lambda_multipliers as: " + str(lambda1_multiplier) + "   lambda2:  " +str(lambda2_multiplier))
        
        # is it targetted
        self.targetted_const = 1
        if targetted == True:
            self.targetted_const = -1


    def perform_validation(self, img1, img2, que1, que2, ans1, ans2, que_len1, que_len2, total_iterations):
        '''
        img: batch of images
        que: batch of questions
        que_len: batch of question lengths
        ans: batch of answers

        Returns: 
        orig: Whether the VQA models prediction on the benign image is correct
        success: Whether the VQA models was attacked successfully
        img: The noisy perturbed image (back to original scale and channels compatible with cv2)
        loss1: Ignored
        loss2: L1 sum of noise added 
        mean_noise: avg per pixel per channel noise
        '''
	img_clone1 = img1.clone()
        img_clone2 = img2.clone()
        #que = Variable(torch.from_numpy(que).long().cuda(async=True))
        #ans = Variable(torch.from_numpy(ans).cuda(async=True))
        #que_len = Variable(torch.from_numpy(que_len).cuda(async=True))

        #pdb.set_trace()        
        ans_1, att_1, a_1 = self.VQA_model.forward_pass(img1, que1, que_len1)
        a_new1 = F.softmax(a_1.view(a_1.size(0), a_1.size(1), -1), 2)
        a_1 = a_new1.view(a_1.size(0), a_1.size(1), a_1.size(2), a_1.size(3))

        ans_2, att_2, a_2 = self.VQA_model.forward_pass(img2, que2, que_len2)
        a_new2 = F.softmax(a_2.view(a_2.size(0), a_2.size(1), -1), 2)
        a_2 = a_new2.view(a_2.size(0), a_2.size(1), a_2.size(2), a_2.size(3))

        #pdb.set_trace()

        prob_value1, ans_index1 = ans_1.data.cpu().max(dim=1)
        #ans stores the target index for the attack or ground truth for untargetted
        _, target_idx1 = ans1.data.cpu().max(dim=1)
        _, target_idx2 = ans2.data.cpu().max(dim=1)

        #orig stores whether the original VQA model predicts the answer correctly
        orig = (ans_index1 == target_idx1).numpy()

        att_cpu1 = att_1.data.cpu().numpy()
        #Break computational graph to not backprop into the VQA model itself
        att_clone1 = Variable(torch.from_numpy(att_cpu1).cuda(), requires_grad= False).view(que.shape[0], 4096, 14, 14)

        att_cpu2 = att_2.data.cpu().numpy()
        #Break computational graph to not backprop into the VQA model itself
        att_clone2 = Variable(torch.from_numpy(att_cpu2).cuda(), requires_grad= False).view(que.shape[0], 4096, 14, 14)
        #####################################################################################################################################Initialize this to a array of False. Use this to zero out loss of entries where already success. Also use for premature exit if needed. Also remove image unNormalize and image reconstruction for now
        #success = False
        success = [False] * que.size(0) #Stores which samples in the batch have already been successfully attacked
        iter_ = 0
        img_cv = None


        #Change this to allow for custom learning rates
        #for param_group in self.optimizer.param_groups:
        #    param_group['lr'] = config.init_lr
        while (iter_ < max_inter):

            iter_ += 1
            #update_learning_rate(self.optimizer, total_iterations) #####################################Replace with scheduler instead using decrease_on_plateau

            #Attack the image now using attention maps
            joint_att = torch.cat((att_clone1, att_clone2), 1)
            purturb = self.attack_model(joint_att)      #Could either be superposition weight map or noise map depending on what you are doing
            update_map = Variable((torch.FloatTensor(success)).unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(img_clone).cuda()) #Use success to construct a mask which prevents changes to perturbed images which succeed in attack
            update_map.requires_grad = False
            #Add noise back to image
            if iter_ == 1:
                img_ = (img_clone1 + purturb)
            else:
                img_ =  (1 - update_map) * (img_clone1 + purturb) + (update_map * img_) #Update only the images for which success is False
            #Get answer and attention maps when perturbed image is fed to the VQA network
            ans_t, att_t, a_t = self.VQA_model.forward_pass(img_, que, que_len)

            #Compute targetted loss
            nll = self.log_softmax(ans_t) # self.targetted_const is 1 if untargetted -1 if targetted
            #ans in targetted case is the target but untargetted case is ground truth ans. (nll*ans) controls the correct loss
            var_success = Variable(torch.FloatTensor(success).cuda())
	    var_success.requires_grad = False
            #pdb.set_trace()
            if iter_ == 1:
                loss1 = self.scaller_const * (nll*ans2).mean()
                mean_noise = torch.abs(purturb).mean(dim=3).mean(dim=2).mean(dim=1)
                loss2 = torch.abs(purturb).mean()  #Using L1 for now, see if you wanna use L2 instead.
                loss = lambda1_multiplier * loss1 + lambda2_multiplier * loss2
            else:
                loss1 = ((self.scaller_const * ((nll*ans2).sum(dim=1) * (1 - var_success))).sum() / torch.sum((1 - var_success))) #Dont penalize samples on which success has already been obtained  
                #Penalize large noise. Can modify to allow a certain extent of noise
                mean_noise = torch.abs((1 - update_map) * purturb).mean(dim=3).mean(dim=2).mean(dim=1) / torch.sum((1 - var_success))
                loss2 = torch.abs((1 - update_map) * purturb).mean(dim=3).mean(dim=2).mean(dim=1).sum() / torch.sum((1 - var_success))  #Penalize only non success cases #Using L1 for now, see if you wanna use L2 instead.
                loss = lambda1_multiplier * loss1 + lambda2_multiplier * loss2


            # get answer index
            prob_value, ans_index = ans_t.data.cpu().max(dim=1)
            _, target_idx2 = ans2.data.cpu().max(dim=1)

            success = np.logical_or(success, np.logical_and(loss2.data.cpu().numpy() < 0.26000, (ans_index.numpy() == target_idx2.numpy())))
            if np.sum(success) == success.shape[0]:
                break


        img_np1 = self.unorm(img_.data).cpu().numpy()
        img_cv = np.transpose(img_np1,(0,2,3,1))
        img_cv = cv2.convertScaleAbs(img_cv.reshape(448,448,3)*255)

        return orig, success, img_cv, loss1, loss2, mean_noise




    def perform(self, img1, img2, que1, que2, ans1, ans2, que_len1, que_len2, total_iterations, val=False):
        '''
        img: batch of images
        que: batch of questions
        que_len: batch of question lengths
        ans: batch of answers
        total_iterations: For scheduler
        val: True if in val mode

        Returns: 
        orig: Whether the VQA models prediction on the benign image is correct
        success: Whether the VQA models was attacked successfully
        img: The noisy perturbed image (back to original scale and channels compatible with cv2)
        loss1: Scaled(lambda) cross entropy loss
        loss2: L1 sum of noise added 
        '''
        # im is torch tensor, converting everything into Variable
        #img = Variable(img.cuda(async=True))
        img_clone1 = img1.clone()
	img_clone2 = img2.clone()
        #que = Variable(torch.from_numpy(que).long().cuda(async=True))
        #ans = Variable(torch.from_numpy(ans).cuda(async=True))
        #que_len = Variable(torch.from_numpy(que_len).cuda(async=True))

	#pdb.set_trace()	
        ans_1, att_1, a_1 = self.VQA_model.forward_pass(img1, que1, que_len1)
	a_new1 = F.softmax(a_1.view(a_1.size(0), a_1.size(1), -1), 2)
        a_1 = a_new1.view(a_1.size(0), a_1.size(1), a_1.size(2), a_1.size(3))       

        ans_2, att_2, a_2 = self.VQA_model.forward_pass(img2, que2, que_len2)
        a_new2 = F.softmax(a_2.view(a_2.size(0), a_2.size(1), -1), 2)
        a_2 = a_new2.view(a_2.size(0), a_2.size(1), a_2.size(2), a_2.size(3))
 
        #pdb.set_trace()

        prob_value1, ans_index1 = ans_1.data.cpu().max(dim=1)
        #ans stores the target index for the attack or ground truth for untargetted
        _, target_idx1 = ans1.data.cpu().max(dim=1)
	_, target_idx2 = ans2.data.cpu().max(dim=1)

        #orig stores whether the original VQA model predicts the answer correctly
        orig = (ans_index1 == target_idx1).numpy()

        att_cpu1 = att_1.data.cpu().numpy()
        #Break computational graph to not backprop into the VQA model itself
        att_clone1 = Variable(torch.from_numpy(att_cpu1).cuda(), requires_grad= False).view(que.shape[0], 4096, 14, 14)

        att_cpu2 = att_2.data.cpu().numpy()
        #Break computational graph to not backprop into the VQA model itself
        att_clone2 = Variable(torch.from_numpy(att_cpu2).cuda(), requires_grad= False).view(que.shape[0], 4096, 14, 14)
	#####################################################################################################################################Initialize this to a array of False. Use this to zero out loss of entries where already success. Also use for premature exit if needed. Also remove image unNormalize and image reconstruction for now
        #success = False
        success = [False] * que.size(0) #Stores which samples in the batch have already been successfully attacked
        iter_ = 0
        img_cv = None

	if not val:
        	step_update_learning_rate(self.optimizer, total_iterations)

        #Change this to allow for custom learning rates
        #for param_group in self.optimizer.param_groups:
        #    param_group['lr'] = config.init_lr
        while (iter_ < max_iter): #Setting max_iter to 1 for now. Change to allow for multi step attack

            iter_ += 1
            #update_learning_rate(self.optimizer, total_iterations) #####################################Replace with scheduler instead using decrease_on_plateau

            #Attack the image now using attention maps
            joint_att = torch.cat((att_clone1, att_clone2), 1)
            purturb = self.attack_model(joint_att)	#Could either be superposition weight map or noise map depending on what you are doing
            update_map = Variable((torch.FloatTensor(success)).unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(img_clone).cuda()) #Use success to construct a mask which prevents changes to perturbed images which succeed in attack
            update_map.requires_grad = False
            #Add noise back to image
            if iter_ == 1:
                img_ = (img_clone1 + purturb)
	    else:
		img_ =  (1 - update_map) * (img_clone1 + purturb) + (update_map * img_) #Update only the images for which success is False
            #Get answer and attention maps when perturbed image is fed to the VQA network
            ans_t, att_t, a_t = self.VQA_model.forward_pass(img_, que, que_len)

            #Compute targetted loss
            nll = self.log_softmax(ans_t) # self.targetted_const is 1 if untargetted -1 if targetted
            #ans in targetted case is the target but untargetted case is ground truth ans. (nll*ans) controls the correct loss
            var_success = Variable(torch.FloatTensor(success).cuda())
            var_success.requires_grad = False
            #pdb.set_trace()
	    if iter_ == 1:
		loss1 = self.scaller_const * (nll*ans2).mean()
                mean_noise = torch.abs(purturb).mean(dim=3).mean(dim=2).mean(dim=1)
                loss2 = torch.abs(purturb).mean()  #Using L1 for now, see if you wanna use L2 instead.
                loss = lambda1_multiplier * loss1 + lambda2_multiplier * loss2
	    else:
            	loss1 = ((self.scaller_const * ((nll*ans2).sum(dim=1) * (1 - var_success))).sum() / torch.sum((1 - var_success))) #Dont penalize samples on which success has already been obtained  
            	#Penalize large noise. Can modify to allow a certain extent of noise
            	mean_noise = torch.abs((1 - update_map) * purturb).mean(dim=3).mean(dim=2).mean(dim=1) / torch.sum((1 - var_success))
            	loss2 = torch.abs((1 - update_map) * purturb).mean(dim=3).mean(dim=2).mean(dim=1).sum() / torch.sum((1 - var_success))  #Penalize only non success cases #Using L1 for now, see if you wanna use L2 instead.
            	loss = lambda1_multiplier * loss1 + lambda2_multiplier * loss2

	    if not val:
            	self.optimizer.zero_grad()
            	loss.backward(retain_graph = True)
            	self.optimizer.step()

            # get answer index
            prob_value, ans_index = ans_t.data.cpu().max(dim=1)
            _, target_idx2 = ans2.data.cpu().max(dim=1)

            success = np.logical_or(success, np.logical_and(loss2.data.cpu().numpy() < 0.26000, (ans_index.numpy() == target_idx2.numpy())))
            if np.sum(success) == success.shape[0]:
            	break
        

	return orig, success, None, loss1, loss2, mean_noise
