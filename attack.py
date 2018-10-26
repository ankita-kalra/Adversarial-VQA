from torch.autograd import Variable
import torch
from att_model import AttackNet
import torch.optim as optim
import numpy as np
import torch.nn as nn
import utils
import cv2

import pdb

max_inter = 250
lr_halflife = 200
initial_lr = 5e-3

def update_learning_rate(optimizer, iteration):
    lr = initial_lr * 0.5**(float(iteration) / lr_halflife)
    for param_group in optimizer.param_groups:
        #print ("LR = ", lr)
        param_group['lr'] = lr

class CarliniAttacker:
    def __init__(self, VQA_model, targetted=False):
        # save a globle vqa model
        self.VQA_model = VQA_model
        self.vocab = VQA_model.get_vocab()
        self.ans_vocab_inv = {b:a for a,b in self.vocab['answer'].items()}
        self.tanh = nn.Tanh().cuda()
        self.targetted = targetted
        self.confidence = 20
        self.scalar_const = Variable(torch.Tensor([1000]).float()).cuda()
        self.unorm = utils.UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def perform_validation(self, im, q, q_len, a):
        img = Variable(im.cuda(async=True))
        img_clone = img.clone()
        que = Variable(torch.from_numpy(q).long().cuda(async=True))
        ans = Variable(torch.from_numpy(a).cuda(async=True))
        que_len = Variable(torch.from_numpy(q_len).cuda(async=True))

        ans_, att_ = self.VQA_model.forward_pass(img, que, que_len)


        prob_value, ans_index = ans_.data.cpu().max(dim=1)
        _, target_idx = ans.data.cpu().max(dim=1)

        if (target_idx.numpy()[0] == ans_index.numpy()[0]):
            print 'initial (target, ans): ', self.ans_vocab_inv[target_idx.numpy()[0]], self.ans_vocab_inv[ans_index.numpy()[0]]
            success, img = self.perform(im, q, q_len, a)
            return True, success, img
        else:
            return False, False, None

    def perform(self, im, q, q_len, a):
        img_shape = im.cpu().numpy().shape

        img = Variable(im.cuda(async=True))
        img_clone = img.clone()
        que = Variable(torch.from_numpy(q).long().cuda(async=True))
        ans = Variable(torch.from_numpy(a).cuda(async=True))
        que_len = Variable(torch.from_numpy(q_len).cuda(async=True))

        perturb = Variable(torch.Tensor(img_shape[0],img_shape[1],img_shape[2],img_shape[3]).normal_(std=0.1).float().cuda(async=True),requires_grad=True)

        boxmax = Variable(torch.Tensor(1).fill_(im.max()).float().cuda())
        boxmin = Variable(torch.Tensor(1).fill_(im.min()).float().cuda())
        boxmul = (boxmax - boxmin) / 2.
        boxplus = (boxmin + boxmax) / 2.

        #print boxplus, boxmul
        # convert to tanh space
        img = torch.atan((img - boxplus)/boxmul * 0.999999)

        success = False
        iter_ = 0
        img_cv = None

        self.optimizer = optim.Adam([perturb])
        while iter_ < max_inter and success == False:
            iter_ += 1
            update_learning_rate(self.optimizer, iter_)
            newimg = self.tanh(perturb + img) * boxmul + boxplus
            #newimg_arctan = torch.atan((newimg - boxplus)/boxmul * 0.999999)
            l2dist = torch.abs(newimg - (self.tanh(img) * boxmul + boxplus)).sum()

            ans_t, _ = self.VQA_model.forward_pass(newimg, que, que_len)

            real = (ans_t*ans).sum(dim=1)
            other = (((1-ans)*ans_t) - (ans*10000)).max()

            if self.targetted:
                # if targetted, optimize for making the other class most likely
                loss1 = torch.max(Variable(torch.Tensor(1).fill_(0.0).cuda()), other-real+self.confidence)
            else:
                # if untargeted, optimize for making this class least likely.
                loss1 = torch.max(Variable(torch.Tensor(1).fill_(0.0).cuda()), real-other+self.confidence)

            # sum up the losses
            loss2 = l2dist.sum()
            loss1 = (self.scalar_const*loss1).sum()
            loss = loss1 + loss2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            prob_value, ans_index = ans_t.data.cpu().max(dim=1)
            _, target_idx = ans.data.cpu().max(dim=1)
            val1 = torch.abs(self.tanh(perturb) * boxmul + boxplus).sum().data.cpu().numpy()[0]
            if iter_ % 50 == 0:
                print iter_, val1, loss1.data.cpu().numpy()[0], loss2.data.cpu().numpy()[0], self.ans_vocab_inv[target_idx.numpy()[0]], self.ans_vocab_inv[ans_index.numpy()[0]]

            #if(loss2.data.cpu().numpy()[0] < 5000 and
            #            (ans_index.numpy()[0] != target_idx.numpy()[0])):

            if(val1 < 188000.0 and
                (ans_index.numpy()[0] != target_idx.numpy()[0])):
                    # original image
                    img_np2 = self.unorm(img_clone.data).cpu().numpy()
                    img_cv2 = np.transpose(img_np2,(0,2,3,1))
                    img_cv2 = cv2.convertScaleAbs(img_cv2.reshape(448,448,3)*255)
                    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
                    cv2.imshow('original', img_cv2)
                    cv2.waitKey()

                    #perturbed image
                    img_np1 = self.unorm(newimg.data).cpu().numpy()
                    img_cv = np.transpose(img_np1,(0,2,3,1))
                    img_cv = cv2.convertScaleAbs(img_cv.reshape(448,448,3)*255)
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    cv2.imshow('purturbed', img_cv)
                    cv2.waitKey()


                    success = True
        return success, img_cv


class Attacker:
    def __init__(self, VQA_model, targetted=False):

        # save a globle vqa model
        self.VQA_model = VQA_model
        # define an attacker net
        self.attack_model = AttackNet()
        # putting it into train mode
        self.attack_model.train() 
        # transfer to gpus
        self.attack_model.cuda()
        # get all the learnable parameters from it
        self.optimizer = optim.Adam([p for p in 
            self.attack_model.parameters() if p.requires_grad])
        # Define softmax
        self.log_softmax = nn.LogSoftmax().cuda()
        self.scaller_const = Variable(torch.Tensor([10000]).float()).cuda()

        # Define unnormalizer
        self.unorm = utils.UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        # is it targetted
        self.targetted_const = 1
        if targetted == True:
            self.targetted_const = -1


    def perform_validation(self, img, que, que_len, ans):
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
        #a in targetted case is the target but untargetted case is ground truth ans

        # im is torch tensor, converting everything into Variable
        #img = Variable(img.cuda(async=True))
        img_clone = img.clone()
        #que = Variable(torch.from_numpy(que).long().cuda(async=True))
        #ans = Variable(torch.from_numpy(ans).cuda(async=True))
        #que_len = Variable(torch.from_numpy(que_len).cuda(async=True))

        ans_, att_ = self.VQA_model.forward_pass(img, que, que_len)

        self.targetted_const = 1

        prob_value, ans_index = ans_.data.cpu().max(dim=1)
        #ans stores the target index for the attack or ground truth for untargetted
        _, target_idx = ans.data.cpu().max(dim=1)
        '''
        print(self.VQA_model.index_to_answer[target_idx.numpy()[0]],
            self.VQA_model.index_to_answer[ans_index.numpy()[0]])

        img_np1 = self.unorm(img.data).cpu().numpy()
        img_cv = np.transpose(img_np1,(0,2,3,1))
        img_cv = cv2.convertScaleAbs(img_cv.reshape(448,448,3)*255)
        cv2.imshow('purturbed', img_cv)
        cv2.waitKey(0)
        '''
        ####################################################################################
        #This needs to be fixed to also do for targetted.
        #Only untargetted for now
        ############################################## Works only with batch_size = 1 for now. Modify for larger sizes later
        if (target_idx.numpy()[0] == ans_index.numpy()[0]):   #If actual VQA model is right then attack it else dont. #if actual answer same as predicted then attack it
            orig, success, img_cv, loss1, loss2, mean_noise = self.perform(img, que, que_len, ans) ################################################################################################################################################ Check. Perform has grads. Not needed in val
            return True, success, img_cv, loss1, loss2, mean_noise #orig = True as original VQA model is right. success decided by attack. img is perturbed image
            #return True, False, None 


            #Try replacing the above 2 lines with this:
            '''
            #Attack the image now using attention maps
            purturb = self.attack_model(att_)
            #Add noise back to image
            img_ = img_clone + purturb

            #Get answer and attention maps when perturbed image is fed to the VQA network
            ans_t, att_t = self.VQA_model.forward_pass(img_, que, que_len)

            #Compute added noise
            mean_noise = torch.abs(purturb).mean()
            noise = torch.abs(purturb).sum()  #Using L1 for now, see if you wanna use L2 instead.

            # get answer index
            prob_value, ans_index = ans_t.data.cpu().max(dim=1)
            _, target_idx = ans.data.cpu().max(dim=1)

            success = False

            if self.targetted_const == 1:  

                if(noise.data.cpu().numpy()[0] < 26000 and 
                        (ans_index.numpy()[0] != target_idx.numpy()[0])): ###Check   #Condition that noise levels are thresholded and targetted attack is successful

                    #print(self.VQA_model.index_to_answer[target_idx.numpy()[0]],
                        self.VQA_model.index_to_answer[ans_index.numpy()[0]])

                    img_np1 = self.unorm(img_.data).cpu().numpy()
                    img_cv = np.transpose(img_np1,(0,2,3,1))
                    img_cv = cv2.convertScaleAbs(img_cv.reshape(448,448,3)*255)
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    #cv2.imshow('purturbed', img_cv)
                    #cv2.waitKey(0)

                    success = True
            else:
                if(noise.data.cpu().numpy()[0] < 26000 and 
                        (ans_index.numpy()[0] == target_idx.numpy()[0])): ###Check    #Condition that noise levels are thresholded and untargetted attack is successful
                    img_np1 = self.unorm(img_.data).cpu().numpy()
                    img_cv = np.transpose(img_np1,(0,2,3,1))
                    img_cv = cv2.convertScaleAbs(img_cv.reshape(448,448,3)*255)
                    #cv2.imshow('purturbed', img_cv)
                    #cv2.waitKey(100)
                    success = True


            return True, success, img_cv, None, noise, mean_noise
            '''
        else:
            return False, False, None, None, None, None #orig = False as the original VQA model itself misclassifies

    def perform(self, img, que, que_len, ans):
        '''
        img: batch of images
        que: batch of questions
        que_len: batch of question lengths
        ans: batch of answers

        Returns: 
        orig: Whether the VQA models prediction on the benign image is correct
        success: Whether the VQA models was attacked successfully
        img: The noisy perturbed image (back to original scale and channels compatible with cv2)
        loss1: Scaled(lambda) cross entropy loss
        loss2: L1 sum of noise added 
        '''
        # im is torch tensor, converting everything into Variable
        #img = Variable(img.cuda(async=True))
        img_clone = img.clone()
        #que = Variable(torch.from_numpy(que).long().cuda(async=True))
        #ans = Variable(torch.from_numpy(ans).cuda(async=True))
        #que_len = Variable(torch.from_numpy(que_len).cuda(async=True))

	#pdb.set_trace()	
        ans_, att_ = self.VQA_model.forward_pass(img, que, que_len)


        prob_value, ans_index = ans_.data.cpu().max(dim=1)
        #ans stores the target index for the attack or ground truth for untargetted
        _, target_idx = ans.data.cpu().max(dim=1)

        #orig stores whether the original VQA model predicts the answer correctly
        if (target_idx.numpy()[0] == ans_index.numpy()[0]):
            orig = True
        else:
            orig = False

        att_cpu = att_.data.cpu().numpy()
        #Break computational graph to not backprop into the VQA model itself
        att_clone = Variable(torch.from_numpy(att_cpu).cuda(), requires_grad= False).view(que.shape[0], 4096, 14, 14)

        success = False
        iter_ = 0
        img_cv = None


        #Change this to allow for custom learning rates
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 5e-3
        
        while (iter_ < max_inter and (success == False)):

            iter_ += 1
            update_learning_rate(self.optimizer, iter_) #####################################Replace with scheduler instead using decrease_on_plateau

            #Attack the image now using attention maps
            purturb = self.attack_model(att_clone)
            #Add noise back to image
            img_ = img_clone + purturb
            #Get answer and attention maps when perturbed image is fed to the VQA network
            ans_t, att_t = self.VQA_model.forward_pass(img_, que, que_len)

            #Compute targetted/untargetted loss
            nll = -1 * self.targetted_const * self.log_softmax(ans_t) # self.targetted_const is 1 if untargetted -1 if targetted
            #ans in targetted case is the target but untargetted case is ground truth ans. (nll*ans) controls the correct loss
            loss1 = self.scaller_const * (nll*ans).sum(dim=1).mean() 
            #Penalize large noise. Can modify to allow a certain extent of noise
            mean_noise = torch.abs(purturb).mean()
            loss2 = torch.abs(purturb).sum()  #Using L1 for now, see if you wanna use L2 instead.
            loss = loss1 + loss2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # get answer index
            prob_value, ans_index = ans_t.data.cpu().max(dim=1)
            _, target_idx = ans.data.cpu().max(dim=1)

            if self.targetted_const == -1:
                ############################################## Works only with batch_size = 1 for now. Modify for larger sizes later
                if(loss2.data.cpu().numpy()[0] < 26000 and 
                        (ans_index.numpy()[0] != target_idx.numpy()[0])): ###Check   #Condition that noise levels are thresholded and targetted attack is successful

                    print(self.VQA_model.index_to_answer[target_idx.numpy()[0]],
                        self.VQA_model.index_to_answer[ans_index.numpy()[0]])

                    img_np1 = self.unorm(img_.data).cpu().numpy()
                    img_cv = np.transpose(img_np1,(0,2,3,1))
                    img_cv = cv2.convertScaleAbs(img_cv.reshape(448,448,3)*255)
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    cv2.imshow('purturbed', img_cv)
                    cv2.waitKey(0)
                    #cv2.imshow('purturbed', img_cv)
                    #cv2.waitKey(100)

                    success = True
                    break
            else:
                if(loss2.data.cpu().numpy()[0] < 26000 and 
                        (ans_index.numpy()[0] == target_idx.numpy()[0])): ###Check    #Condition that noise levels are thresholded and untargetted attack is successful
                    img_np1 = self.unorm(img_.data).cpu().numpy()
                    img_cv = np.transpose(img_np1,(0,2,3,1))
                    img_cv = cv2.convertScaleAbs(img_cv.reshape(448,448,3)*255)
                    #cv2.imshow('purturbed', img_cv)
                    #cv2.waitKey(100)
                    success = True
                    break
            #print(prob_value, ans_index, target_idx, loss2.data.cpu().numpy(),  loss1.data.cpu().numpy())


            #print(loss2, loss1)
        return orig, success, img_cv, loss1, loss2, mean_noise
