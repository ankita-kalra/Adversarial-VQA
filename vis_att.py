
import numpy as np 
import os
import copy
import cv2


import torch
from torch.autograd import Variable
from torchvision import models
import utils 

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

def sent_from_que(que, vocab_dict):
    i = 0
    sent = ''
    while que[i] != 0:
        sent = sent + vocab_dict[que[i]] + ' '
        i += 1
    return sent

def vis_attention(img, q, ans, att_map):
    '''
    Function to visualize the attention maps:
    img: 3 X 448 X 448 
    q: 23
    ans: 1

    returns: att_map over image, questions in english, answers in english
    '''

    q_dict = np.load('q_dict.npy').item()
    a_dict = np.load('a_dict.npy').item()
    unorm = utils.UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


    sent = sent_from_que(q, q_dict)
    anss = (a_dict[ans])

    #Resize att map to full res
    rsz_att_map = cv2.resize(5 * att_map.data.cpu().numpy(), (img.size(2), img.size(2)))    #5 * att values to make maps more salient
    #Convert to 0-255 range
    final_att = np.uint8(255 * rsz_att_map) 


    img_np1 = unorm(img.data).cpu().numpy()
    #COnvert Image to PIL format
    img_cv = np.transpose(img_np1,(1,2,0))
    img_cv = cv2.convertScaleAbs(img_cv.reshape(448,448,3)*255)


    att_over_img = save_class_activation_on_image(img_cv, final_att)
    
    return att_over_img, sent, anss


def save_image(image, path=None):
    '''
    image: 3 X 448 X 448

    saves the image as a png image
    '''

    img_np1 = unorm(image.data).cpu().numpy()
    img_cv = np.transpose(img_np1,(1,2,0))
    img_cv = cv2.convertScaleAbs(img_cv.reshape(448,448,3)*255)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    all_imgs.append(img_cv)
    if path != None:
        cv2.imwrite(path, img_cv)








