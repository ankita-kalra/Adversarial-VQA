import torch
import v_def_vqa_model as vqa_model
from resnet import resnet152
import torch.nn as nn
import torch.nn.init as init


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = resnet152(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer



class VQANet:
	def __init__(self, vqa_model_path = '/home/akalra1/projects/adversarial-attacks/show_ask_answer/pytorch-vqa/logs/2017-08-04_00:55:19.pth'):
		# initialize the resnet
		self.rnet = ResNet().cuda()
		self.rnet.eval()
		for param in self.rnet.parameters():
    			param.requires_grad = False
		# Load the vqa model
		self.log = torch.load(vqa_model_path)
		self.tokens = len(self.log['vocab']['question']) + 1

		self.vqa_net = torch.nn.DataParallel(vqa_model.Net(self.tokens)).cuda()
		#self.vqa_net.eval()
		self.vqa_net.load_state_dict(self.log['weights'])
                for param in self.vqa_net.parameters():
                        param.requires_grad = True
		self.index_to_answer = {v: k for k, v in self.log['vocab']['answer'].iteritems()}


	#def get_vocab(self):
	#	return self.log['vocab']

	def forward(self, im, q, q_len):
		'''
		Performs forward pass on the image
		'''
		im_features = self.rnet(im) #If RGB image feats received
		#im_features = im #Pre computed resnet features
		im_features.requires_grad = True
		ans, att, a, da_dv, da_dq = self.vqa_net(im_features, q, q_len)

		return ans, att, a, da_dv, da_dq

'''
class AttackNet(torch.nn.Module):
	def __init__(self):
		super(AttackNet, self).__init__()
		self.conv1 = nn.ConvTranspose2d(4096, 512, 3, stride=2, padding=1)
		self.conv2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1)
		self.conv3 = nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1)
		self.conv4 = nn.ConvTranspose2d(32, 3, 3, stride=4, padding=1)
		self.conv = nn.Conv2d(4096,1024,3,  padding=1)
		for m in self.modules():
			if isinstance(m, nn.ConvTranspose2d):
				init.xavier_uniform(m.weight)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, att):
		#att = nn.Tanh()(self.conv(att))
		att = nn.Tanh()(self.conv1(att, output_size=[1, 1024, 28, 28]))
		att = nn.Tanh()(self.conv2(att, output_size=[1, 1024, 56, 56]))
		att = nn.Tanh()(self.conv3(att, output_size=[1, 1024, 112, 112]))
		att = self.conv4(att, output_size=[1, 1024, 448, 448])
		#att = nn.Conv2d(3,3, 1)(att, output_size=[1, 1024, 448, 448])
		#return att
		return nn.Tanh()(att)
'''
