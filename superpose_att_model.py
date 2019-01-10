import torch
import vqa_model
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
		# Load the vqa model
		self.log = torch.load(vqa_model_path)
		self.tokens = len(self.log['vocab']['question']) + 1

		self.vqa_net = torch.nn.DataParallel(vqa_model.Net(self.tokens)).cuda()
		self.vqa_net.eval()
		self.vqa_net.load_state_dict(self.log['weights'])

		self.index_to_answer = {v: k for k, v in self.log['vocab']['answer'].iteritems()}


	def get_vocab(self):
		return self.log['vocab']

	def forward_pass(self, im, q, q_len):
		'''
		Performs forward pass on the image
		'''
		im_features = self.rnet(im) #If RGB image feats received
		#im_features = im #Pre computed resnet features
		ans, att, a = self.vqa_net(im_features, q, q_len)

		return ans, att, a


class AttackNet(torch.nn.Module):
	def __init__(self):
		super(AttackNet, self).__init__()
		self.conv0 = nn.ConvTranspose2d(8192, 2048, 3, stride=2, padding=1
		self.conv1 = nn.ConvTranspose2d(2048, 512, 3, stride=2, padding=1)
		self.conv2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1)
		self.conv3 = nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1)
		self.conv4 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1)	#Generating pixel level map. If you want channel level then make the output 3 channels
		#self.conv = nn.Conv2d(4096,1024,3,  padding=1)
		for m in self.modules():
			if isinstance(m, nn.ConvTranspose2d):
				init.xavier_uniform(m.weight)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, att):
		#att = nn.Tanh()(self.conv(att))
		att = nn.ReLU()(self.conv0(att, output_size=[1, 2048, 28, 28]))
		att = nn.ReLU()(self.conv1(att, output_size=[1, 512, 56, 56]))
		att = nn.ReLU()(self.conv2(att, output_size=[1, 128, 112, 112]))
		att = nn.ReLU()(self.conv3(att, output_size=[1, 32, 224, 224]))
		att = nn.Sigmoid()(self.conv4(att, output_size=[1, 1, 448, 448]))		##Generating pixel level map. If you want channel level then make the output 3 channels
		#Sigmoid used to get superposition weights to be in [0,1]
		return att





