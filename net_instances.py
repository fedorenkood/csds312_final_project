import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import models.resnet as resnet
from models import bayesian
from models.lossnet import LossModule


from torch.nn import functional as F

class NetInstance(object):
	def __init__(self, config, net_name, device):
		self.config = config
		self.name = net_name
		self.device = device
		self.net = None
		self.criterion = None
		self.optimizer = None
		self.scheduler = None
		self.instantiate(config[self.name])

	def get_net(self, config):
		return "Call Child"

	def instantiate(self, config):
		momentum = config["momentum"]
		weight_decay = config["weight_decay"]
		lr = config["learn_rate"]
		milestones = config["milestones"]
		#########
		# Model
		#########
		self.get_net(config)
		self.net = self.net.to(self.device)
		if self.device == 'cuda':
			self.net = torch.nn.DataParallel(self.net)
			cudnn.benchmark = True

		#########
		# Model Optimizer
		#########
		self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones)


class LossNetInstance(NetInstance):
	def __init__(self, config, device, feature_sizes, feature_num, name='loss_net'):
		self.feature_sizes = feature_sizes
		self.feature_num = feature_num
		super().__init__(config, name, device)

	def loss_net_loss(self, pred_loss, target_loss):
		margin = self.config['loss_net']["margin"]
		pred_loss_p1 = pred_loss[[i for i in range(0, pred_loss.size(0), 2)]]
		pred_loss_p2 = pred_loss[[i for i in range(1, pred_loss.size(0), 2)]]
		target_loss_p1 = target_loss[[i for i in range(0, pred_loss.size(0), 2)]]
		target_loss_p2 = target_loss[[i for i in range(1, pred_loss.size(0), 2)]]
		loss = torch.zeros(pred_loss.size(0) // 2).to(self.device)
		for i in range(pred_loss.size(0) // 2):
			sign = 1 if target_loss_p1[i] > target_loss_p2[i] else -1
			loss[i] = (sign * (pred_loss_p1[i] - pred_loss_p2[i]) + margin)
		out = torch.sum(torch.clamp(loss, min=0)) / loss.size(0)
		return out

	def get_net(self, config):
		batch_size = self.config["batch_size"]
		self.net = LossModule(feature_sizes=self.feature_sizes, feature_num=self.feature_num, num_examples=batch_size,
							  device=self.device)
		self.criterion = self.loss_net_loss


class ResNet18NetInstance(NetInstance):
	def __init__(self, config, device, name='res_net_18'):
		super().__init__(config, name, device)

	def get_net(self, config):
		self.net = resnet.ResNet18(config['in_dim'])
		self.criterion = nn.CrossEntropyLoss(reduction='none')

	def resnet_criterion(self, predicted, targets):
		def_criterion = nn.CrossEntropyLoss(reduction='none')
		target_loss = def_criterion(predicted, targets)
		# return torch.sum(target_loss) / target_loss.size(0)
		return target_loss


class BayesianCNNInstance(NetInstance):
	def __init__(self, config, device, k, name='bayesian_cnn'):
		self.k = k
		super().__init__(config, name, device)

	def get_net(self, config):
		self.net = bayesian.BayesianCNN(self.k)
		self.criterion = self.bayesian_criterion

	def bayesian_criterion(self, predicted, target):
		return F.nll_loss(predicted.squeeze(1), target)
