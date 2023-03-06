import random

import numpy as np
import torch.utils.data

from net_instances import *
from utils import progress_bar
import math


class ActiveLearner(object):
	def __init__(self, train_set, test_set, config, net_instance, device):
		self.train_set = train_set
		self.test_set = test_set
		self.labeled_set_idx = None
		self.unlabeled_set_idx = None
		self.train_loader = None
		self.test_loader = None
		self.config = config
		self.net_instance = net_instance
		self.device = device
		self.instantiate_sets()

	def instantiate_sets(self):
		num_workers = self.config["num_workers"]
		batch_size = self.config["batch_size"]
		set_increase = self.config["set_increase"]
		initial_idx = list(range(len(self.train_set)))
		random.shuffle(initial_idx)
		self.labeled_set_idx, self.unlabeled_set_idx = self.get_set([], initial_idx, set_increase)
		self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=False,
													   num_workers=num_workers)
		self.instantiate_train_loader()

	def instantiate_train_loader(self):
		num_workers = self.config["num_workers"]
		batch_size = self.config["batch_size"]
		self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=False,
														num_workers=num_workers, pin_memory=True,
														sampler=torch.utils.data.SubsetRandomSampler(self.labeled_set_idx))

	def get_set(self, labeled_idx, unlabeled_idx, sample_size):
		if sample_size < len(unlabeled_idx):
			my_subset = list(range(len(unlabeled_idx)))
			random.shuffle(my_subset)
			my_subset = np.array(my_subset).astype(int)
			labeled_subset_idx = my_subset[:sample_size]
			unlabeled_subset_idx = my_subset[sample_size:]
			unlabeled_idx = np.array(unlabeled_idx)
			labeled_idx += list(unlabeled_idx[labeled_subset_idx.astype(int)])
			unlabeled_idx = list(unlabeled_idx[unlabeled_subset_idx.astype(int)])
		else:
			labeled_idx += unlabeled_idx
			unlabeled_idx = []
		return labeled_idx, unlabeled_idx

	# Training
	def train_networks(self, epoch):
		print('\nEpoch: %d' % epoch)
		self.net_instance.net.train()
		train_loss = 0
		correct = 0
		total = 0
		for batch_idx, (inputs, targets) in enumerate(self.train_loader):
			inputs, targets = inputs.to(self.device), targets.to(self.device)
			self.net_instance.optimizer.zero_grad()
			outputs, _ = self.net_instance.net(inputs)
			target_loss = self.net_instance.criterion(outputs, targets)
			final_loss = self.final_loss(target_loss)
			final_loss.backward()
			self.net_instance.optimizer.step()

			train_loss += final_loss.item()
			total += targets.size(0)
			predicted = self.get_predicted(outputs)
			correct += predicted.eq(targets).sum().item()

			progress_bar(batch_idx, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
						 % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
		self.net_instance.scheduler.step()

	def get_predicted(self, outputs):
		_, predicted = outputs.max(1)
		predicted_label = predicted.view(predicted.size(0))
		return predicted_label

	def final_loss(self, target_loss):
		return torch.sum(target_loss) / target_loss.size(0)

	def examples_select(self):
		subset_size = self.config["subset_size"]
		set_increase = self.config["set_increase"]
		# Update the sets based on the loss prediction
		subset, _ = self.get_set([], self.unlabeled_set_idx, subset_size)
		# Select examples from unlabeled batch
		sorted_loss_idx = self.examples_eval(subset)
		sorted_subset_idx = torch.tensor(subset)[sorted_loss_idx].numpy()
		selected_set_idx = []
		not_selected_set_idx = []
		if set_increase < len(sorted_subset_idx):
			selected_set_idx = list(sorted_subset_idx[-set_increase:])
			not_selected_set_idx = list(sorted_subset_idx[:-set_increase])
		else:
			selected_set_idx = list(sorted_subset_idx)
		self.labeled_set_idx = selected_set_idx
		# self.labeled_set_idx += selected_set_idx
		if subset_size < len(self.unlabeled_set_idx):
			self.unlabeled_set_idx = self.unlabeled_set_idx[subset_size:] + not_selected_set_idx
		else:
			self.unlabeled_set_idx = self.unlabeled_set_idx + not_selected_set_idx
		self.instantiate_train_loader()

	def examples_eval(self, subset):
		raise NotImplementedError()

	def predict(self):
		self.net_instance.net.eval()
		all_predicted = torch.tensor([]).to(self.device)
		all_targets = torch.tensor([]).to(self.device)
		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(self.test_loader):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs, features = self.net_instance.net(inputs)
				predicted_label = self.get_predicted(outputs)
				all_predicted = torch.cat((all_predicted, predicted_label), 0)
				all_targets = torch.cat((all_targets, targets), 0)
		return all_predicted, all_targets

	def accuracy(self, predicted, targets):
		correct = predicted.eq(targets).sum().item()
		total = targets.size(0)
		acc = 100. * correct / total
		return acc


class RandomActiveLearner(ActiveLearner):
	def __init__(self, train_set, test_set, config, device):
		net_instance = ResNet18NetInstance(config, device)
		super().__init__(train_set, test_set, config, net_instance, device)

	def examples_eval(self, subset):
		my_subset = list(range(len(subset)))
		random.shuffle(my_subset)
		return my_subset


class LossActiveLearner(ActiveLearner):
	def __init__(self, train_set, test_set, config, net_instance, device, feature_sizes, lfeature_num):
		super().__init__(train_set, test_set, config, net_instance, device)
		self.loss_net_instance = LossNetInstance(config, device, feature_sizes, lfeature_num)

	def examples_eval(self, subset):
		num_workers = self.config["num_workers"]
		batch_size = self.config["batch_size"]
		subset_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, num_workers=num_workers,
													sampler=torch.utils.data.SubsetRandomSampler(subset))
		pred_loss = self.loss_predict(subset_loader).cpu()
		sorted_loss_idx = np.argsort(pred_loss)
		return sorted_loss_idx

	def loss_predict(self, trainloader):
		self.net_instance.net.eval()
		self.loss_net_instance.net.eval()
		tot_pred_loss = torch.tensor([]).to(self.device)
		with torch.no_grad():
			for inputs, targets in trainloader:
				inputs = inputs.to(self.device)
				# eval net
				_, features = self.net_instance.net(inputs)
				# To train on different layers of features
				loss_input = []
				for i in range(len(features)):
					if features[i].size(2) in self.loss_net_instance.feature_sizes:
						loss_input.append(features[i])
				pred_loss = self.loss_net_instance.net(loss_input)
				pred_loss = pred_loss.view(pred_loss.size(0))
				tot_pred_loss = torch.cat((tot_pred_loss, pred_loss), 0)
		return tot_pred_loss

	def train_networks(self, epoch):
		print('\nEpoch: %d' % epoch)
		self.net_instance.net.train()
		self.loss_net_instance.net.train()
		train_loss = 0
		correct = 0
		total = 0
		for batch_idx, (inputs, targets) in enumerate(self.train_loader):
			inputs, targets = inputs.to(self.device), targets.to(self.device)
			self.net_instance.optimizer.zero_grad()
			self.loss_net_instance.optimizer.zero_grad()
			# train net
			outputs, features = self.net_instance.net(inputs)
			target_loss = self.net_instance.criterion(outputs, targets)
			# train loss net
			# To train on different layers of features
			loss_input = []
			for i in range(len(features)):
				if features[i].size(2) in self.loss_net_instance.feature_sizes:
					loss_input.append(features[i])

			pred_loss = self.loss_net_instance.net(loss_input)
			pred_loss = pred_loss.view(pred_loss.size(0))
			pred_loss_loss = self.loss_net_instance.criterion(pred_loss, target_loss)
			weight = self.config["loss_net"]['lambda']
			final_loss = torch.sum(target_loss) / target_loss.size(0) + weight * pred_loss_loss

			final_loss.backward()
			self.net_instance.optimizer.step()
			self.loss_net_instance.optimizer.step()
			train_loss += final_loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			progress_bar(batch_idx, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
						 % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
		self.net_instance.scheduler.step()
		self.loss_net_instance.scheduler.step()


class ResNet18LossActiveLearner1(LossActiveLearner):
	def __init__(self, train_set, test_set, config, device):
		net_instance = ResNet18NetInstance(config, device)
		super().__init__(train_set, test_set, config, net_instance, device, [4], [512])


class ResNet18LossActiveLearner2(LossActiveLearner):
	def __init__(self, train_set, test_set, config, device):
		net_instance = ResNet18NetInstance(config, device)
		super().__init__(train_set, test_set, config, net_instance, device, [8, 4], [256, 512])


class ResNet18LossActiveLearner3(LossActiveLearner):
	def __init__(self, train_set, test_set, config, device):
		net_instance = ResNet18NetInstance(config, device)
		super().__init__(train_set, test_set, config, net_instance, device, [16, 8, 4], [128, 256, 512])


class ResNet18LossActiveLearner4(LossActiveLearner):
	def __init__(self, train_set, test_set, config, device):
		net_instance = ResNet18NetInstance(config, device)
		super().__init__(train_set, test_set, config, net_instance, device, [32, 16, 8, 4], [64, 128, 256, 512])


class BaldActiveLearner(ActiveLearner):
	def __init__(self, train_set, test_set, config, device):
		net_instance = BayesianCNNInstance(config, device, 1)
		super().__init__(train_set, test_set, config, net_instance, device)

	def train_networks(self, epoch):
		self.net_instance = BayesianCNNInstance(self.config, self.device, 1)
		super().train_networks(epoch)

	def final_loss(self, target_loss):
		return target_loss

	def examples_eval(self, subset):
		num_workers = self.config["num_workers"]
		batch_size = self.config["batch_size"]
		subset_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, num_workers=num_workers,
													sampler=torch.utils.data.SubsetRandomSampler(subset))
		N = len(subset)
		num_classes = 10
		k = 1
		self.net_instance = BayesianCNNInstance(self.config, self.device, k)
		log_probabilities = torch.empty((N, k, num_classes)).to(self.device)
		self.net_instance.net.eval()
		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(subset_loader):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs, _ = self.net_instance.net(inputs)
				lower = batch_idx * batch_size
				upper = min(lower + subset_loader.batch_size, N)
				log_probabilities[lower:upper].copy_(outputs.double(), non_blocking=True)

			scored_subset = self.get_bald_scores(log_probabilities)
		return scored_subset[1]

	def get_predicted(self, outputs):
		predicted = (torch.logsumexp(outputs.double(), dim=1) - math.log(1))
		_, predicted = predicted.max(1)
		return predicted

	def get_bald_scores(self, log_probabilities: torch.Tensor):
		scores = -self.conditional_entropy(log_probabilities)
		scores += self.entropy(log_probabilities)

		subset_scores, subset_indices = torch.topk(scores, len(scores))

		return subset_scores.tolist(), subset_indices.tolist()

	def conditional_entropy(self, log_probabilities: torch.Tensor) -> torch.Tensor:
		N, K, C = log_probabilities.shape
		nats = log_probabilities * torch.exp(log_probabilities)
		entropies = (-torch.sum(nats, dim=(1, 2)) / K)
		return entropies

	def entropy(self, log_probabilities: torch.Tensor) -> torch.Tensor:
		N, K, C = log_probabilities.shape
		mean_log_probabilities = torch.logsumexp(log_probabilities, dim=1) - math.log(K)
		nats = mean_log_probabilities * torch.exp(mean_log_probabilities)
		entropies = (-torch.sum(nats, dim=1))
		return entropies
