import torch
import torch.nn as nn
import torch.nn.functional as F

class LossModule(nn.Module):
	def __init__(self, feature_sizes, feature_num, num_examples, device):
		self.feature_sizes = feature_sizes
		self.feature_num = feature_num
		super(LossModule, self).__init__()
		self.GAP = []
		for kernel_size in feature_sizes:
			self.GAP.append(nn.AvgPool2d(kernel_size))
		self.FC = []
		for f_num in feature_num:
			self.FC.append(nn.Linear(f_num, num_examples).to(device))
		self.linear = nn.Linear(len(feature_sizes) * num_examples, 1)

	def forward(self, features):
		outputs = []
		for i in range(len(features)):
			out = self.GAP[i](features[i])
			out = out.view(out.size(0), -1)
			out = self.FC[i](out)
			out = F.relu(out)
			outputs.append(out)

		return self.linear(torch.cat(outputs, 1))
