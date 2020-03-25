# Import torch library
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torchvision import models

# Basic imports
import os
import sys
import glob
import math
import numpy as np

# Mesh convolutions import
from spherenet import SphereConv2D, SphereMaxPool2D

# Spherical MSE
class SphereMSE(nn.Module):
	def __init__(self, h, w):
		super(SphereMSE, self).__init__()
		pi = 3.1415926
		self.h, self.w = h, w
		weight = torch.zeros(1, 1, h, w)
		theta_range = torch.linspace(0, pi, steps=h + 1)
		dtheta = pi / h
		dphi = 2 * pi / w
		for theta_idx in range(h):
			weight[:, :, theta_idx, :] = dphi * (math.sin(theta_range[theta_idx]) + math.sin(theta_range[theta_idx+1]))/2 * dtheta
		self.weight = Parameter(weight, requires_grad=False)

	def forward(self, out, target):
		return torch.sum((out - target) ** 2 * self.weight) / out.size(0)
	
# Spherical Block = Spherical Conv + Norm + ReLU
class SphereBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True,
				 is_relu=True):
		super(SphereBlock, self).__init__()
		# Spherical Convolution
		self.conv = SphereConv2D(in_channels, out_channels, stride=stride, bias=False)
		# Batch normalization
		self.bn = nn.BatchNorm2d(out_channels, eps=1e-4)
		# ReLU activation
		self.relu = nn.ReLU(inplace=True)
		# If no BN or ReLU indicated, then the step is avoided
		if is_bn is False: self.bn = None
		if is_relu is False: self.relu = None

	def forward(self, x):
		# Convolve input
		x = self.conv(x)
		# Batch normalize if the layer exists
		if self.bn is not None: x = self.bn(x)
		# ReLU if the layer exists
		if self.relu is not None: x = self.relu(x)
		return x

# Spherical Encoder
class SphereEncoder(nn.Module):
	def __init__(self, x_channels, y_channels, kernel_size=3):
		super(SphereEncoder, self).__init__()
		# Calculate padding
		padding = (kernel_size - 1) // 2
		# Block with two Spherical Blocks
		self.encode = nn.Sequential(
			SphereBlock(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
						 groups=1),
			SphereBlock(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
						 groups=1),
		)
		# Spherical pooling
		self.pool = SphereMaxPool2D(stride=2)

	def forward(self, x):
		# Forward input
		y = self.encode(x)
		# Pooling
		y_pooled = self.pool(y)
		# y_pooled = F.max_pool2d(y, kernel_size=2, stride=2)
		return y, y_pooled


# Mesh Decoder
class SphereDecoder(nn.Module):
	def __init__(self, x_channels, y_channels, kernel_size=3):
		super( SphereDecoder, self).__init__()
		# Calculate padding
		padding = (kernel_size - 1) // 2
		# Block with three Spherical Blocks
		self.decode = nn.Sequential(
			SphereBlock(2 * x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1,
						 stride=1, groups=1),	
			SphereBlock(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1,
						 stride=1, groups=1),	
			SphereBlock(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1,
						 stride=1, groups=1),	
		)

	def forward(self, down, x):
		N, C, H, W = down.size()
		# Upsampling
		y = F.upsample(x, size=(H, W), mode='bilinear', align_corners=True )
		y = torch.cat([y, down], 1)
		# Forward input
		y = self.decode(y)
		return y


class SphereNet(nn.Module):
	def __init__(self):
		super(SphereNet, self).__init__()

		self.down1 = SphereEncoder(3, 24, kernel_size=3)  
		self.down2 = SphereEncoder(24, 64, kernel_size=3)  
		self.down3 = SphereEncoder(64, 128, kernel_size=3)  
		self.down4 = SphereEncoder(128, 256, kernel_size=3) 

		self.center = nn.Sequential(
			SphereBlock(256, 256, kernel_size=3, padding=1, stride=1),
		)
		
		self.up4 = SphereDecoder(256, 128, kernel_size=3)  
		self.up3 = SphereDecoder(128, 64, kernel_size=3)  
		self.up2 = SphereDecoder(64, 24, kernel_size=3)  
		self.up1 = SphereDecoder(24, 24, kernel_size=3)  
		
		self.end = SphereConv2D(24, 1, stride=1, bias=True)
		
	def forward(self, x):
		out = x  
		
		down1, out = self.down1(out) 
		down2, out = self.down2(out)  
		down3, out = self.down3(out)
		down4, out = self.down4(out) 

		out = self.center(out)

		out = self.up4(down4, out)
		out = self.up3(down3, out)
		out = self.up2(down2, out)
		out = self.up1(down1, out)

		out = self.end(out)
		out = torch.squeeze(out, dim=1)
		return out	

