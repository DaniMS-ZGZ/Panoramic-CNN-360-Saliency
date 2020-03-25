import os
import cv2
import random
import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt

# Import config file
import config

# Import torch utilities
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

# Generate two random datasets (train and val)
def get_random_datasets(total, train, val, ipath, opath, trans=None):
	r = range(0,total)
	to_train = random.sample(r, train)
	r = [x for x in r if (x not in to_train)]
	to_val = random.sample(r, val)
	train_dataset = SaliencyDataset(to_train, ipath, opath, trans)
	val_dataset = SaliencyDataset(to_val, ipath, opath, trans)
	return train_dataset, val_dataset

# Dataset class
class SaliencyDataset(Dataset):
	def __init__(self, indexes, input_path, sal_path, transform=None):
		self.input_path = input_path			# Input path
		self.sal_path = sal_path				# Salmaps path
		self.transform = transform				# Transformations to image
		
		self.inputs = []						# Path to dataset images (so they are not loaded in batch)
		self.salmaps = []						# Path to dataset salmaps (so they are not loaded in batch)
		
		for file_name in os.listdir(self.input_path):
			self.inputs.append(os.path.join(self.input_path, file_name))
		
		if self.sal_path is not None:
			for file_name in os.listdir(self.sal_path):
				self.salmaps.append(os.path.join(self.sal_path, file_name))	
		
		# Crop the dataset to a pre-fixed subset
		self.inputs = [self.inputs[i] for i in indexes]	
		if self.sal_path is not None:
			self.salmaps = [self.salmaps[i] for i in indexes]	
		else:
			self.salmaps = None
						
	def __len__(self):
		return len(self.inputs)

	def __getitem__(self, idx):
		
		# Load i-th image
		image = cv2.imread(self.inputs[idx],cv2.IMREAD_COLOR)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (256,128), interpolation=cv2.INTER_AREA)
		image = image.astype(np.float32) / 255.0

		if self.salmaps is not None:
			# Load the i-th salmap
			salmap = cv2.imread(self.salmaps[idx],cv2.IMREAD_GRAYSCALE)
			salmap = cv2.resize(salmap, (256,128), interpolation=cv2.INTER_AREA)
			salmap = salmap.astype(np.float32)/ 255.0
		else:
			# If salmap is not defined, return image but do not use it
			salmap = image
			
		# Apply any other transforms to image
		if self.transform:
			image = self.transform(image)
			if self.salmaps is not None:
				salmap = self.transform(salmap)

		# Return item
		if salmap is not None:
			return [image, salmap]

		