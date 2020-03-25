# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils import data as tdata
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torchsummary import summary

from collections import defaultdict
import copy
import time
import os
import glob
import sys
import numpy as np
import math
import matplotlib.pyplot as plt

# Import custom dataset class and custom generator
from dataset import SaliencyDataset, get_random_datasets

# Import config file
import config

# Import MSE
from spherenet_model import SphereMSE

# Save loss info
train_loss = []
val_loss = []

# Print one predictione xample (to test each iteration how it is going...)
# A dirctory called temp/ is required
def print_one_ex(model, sal_test_loader, device, epoch, with_label=None):
	# Get sample (only 1)
	inputs, labels = next(iter(sal_test_loader))
	
	# Prepare data
	inputs = inputs.to(device)
	if with_label is not None:
		labels = labels.to(device)

	# Predict
	pred = model(inputs)
	
	# Squeeze extra dims
	pred = np.squeeze(np.array(pred[0].detach().cpu()))
	if with_label is not None:
		label = np.squeeze(np.array(labels[0].detach().cpu()))

	# Clip
	pred = np.clip(pred, 0, 1)
		
	# Results
	if with_label is not None:
		cm = np.corrcoef(pred.flat, label.flat)[0,1]
		print("Correlation = " + str(cm))	

	plt.imshow(pred, cmap='gray')
	plt.axis('off')
	plt.savefig("temp/figure" + str(epoch) + ".png", bbox_inches='tight')
	plt.clf()
		
	if with_label is not None and epoch == 0:
		plt.imshow(np.squeeze(labels[0].cpu()), cmap='gray')
		# plt.colorbar()
		plt.axis('off')
		plt.savefig("temp/gt.png", bbox_inches='tight')
		plt.clf()
		

# Calculate loss and add it to metrics
def calc_loss(pred, target, metrics, mode, model=None, bce_weight=0.5):
	loss_criterion = SphereMSE(pred.size(-2), pred.size(-1)).cuda()
	loss = loss_criterion(np.squeeze(pred), np.squeeze(target)) 
	if mode == 'train':
		metrics['train_loss'] += loss.data.cpu().numpy() * target.size(0)
	else:
		metrics['val_loss'] += loss.data.cpu().numpy() * target.size(0)
	
	return loss

# Print metrics after each iteration
def print_metrics(metrics, epoch_samples, mode):
	outputs = []
	for k in metrics.keys():
		if mode in k:
			outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
	if mode == 'train':
		print("{}: {}".format("TRAIN", ", ".join(outputs)))
	else:
		print("{}: {}".format("VAL", ", ".join(outputs)))

# Train the model		
def train(model, optimizer, device, num_epochs=25, restore=False):
	best_model_wts = copy.deepcopy(model.state_dict())
	best_loss = 1e10
	
	# Generate a dataset with one sample (to check the performance each iteration)
	saliency_test_set = SaliencyDataset([21],config.test_ipath,None,transform = config.trans)
	sal_test_loader = DataLoader(saliency_test_set, batch_size=1, shuffle=False, num_workers=0)

	# Initial epoch
	init = 0
	# Batch size
	batch_size = config.batch_size
	
	# Restoring an checkpoint
	if restore:
		checkpoint = torch.load(config.ckpt_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		init = int(checkpoint['epoch']) + 1
		best_loss = checkpoint['loss']

	# Iterate
	for epoch in range(init, num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 7)

		# Time measure
		since = time.time()

		# Train mode
		model.train()

		# Current metrics
		metrics = defaultdict(float)
		
		# Current samples
		epoch_samples = 0	
			 	
		# Randomize elements
		saliency_train_set, saliency_val_set = get_random_datasets(config.total,
				config.train,
				config.val,
				config.ipath,
				config.opath,
				config.trans)
				
		# Build dataloaders
		dataloader = DataLoader(saliency_train_set, batch_size=batch_size, shuffle=True, num_workers=0)
		sal_dataloader = DataLoader(saliency_val_set, batch_size=batch_size, shuffle=True, num_workers=0)
	
		# Iterate dataloaders
		for inputs, labels in dataloader:
			# Send data to device
			inputs = inputs.to(device)
			labels = labels.to(device)

			# Forward inputs and update network
			outputs = model(inputs)
			loss = calc_loss(outputs, labels, metrics, 'train', model)	
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			# Statistics
			epoch_samples += inputs.size(0)
			print("Sample " + str(epoch_samples) + "/" + str(len(dataloader) * batch_size), end="\r")

		# Print epoch metrics
		print()
		print_metrics(metrics, epoch_samples, 'train')
		
		# Append metrics
		train_loss.append(float(metrics['train_loss'] / epoch_samples))	
				
		# Eval mode
		model.eval()
		
		# Validation
		epoch_samples = 0
		for inputs, labels in sal_dataloader:
			# Send data to device
			inputs = inputs.to(device)
			labels = labels.to(device)

			# Forward inputs (WITH NO UPDATING)
			outputs = model(inputs)
			loss = calc_loss(outputs, labels, metrics, 'val', model)	

			# Statistics
			epoch_samples += inputs.size(0)
			print("Sample " + str(epoch_samples) + "/" + str(len(sal_dataloader) * batch_size), end="\r")

		# Print epoch metrics
		print()
		print_metrics(metrics, epoch_samples, 'val')
		
		# Append metrics
		epoch_loss = metrics['val_loss'] / epoch_samples	
		val_loss.append(float(metrics['val_loss'] / epoch_samples))
		
		# Deep copy the model if it is better
		if epoch_loss < best_loss:
			print("Saving best model...")
			best_loss = epoch_loss
			best_model_wts = copy.deepcopy(model.state_dict())
			torch.save(model, config.model_path)
			
		# Checkpointing (each iteraton)
		torch.save({
			'epoch': epoch,
			'model_state_dict': best_model_wts,
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': epoch_loss
			}, config.ckpt_path)
		
		# Print one example predicted with current model
		# print_one_ex(model, sal_test_loader, device, epoch, with_label=None)
		
		# Elapsed time 
		time_elapsed = time.time() - since
		print('Elapsed time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		
	# End of the training	
	print('[END OF THE TRAINING] - Best val loss: {:4f}'.format(best_loss))

	# Load best model weights
	model.load_state_dict(best_model_wts)
	
	# Save final model
	torch.save(model, config.model_path)
	
	# Return model and metrics data
	return model, [train_loss, val_loss]
	

