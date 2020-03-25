import os
import sys; sys.path.append("utils")
import cv2
import time; start_time = time.time()
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Import config file
import config

# Import torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torchsummary import summary

# Import custom dataset class
from dataset import SaliencyDataset, get_random_datasets

# Import network model
from spherenet_model import SphereNet 		# New refactored model

from spherenet import SphereConv2D

# Import training function
from train import train

# Import testing function
from test import test_model


def get_time():
	return ("[" + str("{:.4f}".format(time.time() - start_time)) + "]: ")

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage:\n python main.py --train [--restore] \n python main.py --test [<number_of_image] [--plot] \n python main.py --multitest")
		
	if "--train" in sys.argv:	
		
		# Check device o run on
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print(get_time() + "Working on " + str(device))

		# Instantiate model
		model = SphereNet().to(device)
	
		print(get_time() + "Model has been loaded.")
		
		# Weight initialization (in case)
		def weight_init(m):
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, SphereConv2D):
				nn.init.xavier_uniform_(m.weight.data)

		if not "--restore" in sys.argv:
			model.down1.apply(weight_init)
			model.down2.apply(weight_init)
			model.down3.apply(weight_init)
			model.down4.apply(weight_init)
			model.center.apply(weight_init)
			model.up1.apply(weight_init)
			model.up2.apply(weight_init)
			model.up3.apply(weight_init)
			model.up4.apply(weight_init)
			model.end.apply(weight_init)
		else:
			print("Weights not initializd due to restoring task")

		# Optimizer (only requiring grad. parameters)
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4,  momentum=0.9, weight_decay=1e-5)

		# Do training
			
		restore = False
		if "--restore" in sys.argv:
			print("Restoring...")
			restore = True
		
		print(get_time() + "Starting training...")
		model, to_plot = train(model, optimizer, device, num_epochs=config.epochs, restore=restore)
		print(get_time() + "Training has been done.")

		plt.plot(to_plot[0])
		plt.plot(to_plot[1])
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train loss', 'Validation loss'])
		plt.savefig("sphere_loss.png")
		plt.clf()

	elif "--test" in sys.argv:
	
		# Check device o run on
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print(get_time() + "Working on " + str(device))
		
		# Load the model from disk
		PATH = config.model_path
		model = torch.load(PATH)
		print(get_time() + "Model has been loaded.")
		
		# If there is a 3rd par, then the image index to test on should be there
		num = 0
		if len(sys.argv) > 2:
			num = int(sys.argv[2])
		
		# Generate a dataset with new test samples
		saliency_test_set = SaliencyDataset([num],config.test_ipath,config.test_opath, transform = config.trans)

		# generate the corresponding data loader
		sal_test_loader = DataLoader(saliency_test_set, batch_size=1, shuffle=False, num_workers=0)

		# Test the model
		test_model(model, device, sal_test_loader)

		print(get_time() + "Testing has been done.")

	elif "--multitest" in sys.argv:	

		# Check device o run on
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print(get_time() + "Working on " + str(device))
		
		# Load the model from disk
		PATH = config.model_path
		model = torch.load(PATH)		
		print(get_time() + "Model has been loaded.")

		# Generate a dataset with new test samples
		saliency_test_set = SaliencyDataset(range(0,config.test_total),config.test_ipath,config.test_opath, transform = config.trans)

		# generate the corresponding data loader
		sal_test_loader = DataLoader(saliency_test_set, batch_size=1, shuffle=False, num_workers=0)

		# Test the model
		test_model(model, device, sal_test_loader, multitest=True)

		print(get_time() + "Testing has been done.")