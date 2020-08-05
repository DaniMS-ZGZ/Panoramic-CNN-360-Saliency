from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import cv2
import numpy as np


onlyfiles = [f for f in listdir("img") if isfile(join("img", f))]
imgs = []
maps = []


for i in range(0,len(onlyfiles),2):
	imgs.append("img/" + onlyfiles[i])
	maps.append("img/" + onlyfiles[i+1])

		
print(imgs)
print(maps)


def noisy(noise_typ,image):
	if noise_typ == "gauss":
		row,col,ch= image.shape
		mean = 50
		var = 5
		sigma = var**0.5
		gauss = np.random.normal(mean,sigma,(row,col,ch))
		gauss = gauss.reshape(row,col,ch)
		noisy = image + gauss
		return noisy
	elif noise_typ == "s&p":
		row,col,ch = image.shape
		s_vs_p = 0.8
		amount = 0.04
		out = np.copy(image)
		# Salt mode
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt))
			for i in image.shape]
		out[coords] = 1

		# Pepper mode
		num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper))
			for i in image.shape]
		out[coords] = 0
		return out
	elif noise_typ == "poisson":
		vals = len(np.unique(image))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy
	elif noise_typ =="speckle":
		row,col,ch = image.shape
		gauss = np.random.randn(row,col,ch)
		gauss = gauss.reshape(row,col,ch)        
		noisy = image + image * gauss
		return noisy


opath = "out/S"
spath = "sal/S"
n_it = 0
for i, s in zip(imgs,maps):
	
	image = cv2.imread(i,cv2.IMREAD_COLOR)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (1024,512), interpolation=cv2.INTER_AREA)
	# image = image.astype(np.float32) / 255.0
	
	
	salmap = cv2.imread(s,cv2.IMREAD_GRAYSCALE)
	salmap = cv2.resize(salmap, (1024,512), interpolation=cv2.INTER_AREA)
	# salmap = salmap.astype(np.float32)/ 255.0
	salmap = salmap[10:-10,10:-10]

	# Flip
	cv2.imwrite(opath + str(n_it) + "_01.jpg",image)
	cv2.imwrite(spath + str(n_it) + "_01.jpg",salmap)
	
	flipVertical = cv2.flip(image, 0)
	cv2.imwrite(opath + str(n_it) + "_02.jpg",flipVertical)
	flipVertical = cv2.flip(salmap, 0)
	cv2.imwrite(spath + str(n_it) + "_02.jpg",flipVertical)
	
	flipHorizontal = cv2.flip(image, 1)
	cv2.imwrite(opath + str(n_it) + "_03.jpg",flipHorizontal)
	flipHorizontal = cv2.flip(salmap, 1)
	cv2.imwrite(spath + str(n_it) + "_03.jpg",flipHorizontal)
	
	flipBoth = cv2.flip(image, -1)
	cv2.imwrite(opath + str(n_it) + "_04.jpg",flipBoth)
	flipBoth = cv2.flip(salmap, -1)
	cv2.imwrite(spath + str(n_it) + "_04.jpg",flipBoth)
	
	# Noise
	
	n = noisy("gauss", image)
	cv2.imwrite(opath + str(n_it) + "_05.jpg",n)
	cv2.imwrite(spath + str(n_it) + "_05.jpg",salmap)
	
	n = noisy("s&p", image)
	cv2.imwrite(opath + str(n_it) + "_06.jpg",n)
	cv2.imwrite(spath + str(n_it) + "_06.jpg",salmap)
	
	n = noisy("speckle", image)
	cv2.imwrite(opath + str(n_it) + "_07.jpg",n)
	cv2.imwrite(spath + str(n_it) + "_07.jpg",salmap)
	
	n = noisy("poisson", image)
	cv2.imwrite(opath + str(n_it) + "_08.jpg",n)
	cv2.imwrite(spath + str(n_it) + "_08.jpg",salmap)

	n_it = n_it + 1
	print("Done " + i)