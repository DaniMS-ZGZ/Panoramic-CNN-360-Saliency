# Panoramic-CNN-360-Saliency
Code and models for Panoramic CNN 360º Saliency: [include URL when ready]

## teaser here

## Authors

Daniel Martin - http://webdiis.unizar.es/~danims/  
Ana Serrano - http://webdiis.unizar.es/~aserrano/  
Belen masia - http://webdiis.unizar.es/~bmasia/  

All the authors are part of the [Graphics & Imaging Lab](https://graphics.unizar.es)

## Workshop info

This paper was submitted to BLA BLAH

# How to use our model
Our model was created and run over Anaconda. We provide a ```.yml``` file with all the dependencies installed. To create an Anaconda environment from our configuration .yml file, just run the following command in Anaconda:

```
conda env create --name <envname> --file=unet_gpu.yml
```

A new Anaconda environment will be created with all the necessary dependencies. We are using the following main packages:

- ```PyTorch 1.2.0 w/ CUDA```
- ```torchvision 0.4.0```
- ```cudatoolkit 10.0.130```
- ```opencv 4.1.2```

However, make sure you have installed all dependencies in the ```.yml```` file.

## Training process
You can train our model with your own data. You will need to modify the ```config.py``` file:

- ```total``` is the total number of images used in training process.
- ```train``` is the number of images used for training.
- ```val``` is the number of images used for validation.
- ```batch_size``` is the size of the batch for the training process.
- ```epochs``` is the number of epochs in the training process.

After your configuration is done, you can run the training process.

```
python main.py --train
```

Note that you will need an aditional directory, ```/checkpoints/```, where a ```.tar``` file will be saved each epoch. The model will be updated in ```/models/``` each time an iteration has better validation results.

### Restoring from a checkpoint

If you had to stop your training process, you can restore it wherever you left it. You just have to make sure your model and checkpoint file is in the corresponding folders, and run:

```
python main.py --train --restore
```

## Testing process
You can test our model over one single image or run the testing process over all images in a directory. In both cases, you have to modify the ```config.py``` file:

- ```test_total``` indicates how many images are there in your directory. *Note: You only have to modify this value if you are using the multiple image testing*
- ```test_ipath``` is the directory where your images to be predicted are located.
- ```test_opath``` is the directory where the ground-truth maps of your images to be predicted are located. *Note: This is helpful to compare the result of the network with the corresponding GT*
- ```test_save_path``` is the directory where the predicted saliency maps will be stored.

After your configuration is done, you may choose whether you want to test with a single image or multiple images. Note that, in both cases, results will be saved in the directory you wrote in the ```config.py``` file.

### Single image prediction
To predict the saliency map for a single image, you just have to run:
```
python main.py --test <id>
```

### Directory prediction
```
python main.py --multitest
```
