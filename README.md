
# EraV2_S5_Assignment

Here the terms functionalities and components are used synonymously.

## Files Description

### S5.ipynb

  File S5.ipynb is the starting point where the execution of the notebook is possible. It imports and invokes the functionalities from other python files such as model.py and utils.py.

### model.py

  File model.py is the one where the Network architecture or AI model that works as neural network system that gets trained and can be further used during inference to test.

### utils.py

  File utils.py is the one where the utilities such as plotting, dataset handling, transformations on data, training and testing functionalities are implemented.


## Components Explanation of S5.ipynb

### main

  There is no main function defined here. But this file can be considered as the main execution / starting point of all. It does first import of required libraries including the model and utils that are user defined here. By invoking **isCudaAvailable**(), it gets to know whether GPU is available or not. Based on this info, device will be prepared.
  Dataset shall be obtained by invoking **getTrainDataset**() with prior obtaining trainforms by invoking **train_transforms**() that are meant for training purposes. In the same way, Dataset shall be obtained by invoking **getTestDataset**() with prior obtaining trainforms by invoking **test_transforms**() that are meant for testing purposes. train_loader and test_loader can be obtained by invoking **getTrainLoader**() and **getTestLoader**() with an intended batch size value.
  Sample images of training can be displayed by invoking **plotImages**().
  Device that is going to be used for training as well as testing can be obtained by invoking **getDevice**().
  A model is prepared from network Net2's instance and attaching it to the device obtained. This model's summary shall be displayed if interested to view it.
  Using optimizer and scheduler obtained from **getOptimizer**() and **getScheduler**() respectively, the training is conducted by invoking the function **train**(). In the same way, the testing is performed by the function **test**().
  End results of training and testing can be visualized by invoking **plotResults**().
  
## Components Explanation of model.py

### Constructor of Net2 Class

  The class constructor of network is named as __init__. Here the individual layers are mentioned with input and output channel information along with number of kernels in the case of Convolution layers. In the case of fully connected layers, the number of input and output neurons are mentioend. 
  

### Forward Propagation

  The functionality of forward propagation is implemented in the function **forward**(). How the network is prepared by arranging the various layers in sequence is done here.

## Components Explanation of utils.py

### Dataset

  The dataset used here is MNIST and is downloaded from pytorch's torchvision. Here dataset undergoes various transformations that are explained below and being used for training. Even testing also needs some data transformations. **getTrainDataset**() and **getTestDataset**() are the functions that do it.

### Transformations

  Various transformations used on the train dataset are CenterCrop, Resize, RandomRotation, ToTensor and Normalize. **getTrainTransforms**() is the function that does it.
  Various transformations used on the test dataset are ToTensor and Normalize. **getTestTransforms**() is the function that does it.

### Dataloading

  DataLoader wraps an iterable around the Dataset to enable easy access to the samples. **getTrainLoader**() and **getTestLoader**() are the functions that do it.

### CUDA Availability

  The functionality of whether we can avail CUDA i.e., GPU or not can be retrieved here. **isCudaAvailable**() is the function that does it.
  
### Device

  The device whether GPU or CPU that is going to be used during training and testing not can be retrieved here. **getDevice**() is the function that does it.
  
### Plotting

  Using pyplot, sample images that used during training are being plotted. **plotImages**() is the function that does it.
  Using pyplot, the result of training and validation are being plotted. **plotResults**() is the function that does it.

### Summary of Model

  The summary of the model is displayed here. **printSummary**() is the function that does it.
  
### Prediction Count

  The correct prediction count is being returned from here. **GetCorrectPredCount**() is the function that does it.

### Optimizer

  The SGD (Stochastic Gradient Descent) optimizer can be retrieved here. **getOptimizer**() is the function that does it.
  
### Scheduler

  The LR (Learning Rate based) scheduler can be retrieved here. **getScheduler**() is the function that does it.
  
### Training

  The important part of AI system preparation, i.e., training can be achieved by this. It returns consolidated train_acc(training accuracy) and train_losses(training losses). **train**() is the function that does it.
  
### Testing

  The important part of AI system preparation, i.e., testing can be achieved by this. It returns consolidated test_acc(testing accuracy) and test_losses(testing losses). **test**() is the function that does it.
