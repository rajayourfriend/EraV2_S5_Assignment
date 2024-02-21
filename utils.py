
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

#!pip install torchsummary
from torchsummary import summary

from tqdm import tqdm

def isCudaAvailable():
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)
  return cuda


def getDevice(use_cuda=False):
  return torch.device("cuda" if use_cuda else "cpu")



def getTrainTransforms():
  train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])
  return train_transforms

def getTestTransforms():
  test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
  return test_transforms 




def getTrainDataset(train_transforms):
  train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
  return train_data

def getTestDataset(test_transforms):
  test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
  return test_data




def getTrainLoader(batch_size, train_data):
  kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
  train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
  return train_loader
  
def getTestLoader(batch_size, test_data):
  kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
  test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
  return test_loader



def plotImages(train_loader):
  batch_data, batch_label = next(iter(train_loader))

  fig = plt.figure()

  for i in range(12):
    plt.subplot(3,4,i+1)
    plt.tight_layout()
    plt.imshow(batch_data[i].squeeze(0), cmap='gray')
    plt.title(batch_label[i].item())
    plt.xticks([])
    plt.yticks([])
  return


def printSummary(model, input_size=(1, 28, 28)):
  summary(model, input_size)


def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion, train_acc, train_losses):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))
  return train_acc, train_losses

def test(model, device, test_loader, criterion, test_acc, test_losses):
  model.eval()

  test_loss = 0
  correct = 0

  with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
      data, target = data.to(device), target.to(device)

      output = model(data)
      test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

      correct += GetCorrectPredCount(output, target)


  test_loss /= len(test_loader.dataset)
  test_acc.append(100. * correct / len(test_loader.dataset))
  test_losses.append(test_loss)

  print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
  return test_acc, test_losses


def getOptimizer(model, lr=0.01, momentum=0.9):
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
  return optimizer

def getScheduler(optimizer, step_size=15, gamma=0.1, verbose=True):
  scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma, verbose=verbose)
  return scheduler
  

def plotResults(train_losses, train_acc, test_losses, test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")

