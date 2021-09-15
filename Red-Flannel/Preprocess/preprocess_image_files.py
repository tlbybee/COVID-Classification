#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:20:35 2021

"""


import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time; _START_RUNTIME = time.time()
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models



# set seed
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# Define data path
IMAGE_PATH = "../../images/data_split/train"

WEIGHT_PATH = os.path.join(IMAGE_PATH, '..', 'resnet18_weights_9.pth')

"""
Show the images in 
"""

def imshow(img, title):
    npimg = img.numpy()
    plt.figure(figsize=(15, 7))
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def show_batch_images(dataloader):
    images, labels = next(iter(dataloader))
    img = torchvision.utils.make_grid(images, padding=25)
    imshow(img, title=["COVID-19"  if x==0 else "Normal" if x==1 else "Pnuemonia"
                                 for x in labels])


"""
Preprocess

"""


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


dataset = torchvision.datasets.ImageFolder(IMAGE_PATH,transform=transform)
#print(dataset)
print(f"Total images in dataset {len(dataset)}")

dataloader = torch.utils.data.DataLoader(dataset, #num_workers=4,
                                         batch_size=32, shuffle=True)


print('\nImage Classes ' ,dataset.class_to_idx, '\n')


for i in range(2):
    show_batch_images(dataloader)

"""
Function to split the dataset to Train, Test and Validate
"""
def load_data(dataset, split_size = 0.7): 
   
    """
    Split data into 
    Train set       - 70%
    Validation set  - 10%
    Test set        - 20%
    """

    np = len(dataset)
    s1 = int(split_size * np)
    s2 = int((split_size + 0.1) *  np)


    train_set = torch.utils.data.Subset(dataset, range(s1))  
    val_set = torch.utils.data.Subset(dataset, range(s1, s2))  
    test_set = torch.utils.data.Subset(dataset, range(s2, np))  
    
    print(f"Total images in Train Set {len(train_set)}")
    print(f"Total images in Test Set {len(test_set)}")
    print(f"Total images in Validation Set {len(val_set)}")
    
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=32,
                                               shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=32,
                                             shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=32,
                                              shuffle=False)
    
    return train_loader,val_loader,test_loader
   
    
"""
Load Data
"""

train_loader, val_loader, test_loader = load_data(dataset)   

alexnet = models.alexnet(pretrained=True)

alexnet.eval()

print(alexnet)

"""
The standard Alexnet model has 1000 out features for the last Linear layer.
Since the data set only has 3 classes, change the last layer to have 3 outs
"""
alexnet.classifier[6] = torch.nn.Linear(alexnet.classifier[6].in_features, 4, bias=True)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)


n_epochs = 1

def train_model(model, train_dataloader, n_epoch=n_epochs, optimizer=optimizer, criterion=criterion):
    import torch.optim as optim

    # prep model for training
    model.train() 
    
    
    for epoch in range(n_epoch):
        curr_epoch_loss = []
        for data, target in train_dataloader:
            # your code here
            
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            curr_epoch_loss.append(loss.cpu().data.numpy())
        print(f"Epoch {epoch}: curr_epoch_loss={np.mean(curr_epoch_loss)}")
    
    return model



#alexnet = train_model(alexnet, train_loader)










