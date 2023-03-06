'''
Author: Zachary LeClaire

This file trains training. You can feed args in command line

Training & Resnet model code based on this tutorial: https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
The article above describes how to make Resnet 34. This is outside of the scope of our papers, so I just extended it & added more
architectures in resnet.py and Lenet.py

https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320

If you don't have CUDA and don't wanna wait like a million years to run it on your CPU, I made this colab environment:
https://colab.research.google.com/drive/1_jg82H_AAEVB8zOMou33IbE8YpoXTTKQ?usp=sharing
'''

import argparse
import numpy as np
import gc
import os


import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

import sys
sys.path.append("./code/models")

from resnet import Resnet18, Resnet34, Resnet50
from lenet import LeNet5

"""
Data loader function

:param data_dir: path for data
:param batch_size: 
:param random_seed: seed for shuffling RNG
:param valid size: float param for data splitting (cross val)
:param shuffle: bool flag to shuffle data
:param test: bool flag for whether test data or not
:return:
"""


def cifar_data_loader(data_dir, batch_size, random_seed=420, valid_size=0.1, shuffle=True, test=False):
    # normalize dataset
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    if test:
        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
        return data_loader

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)

def MNIST_data_loader(data_dir, batch_size, random_seed=420, valid_size=0.1, shuffle=True, test=False):
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor()])

    if test:
        valid_dataset = datasets.MNIST(root='mnist_data',
                                   train=False,
                                   transform=transform)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        return valid_loader
    # download and create datasets for training:

    train_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    valid_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)



if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #arg parsing
    parser = argparse.ArgumentParser(description='Trains training for adversarial testing')
    #parser.add_argument('data_set', metavar='DATA_SET', type=str, help='dataset name (MNIST or CIFAR-10)')
    parser.add_argument('arch', metavar='arch', type=str, help='model architecture (Resnet18, Resnet34)')
    parser.add_argument('epochs', metavar='epoch', type=int, help='number of epochs model runs for')
    parser.add_argument('dataset', metavar='dataset', type=str, help='which data set: cifar10 or mnist')
    parser.add_argument('learning_rate', metavar='lr', type=float, help='learning rate of model')
    parser.add_argument('save_path', metavar = 'savepath', type=str, help='Path for saved model')
    parser.set_defaults(cv=True, gain_ratio=False)
    args = parser.parse_args()


    #check if save path is valid before running program:
    if not os.path.isdir(args.save_path):
        print("No such file '{}'".format(), file=sys.stderr)


    # hyper parameters
    num_epochs = args.epochs
    batch_size = 16
    num_classes = 10

    learning_rate = args.learning_rate


    # load data for model
    if(args.dataset == "cifar10"):
        # CIFAR10 dataset
        train_loader, valid_loader = cifar_data_loader(data_dir='./data', batch_size=64)

        test_loader = cifar_data_loader(data_dir='./data', batch_size=64, test=True)
        num_classes = 10
    elif (args.dataset == "mnist"):
        # MNIST dataset
        train_loader, valid_loader = MNIST_data_loader(data_dir='./data', batch_size=64)

        data_loader = MNIST_data_loader(data_dir='./data', batch_size=64, test=True)
        num_classes = 10


    # get model:
    if args.arch == "resnet18":
        model = Resnet18(num_classes)
    elif args.arch == "resnet34":
        model = Resnet34(num_classes)
    elif args.arch == "resnet50":
        model = Resnet50(num_classes)
    elif args.arch == "lenet5":
        model = LeNet5(num_classes)

    if torch.cuda.is_available():
        model.cuda()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9)

    # Train the model
    total_step = len(train_loader)

    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(train_loader):

            # Move tensors to the configured device (if CUDA)
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))

        # Validation, so we turn off grad
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

    # testing model:
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))
    torch.save(model.state_dict(), args.save_path)