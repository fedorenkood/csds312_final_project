"""
Simple Template code to get pytorch model to run

This file is for testing purposes Dataloading & optimization

Dataloader with CIFAR10:
https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

... and here for dataloader with MNIST. They include information on normalization params:
https://nextjournal.com/gkoehler/pytorch-mnist

# See here a similar guide on Pytorch Dataloaders:
# https://blog.paperspace.com/dataloaders-abstractions-pytorch/
"""

import torch
import torchvision
import torchvision.transforms as transforms
from code.models.simpleCNN import simpleCNN
import torch.optim as optim
import torch.nn as nn


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Normalize dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    # you'll need to change this for MNIST (0-9) or Imagenet
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = simpleCNN()

    criterion = nn.CrossEntropyLoss()

    # momentum is for momentum SGD. See here: https://paperswithcode.com/method/sgd-with-momentum
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # NOTE: Remmember to change the number of epochs this should run for
    for epoch in range(0, 5):

        total_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch}, loss: {total_loss}')

    #save the model in saved_models dir !!
    #format: model_dataset. Eg) Resnet10_CIFAR10
    torch.save(net.state_dict(), './saved_models/saved_model.pth')


# we need to add an if-clause here for Windows parallelism
if __name__ == '__main__':
    main()