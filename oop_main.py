import argparse
import json

import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from active_learner import *
from net_instances import *


def select_classes(dataset, include_list):
    if type(dataset.targets) is not torch.Tensor:
        dataset.targets = torch.tensor(dataset.targets)
    idx = dataset.targets == -1
    for label in include_list:
        idx |= dataset.targets == label
    dataset.data = dataset.data[idx.numpy().astype(bool)]
    dataset.targets = dataset.targets[idx]
    return dataset


def cifar10_data():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    include_list = [1, 7]
    trainset = select_classes(torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train), include_list)

    testset = select_classes(torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test), include_list)

    trainset.classes = list(np.array(trainset.classes)[include_list])
    classes = trainset.classes
    return trainset, testset, classes


def mnist_data():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
         (0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
         (0.1307,), (0.3081,))
    ])

    include_list = [5, 7]
    trainset = select_classes(torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train), include_list)

    testset = select_classes(torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test), include_list)

    trainset.classes = list(np.array(trainset.classes)[include_list])
    classes = trainset.classes
    return trainset, testset, classes


def run_oop():

    #########
    # Config
    #########
    with open("config.json") as json_data_file:
        config = json.load(json_data_file)

    num_experiments = config["num_experiments"]
    al_cycles = config["al_cycles"]
    epochs = config["epochs"]

    #########
    # Seed
    #########
    random.seed("12345")
    torch.manual_seed(12345)

    #########
    # Data Loop:
    # runs of a specific dataset based on the arguments
    #########
    datasets = []
    datasets.append(cifar10_data())
    datasets.append(mnist_data())
    for train_set, test_set, classes in datasets:
        if type(train_set) is torchvision.datasets.mnist.MNIST:
            config['res_net_18']['in_dim'] = 1
        elif type(train_set) is torchvision.datasets.cifar.CIFAR10:
            config['res_net_18']['in_dim'] = 3

        #########
        # Active Learner Loop
        # Iterates through all the active learners and their corresponding settings and runs experiments on them
        #########
        active_learners = []
        active_learners.append(ResNet18LossActiveLearner4(train_set, test_set, config, device))
        active_learners.append(ResNet18LossActiveLearner3(train_set, test_set, config, device))
        active_learners.append(ResNet18LossActiveLearner2(train_set, test_set, config, device))
        active_learners.append(ResNet18LossActiveLearner1(train_set, test_set, config, device))
        active_learners.append(RandomActiveLearner(train_set, test_set, config, device))
        # active_learners.append(BaldActiveLearner(train_set, test_set, config, device))
        for active_learner in active_learners:
            #########
            # Experiments Loop
            # runs the number of experiments as defined in the config file
            #########
            for experiment_num in range(num_experiments):
                #########
                # Active Learning Loop
                #########
                best_acc = 0  # best test accuracy
                start_epoch = 0  # start from epoch 0 or last checkpoint epoch
                num_examples = []
                accuracy = []
                all_predicted, all_targets = [], []
                active_learner.instantiate_sets()
                for iteration in range(al_cycles):
                    #########
                    # Running Epochs
                    #########
                    for epoch in range(start_epoch, start_epoch + epochs):
                        active_learner.train_networks(epoch)
                    # torch.cuda.memory_summary(device=None, abbreviated=False)
                    # torch.cuda.empty_cache()
                    num_examples.append(len(active_learner.labeled_set_idx))
                    all_predicted, all_targets = active_learner.predict()
                    acc = active_learner.accuracy(all_predicted, all_targets)
                    accuracy.append(acc)
                    # Update the sets based on the loss prediction
                    active_learner.examples_select()

                datapoint = {
                    'dataset': train_set.__class__.__name__,
                    'active_learner': active_learner.__class__.__name__,
                    'experiment_num': experiment_num,
                    'al_cycles': al_cycles,
                    # 'pred_labels': all_predicted,
                    # 'target_lables': all_targets,
                    'acc': accuracy,
                    'num_examples': num_examples
                }
                with open("datapoints.json", "a") as myfile:
                    myfile.write(json.dumps(datapoint) + ",\n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--net', default="resnet", type=str, help='learning rate')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(device)
    # run_program()
    run_oop()
