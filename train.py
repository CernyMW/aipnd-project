# Imports here
import numpy as np
import pandas as pd
import scipy
import json
import time
import os

import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision import datasets
from torch import utils as utils

import torchvision.models as models

import argparse

from PIL import Image
import matplotlib.pyplot as plt

from collections import OrderedDict



torch.cuda.init()
torch.cuda.empty_cache()

defaults = {
    'learning_rate' : 0.001,
    'device' : "cuda:0",
    'epochs' : 3,
    'hidden_units' : 1024,
    'arch' : "vgg16",
    'save_dir' : os.getcwd()
}

#valid_archs = ("vgg16", "resnet", "alexnet", "squeezenet", "densenet", "inception")
valid_archs = ("vgg16", "densenet")

def check_valid_dir(dirpath):
    return os.path.isdir(dirpath)

def set_architecture_parameters(configs):

    assert(configs['arch'] in valid_archs)

    return


def process_args():
    ''' Process the command line arguments
    '''
    configs = dict()

    parser = argparse.ArgumentParser(description='Train a neural network.')
    parser.add_argument('--learning_rate', action="store", type=float, help='Learning rate for the training. Defaults to 0.001')
    parser.add_argument('--gpu', action="store_true", help='Use GPU rather than CPU for training.')
    parser.add_argument('--epochs', action="store", type=int, help='Number of epochs to cycle the training. Defaults to 3.')
    parser.add_argument('--hidden_units', action="store", type=int, help='Number of  units in the hidden layer. Defaults to 1024.')
    parser.add_argument('--arch', action="store", type=str, help='Neural Network training architecture. Defaults to VGG13.')
    parser.add_argument('--save_dir', action="store", type=str, help='Directory to save the trained neural network in.')
    parser.add_argument('data_dir', help='Directory with training, testing, and validation images. Subdirectories must exist named \\train, \\test, \\valid')

    args = parser.parse_args()

    if(args.learning_rate == None):
        configs['learning_rate'] = defaults['learning_rate']
    else:
        configs['learning_rate'] = args.learning_rate

    configs['device'] = "cpu"
    if(args.gpu == True and torch.cuda.is_available()):
        configs['device'] = "cuda:0"

    if(args.epochs == None):
        configs['epochs'] = defaults['epochs']
    else:
        configs['epochs'] = args.epochs

    if(args.hidden_units == None):
        configs['hidden_units'] = defaults['hidden_units']
    else:
        configs['hidden_units'] = args.hidden_units

    if(args.arch == None):
        configs['arch'] = defaults['arch']
    else:
        configs['arch'] = str(args.arch).lower()
        if(configs['arch'] not in valid_archs):
            print("Invalid neural network architecture. Acceptable: VGG16, DenseNet")
            exit()

    if(check_valid_dir(args.data_dir)):
        configs['data_dir'] = args.data_dir
        configs['train_dir'] = args.data_dir + "/train"
        configs['valid_dir'] = args.data_dir + "/valid"
        configs['test_dir'] = args.data_dir + "/test"
    else:
        print("Invalid data directory.")
        exit()

    if(args.save_dir == None):
        configs['save_dir'] = defaults['save_dir']
    else:
        if(check_valid_dir(args.save_dir)):
            configs['save_dir'] = args.save_dir
        else:
            print("Invalid checkpoint save directory (directory needs to exist).")
            exit()

    return configs

def model_init(configs):
    
    training_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    validation_data_transforms = transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    testing_data_transforms = transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets = dict()
    image_datasets['training'] = datasets.ImageFolder(configs['train_dir'], transform=training_data_transforms)
    image_datasets['validation'] = datasets.ImageFolder(configs['valid_dir'], transform=validation_data_transforms)
    image_datasets['testing'] = datasets.ImageFolder(configs['test_dir'], transform=testing_data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = dict()
    if configs['device']=="cpu":
        b_size = 4
    else:
        b_size = 16

    dataloaders['training'] = utils.data.DataLoader(image_datasets['training'], batch_size = b_size, shuffle = True)
    dataloaders['validation'] = utils.data.DataLoader(image_datasets['validation'], batch_size = b_size, shuffle = False)
    dataloaders['testing'] = utils.data.DataLoader(image_datasets['testing'], batch_size = b_size, shuffle = False)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    return image_datasets, dataloaders, cat_to_name

def model_create(training_dataset, configs):
    
    arch = configs['arch']
    if(arch == "vgg16"):
        the_model =  models.vgg16(pretrained = True)
        inp = the_model.classifier[0].in_features        
    #elif(arch == "resnet"):
    #    the_model = models.resnet50(pretrained = True)
    #    inp = 2048
    #elif(arch == "alexnet"):
    #    the_model = models.alexnet(pretrained = True)
    #    inp = 9216
    #elif(arch == "squeezenet"):
    #    the_model = models.squeezenet1_1(pretrained = True)
    #    print(the_model)
    #    inp = 512
    elif(arch == "densenet"):
        the_model = models.densenet121(pretrained = True)
        inp = 1024
    #elif(arch == "inception"):
    #    the_model = models.inception_v3(pretrained = True)
    #    inp = 128
    

    # Freeze parameters so we don't backprop through them
    for param in the_model.parameters():
        param.requires_grad = False

    configs['state_dict'] = the_model.state_dict()    

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(inp, configs['hidden_units'])),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.2)),
                              ('fc2', nn.Linear(configs['hidden_units'], 102)),
                              ('dropout2', nn.Dropout(p=0.2)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    the_model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(the_model.classifier.parameters(), lr=configs['learning_rate'])    

    # set up some checkpoint data
    configs['opt_dict'] = optimizer.state_dict()
    configs['classifier_dict'] = classifier.state_dict()
    configs['class_to_idx'] = training_dataset.class_to_idx
    configs['criterion'] = criterion
    configs['optimizer'] = optimizer
    configs['classifier'] = classifier
    
    return the_model, criterion, optimizer

def check_accuracy_on_test(model, testloader):   
    
    correct = 0
    total = 0
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        for images, labels in testloader:
            #images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def do_deep_learning(model, trainloader, testloader, print_every, criterion, optimizer, configs):
    
    epochs = configs['epochs']
    device = configs['device']

    print("Getting ready to deep learn on device: ",device)
    print_every = print_every
    steps = 0

    # change to cuda, if available
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes.
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                validation_accuracy = check_accuracy_on_test(model, testloader)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Accuracy: {:.2%}".format(validation_accuracy))

                running_loss = 0



def save_checkpoint(model_config):

    torch.save(model_config, model_config['save_dir']+'\\checkpoint.tph')



configs = process_args()

image_datasets, dataloaders, cat_to_name = model_init(configs)

set_architecture_parameters(configs)
the_model, criterion, optimizer = model_create(image_datasets['training'], configs)
do_deep_learning(the_model, dataloaders['training'], dataloaders['validation'], 40, criterion, optimizer, configs)
validation_accuracy = check_accuracy_on_test(the_model, dataloaders['testing'])
print("Accuracy against test dataset = {:.2%}".format(validation_accuracy))
save_checkpoint(configs)
