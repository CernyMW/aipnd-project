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

def process_args():
    ''' Process the command line arguments
    '''
    configs = dict()

    parser = argparse.ArgumentParser(description='Predict an object\'s type.')
    parser.add_argument('image_path', help='Image to predict.')
    parser.add_argument('checkpoint_path', help='Pretrained neural network checkpoint file.')
    parser.add_argument('--gpu', action="store_true", help='Use GPU rather than CPU for training.')
    parser.add_argument('--topk', action="store", type=int, help='Top k category guesses.')
    parser.add_argument('--category_names', action="store", type=str, help='Category names file.', default='cat_to_name.json')
    
    args = parser.parse_args()

    configs['predict_device'] = "cpu"
    if(args.gpu == True and torch.cuda.is_available()):
        configs['predict_device'] = "cuda:0"

    if(os.path.isfile(args.image_path)):
        configs['image_path'] = args.image_path
    else:
        print("No or invalid image file.")
        exit()

    if(os.path.isfile(args.checkpoint_path)):
        configs['checkpoint_path'] = args.checkpoint_path
    else:
        print("No or invalid checkpoint file.")
        exit()

    if(os.path.isfile(args.category_names)):
        configs['category_names'] = args.category_names
    else:
        print("No or invalid category names file.")
        exit()

    if (args.topk == None):
        configs['topk'] = 5
    else:
        configs['topk'] = args.topk

    return configs




def load_checkpoint(configs):

    configs.update(torch.load(configs['checkpoint_path'], map_location=torch.device(configs['predict_device'])))
    #configs.update(torch.load(configs['checkpoint_path'], ))
    return configs
    
def model_create(configs):
    
    arch = configs['arch']
    print("Model architecture: ", arch)
    if(arch == "vgg16"):
        the_model =  models.vgg16(pretrained = True)
        inp = the_model.classifier[0].in_features        
    elif(arch == "resnet"):
        the_model = models.resnet50(pretrained = True)
        inp = 2048
    elif(arch == "alexnet"):
        the_model = models.alexnet(pretrained = True)
        inp = 9216
    elif(arch == "squeezenet"):
        the_model = models.squeezenet1_1(pretrained = True)
        inp = 26624
    elif(arch == "densenet"):
        the_model = models.densenet121(pretrained = True)
        inp = 1024
    elif(arch == "inception"):
        the_model = models.inception_v3(pretrained = True)
        inp = 128
    

    # Freeze parameters so we don't backprop through them
    for param in the_model.parameters():
        param.requires_grad = False
    
    the_model.load_state_dict(configs['state_dict'])
    the_model.class_to_idx = configs['class_to_idx']
    optimizer = configs['optimizer']
    criterion = configs['criterion']
    classifier = configs['classifier']
    #the_model.classifier.load_state_dict(configs['classifier_dict'])
    
    the_model.optimizer = optimizer
    the_model.classifier = classifier

    
        

    return the_model, criterion, optimizer

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil = Image.open(image)
    
    xform = transforms.Compose([transforms.Resize(256), 
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    pil = xform(pil)
    
    return pil


def predict(image_path, model, configs):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    model.eval()
    image = process_image(image_path)

    if(configs["predict_device"] == "cuda:0"):    
        model = model.cuda()

    image = image.unsqueeze(0)
    
    with torch.no_grad():
    
        if(configs["predict_device"] == "cuda:0"):    
            image = image.cpu()   
            image = image.cuda()
    
        output = model(image)
        probs, labels = torch.topk(output, configs['topk'])
        probs = probs.exp()

    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()

    if(configs["predict_device"] == "cuda:0"):    
        labels = labels.cpu()
        probs = probs.cpu()

    for label in labels.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])
        
    return probs.numpy()[0], mapped_classes

#image_path = 'flowers\\valid\\99\\image_08063.jpg'

#topk = 5

configs = process_args()
configs = load_checkpoint(configs)

image_path = configs['image_path']

the_model, criterion, optimizer = model_create(configs)
probabilities, mapped_classes = predict(image_path, the_model, configs)
#print(type(probabilities), mapped_classes)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

print("Image: ", image_path)
print("Category Names File: ", configs['category_names'])
print()

x = 0
for x, i in enumerate(mapped_classes):
    print("{:<30} {:.2%}".format(cat_to_name[i], probabilities[x]))
    