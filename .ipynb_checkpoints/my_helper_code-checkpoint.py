# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:01:29 2021

@author: Ahsan
"""
from helper_code import *
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as pl
import torch
from torch.utils.data.dataset import Dataset
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch.optim import lr_scheduler
import time
import tqdm
from evaluate_model import *

def one_zero(data):
    result = []
    for i,j in enumerate(data):
        if (data[i] > .5) :
            result.append(1)
        else :
            result.append(0)
    return result

def generating_output_files(model, test_classes,test_loader, output_directory) :
    model.eval()
    for inputs, target, header_files in test_loader:
        header_files = header_files[0]        
        input_var = torch.autograd.Variable(inputs.cuda().float())
        target_var = torch.autograd.Variable(target.cuda().float())
        output = model(input_var)
        probabilities = output.detach().cpu().numpy().squeeze()
        labels = one_zero(probabilities)
        header = load_header(header_files)
        recording_id = get_recording_id(header)
        head, tail = os.path.split(header_files)
        root, extension = os.path.splitext(tail)
        output_file = os.path.join(output_directory, root + '.csv')
        save_outputs(output_file, recording_id, test_classes, labels, probabilities)
        
        
def test_model(model, test_classes, test_loader, label_directory, output_directory) : 
    generating_output_files(model, test_classes, test_loader, output_directory)
    classes, auroc, auprc, auroc_classes, auprc_classes, accuracy, f_measure, f_measure_classes, challenge_metric = evaluate_model('test_data','test_outputs')
    print(f'Auroc : {auroc}')
    print(f'Accuracy : {accuracy}')
    print(f'f1 {f_measure}')
    return	auroc,accuracy,challenge_metric		
    
    
