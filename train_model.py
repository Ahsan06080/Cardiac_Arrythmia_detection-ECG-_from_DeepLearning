import sys
import os
import argparse
import logging
import json
import time
import subprocess
from shutil import copyfile

import numpy as np
from sklearn import metrics
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import DataParallel

from tensorboardX import SummaryWriter

import sys


if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 3:
        raise Exception('Include the data and model folders as arguments, e.g., python train_model.py data model.')

    data_directory = sys.argv[1]
    model_directory = sys.argv[2]

    training_code(data_directory, model_directory) ### Implement this function!

    print('Done.')
