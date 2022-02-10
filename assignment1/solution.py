# This script contains the helper functions you will be using for this assignment

import os
import random

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BassetDataset(Dataset):
    """
    BassetDataset class taken with permission from Dr. Ahmad Pesaranghader

    We have already processed the data in HDF5 format: er.h5
    See https://www.h5py.org/ for details of the Python package used

    We used the same data processing pipeline as the paper.
    You can find the code here: https://github.com/davek44/Basset
    """

    # Initializes the BassetDataset
    def __init__(self, path='./data/', f5name='er.h5', split='train', transform=None):
        """
        Args:
            :param path: path to HDF5 file
            :param f5name: HDF5 file name
            :param split: split that we are interested to work with
            :param transform (callable, optional): Optional transform to be applied on a sample
        """

        self.split = split

        split_dict = {'train': ['train_in', 'train_out'],
                      'test': ['test_in', 'test_out'],
                      'valid': ['valid_in', 'valid_out']}

        assert self.split in split_dict, "'split' argument can be only defined as 'train', 'valid' or 'test'"

        # Open hdf5 file where one-hoted data are stored
        self.dataset = h5py.File(os.path.join(path, f5name.format(self.split)), 'r')

        # Keeping track of the names of the target labels
        self.target_labels = self.dataset['target_labels']

        # Get the list of volumes
        self.inputs = self.dataset[split_dict[split][0]]
        self.outputs = self.dataset[split_dict[split][1]]

        self.ids = list(range(len(self.inputs)))
        if self.split == 'test':
            self.id_vars = np.char.decode(self.dataset['test_headers'])

    def __getitem__(self, i):
        """
        Returns the sequence and the target at index i

        Notes:
        * The data is stored as float16, however, your model will expect float32.
          Do the type conversion here!
        * Pay attention to the output shape of the data.
          Change it to match what the model is expecting
          hint: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        * The target must also be converted to float32
        * When in doubt, look at the output of __getitem__ !
        """

        idx = self.ids[i]

        # Sequence & Target
        output = {'sequence': None, 'target': None}

        # WRITE CODE HERE

        return output

    def __len__(self):
        # WRITE CODE HERE
        return 0

    def get_seq_len(self):
        """
        Answer to Q1 part 2
        """
        # WRITE CODE HERE
        return 0

    def is_equivalent(self):
        """
        Answer to Q1 part 3
        """
        # WRITE CODE HERE
        return 0


class Basset(nn.Module):
    """
    Basset model
    Architecture specifications can be found in the supplementary material
    You will also need to use some Convolution Arithmetic
    """

    def __init__(self):
        super(Basset, self).__init__()

        self.dropout = ?  # should be float
        self.num_cell_types = 164

        self.conv1 = nn.Conv2d(1, 300, (19, ?), stride=(1, 1), padding=(9, 0))
        self.conv2 = nn.Conv2d(300, ?, (?, 1), stride=(1, 1), padding=(?, 0))
        self.conv3 = nn.Conv2d(?, 200, (?, 1), stride=(1, 1), padding=(4, 0))

        self.bn1 = nn.BatchNorm2d(300)
        self.bn2 = nn.BatchNorm2d(?)
        self.bn3 = nn.BatchNorm2d(200)
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((?, 1))
        self.maxpool3 = nn.MaxPool2d((?, 1))

        self.fc1 = nn.Linear(13*200, ?)
        self.bn4 = nn.BatchNorm1d(?)

        self.fc2 = nn.Linear(1000, ?)
        self.bn5 = nn.BatchNorm1d(?)

        self.fc3 = nn.Linear(?, self.num_cell_types)

    def forward(self, x):
        """
        Compute forward pass for the model.
        nn.Module will automatically create the `.backward` method!

        Note:
            * You will have to use torch's functional interface to 
              complete the forward method as it appears in the supplementary material
            * There are additional batch norm layers defined in `__init__`
              which you will want to use on your fully connected layers
            * Don't include the output activation here!
        """

        # WRITE CODE HERE
        return 0


def compute_fpr_tpr(y_true, y_pred):
    """
    Computes the False Positive Rate and True Positive Rate
    Args:
        :param y_true: groundtruth labels (np.array of ints [0 or 1])
        :param y_pred: model decisions (np.array of ints [0 or 1])

    :Return: dict with keys 'tpr', 'fpr'.
             values are floats
    """
    output = {'fpr': 0., 'tpr': 0.}

    # WRITE CODE HERE

    return output


def compute_fpr_tpr_dumb_model():
    """
    Simulates a dumb model and computes the False Positive Rate and True Positive Rate

    :Return: dict with keys 'tpr_list', 'fpr_list'.
             These lists contain the tpr and fpr for different thresholds (k)
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05
                 ...
                 output['fpr_list'][-1] corresponds to k=0.95

            Do the same for output['tpr_list']

    """
    output = {'fpr_list': [], 'tpr_list': []}

    # WRITE CODE HERE

    return output


def compute_fpr_tpr_smart_model():
    """
    Simulates a smart model and computes the False Positive Rate and True Positive Rate

    :Return: dict with keys 'tpr_list', 'fpr_list'.
             These lists contain the tpr and fpr for different thresholds (k)
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05
                 ...
                 output['fpr_list'][-1] corresponds to k=0.95

            Do the same for output['tpr_list']
    """
    output = {'fpr_list': [], 'tpr_list': []}

    # WRITE CODE HERE

    return output


def compute_auc_both_models():
    """
    Simulates a dumb model and a smart model and computes the AUC of both

    :Return: dict with keys 'auc_dumb_model', 'auc_smart_model'.
             These contain the AUC for both models
             auc values in the lists should be floats
    """
    output = {'auc_dumb_model': 0., 'auc_smart_model': 0.}

    # WRITE CODE HERE

    return output


def compute_auc_untrained_model(model, dataloader, device):
    """
    Computes the AUC of your input model

    Args:
        :param model: solution.Basset()
        :param dataloader: torch.utils.data.DataLoader
                           Where the dataset is solution.BassetDataset
        :param device: torch.device

    :Return: dict with key 'auc'.
             This contains the AUC for the model
             auc value should be float

    Notes:
    * Dont forget to re-apply your output activation!
    * Make sure this function works with arbitrarily small dataset sizes!
    * You should collect all the targets and model outputs and then compute AUC at the end
      (compute time should not be as much of a consideration here)
    """
    output = {'auc': 0.}

    # WRITE CODE HERE

    return output


def compute_auc(y_true, y_model):
    """
    Computes area under the ROC curve (using method described in main.ipynb)
    Args:
        :param y_true: groundtruth labels (np.array of ints [0 or 1])
        :param y_model: model outputs (np.array of float32 in [0, 1])
    :Return: dict with key 'auc'.
             This contains the AUC for the model
             auc value should be float

    Note: if you set y_model as the output of solution.Basset, 
    you need to transform it before passing it here!
    """
    output = {'auc': 0.}

    # WRITE CODE HERE

    return output


def get_critereon():
    """
    Picks the appropriate loss function for our task
    criterion should be subclass of torch.nn
    """

    # WRITE CODE HERE

    return critereon


def train_loop(model, train_dataloader, device, optimizer, criterion):
    """
    One Iteration across the training set
    Args:
        :param model: solution.Basset()
        :param train_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Return: total_score, total_loss.
             float of model score (AUC) and float of model loss for the entire loop (epoch)
             (if you want to display losses and/or scores within the loop, 
             you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!

    Note: you don’t need to compute the score after each training iteration.
    If you do this, your training loop will be really slow!
    You should instead compute it every 50 or so iterations and aggregate ...
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE

    return output['total_score'], output['total_loss']


def valid_loop(model, valid_dataloader, device, optimizer, criterion):
    """
    One Iteration across the validation set
    Args:
        :param model: solution.Basset()
        :param valid_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Return: total_score, total_loss.
             float of model score (AUC) and float of model loss for the entire loop (epoch)
             (if you want to display losses and/or scores within the loop, 
             you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!
    
    Note: if it is taking very long to run, 
    you may do simplifications like with the train_loop.
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE

    return output['total_score'], output['total_loss']
