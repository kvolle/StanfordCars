import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms

import os
import numpy as np
import matplotlib.pyplot as plt

#needed for evaluation metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd

#needed for calibration metrics
from sklearn.calibration import calibration_curve
from torchmetrics.classification import MulticlassCalibrationError
from sklearn.metrics import brier_score_loss

#new loss
from calibratedclassification.DominoLoss import DOMINO_Loss
from calibratedclassification.DominoLossW import DOMINO_Loss_W

from calibratedclassification.reliability_diagrams import *

import random
from calibratedclassification.RCR import RCRMMetric, RCRGMetric

from PIL import Image
from sklearn.model_selection import train_test_split

def plot_confusion_matrix(labels, pred_labels, classes):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, display_labels=classes)
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    plt.grid(False)
    plt.xticks(rotation=90)

# Create a dataset from the CustomDataset class
class CustomDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = sorted(os.listdir(root_dir))

    def __len__(self):
        return sum([len(files) for _, _, files in os.walk(self.root_dir)])

    def __getitem__(self, idx):
        class_name = self.class_names[idx // len(os.listdir(os.path.join(self.root_dir, self.class_names[0])))]
        class_dir = os.path.join(self.root_dir, class_name)
        image_name = os.listdir(class_dir)[idx % len(os.listdir(class_dir))]
        image_path = os.path.join(class_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        label = self.class_names.index(class_name)
    
        if self.transform:
            image = self.transform(image)
    
        return image, label
    
def set_matrix_penalty(file=None, classes=10):
    if file is None:
        matrix_penalty = np.eye(classes)
    else:
        matrix_vals = pd.read_csv(file, index_col=0, header=0)
    matrix_penalty = 3.0 * torch.from_numpy(matrix_vals.to_numpy())
    matrix_penalty = matrix_penalty.float().cuda()
    return matrix_penalty

class Experiment():
    def __init__(self, model, criterion, optimizer, scheduler = None, device='cpu', valid_loss=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.valid_loss = valid_loss # If true, calculate loss on validation set

    def step(self, dataloader, training=False):
        running_loss = 0.
        correct = 0
        seen = 0

        if training:
            self.model.train()
        else:
            self.model.eval()

        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(self.device) # TODO check if this is redundant
            labels = labels.to(self.device) # TODO check if this is redundant

            bs, c, h, w = inputs.shape
            if c == 1: # Handle grayscale by tiling
                inputs = inputs.repeat(1,3, 1, 1).float()

            if training:
                self.optimizer.zero_grad() # TODO learning scheduler

            outputs = self.model(inputs)
            correct += (outputs.argmax(dim=1) == labels).float().sum()
            seen += len(labels)

            if training or self.valid_loss:
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

            if training:
                loss.backward()
                self.optimizer.step()
                
        return correct/seen, running_loss/seen

    def report(self, dataloader):
        correct = 0
        seen = 0

        self.model.eval()
        num_batches = 0.

        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(self.device) # TODO check if this is redundant
            labels = labels.to(self.device) # TODO check if this is redundant

            bs, c, h, w = inputs.shape
            if c == 1: # Handle grayscale by tiling
                inputs = inputs.repeat(1,3, 1, 1).float()

            outputs = self.model(inputs)
            correct += (outputs.argmax(dim=1) == labels).float().sum()
            seen += len(labels)

            if i==0:
                outputs_total = outputs.cpu().detach().numpy()
                labels_total = labels.cpu().detach().numpy()
                inputs_total = inputs.cpu().detach().numpy()
            else:
                outputs_total = np.concatenate((outputs_total, outputs.cpu().detach().numpy()), axis=0)
                labels_total = np.concatenate((labels_total, labels.cpu().detach().numpy()), axis=0)
                inputs_total = np.concatenate((inputs_total, inputs.cpu().detach().numpy()), axis=0)
                
            num_batches += 1

        return outputs_total, labels_total, inputs_total, num_batches, correct/seen


