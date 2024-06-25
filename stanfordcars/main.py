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

import configparser

from utils import plot_confusion_matrix, CustomDataset, set_matrix_penalty, Experiment

cfg = configparser.ConfigParser()
cfg.read('./stanfordcars/default.cfg')
print(cfg.sections())
cfg = cfg['PARAMS'] # Slightly hacky

# Save frequently read config values
BATCH_SIZE = int(cfg['BATCH_SIZE'])
MODEL_SAVE_PATH = cfg['working_root'] + cfg['local_model_save']
RESULT_SAVE_PATH = cfg['working_root'] + cfg['local_result_save']
MODEL_NAME = cfg['model_name']
RESULT_MODEL = RESULT_SAVE_PATH + MODEL_NAME + '.pth'
DOMINO = cfg.getboolean('DOMINO')
DOMINOW = cfg.getboolean('DOMINOW')
EPOCHS = int(cfg['epochs'])
LOGGING_FREQ = int(cfg['logging_freq'])
SAVE_FREQ = int(cfg['save_freq'])

# Create directories if they don't exist
if not os.path.exists(MODEL_SAVE_PATH): os.mkdir(MODEL_SAVE_PATH)
if not os.path.exists(RESULT_SAVE_PATH): os.mkdir(RESULT_SAVE_PATH)

# Define transformations to be applied to each image
data_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.25),  # Apply with 50% probability
    transforms.RandomApply([transforms.RandomRotation(15)], p=0.25),  # Apply with 50% probability
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.25),  # Apply with 50% probability
    transforms.ToTensor()
])

# TODO Make train and valid transforms and make configurable

#Create custom datasets for train and validation
train_dataset = CustomDataset(cfg['train_dir'], transform=data_transform)
valid_dataset = CustomDataset(cfg['valid_dir'], transform=data_transform)

#Create data loaders
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = valid_loader
class_names = train_dataset.class_names

#Print dataset sizes and number of classes
print(f"Number of images in training set: {len(train_dataset)}")
print(f"Number of images in validation set: {len(valid_dataset)}")
print(f"Number of classes: {len(train_dataset.class_names)}")
print("Class names:", class_names)

# load a pre-trained model
# TODO make this automatic
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(2048, len(class_names))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Specify Loss # TODO - should always have file, figure out where to put it
if DOMINO:
    criterion = DOMINO_Loss()
    #file  = cfg['matrix_file'] 
    file = None
    matrix_penalty = set_matrix_penalty(file, len(class_names))
    a = 0.5
    b = 0.5

elif DOMINOW:
    criterion = DOMINO_Loss_W()
    #file  = cfg['matrix_file'] 
    file = None
    matrix_penalty = set_matrix_penalty(file, len(class_names))
    
else:
    criterion = nn.CrossEntropyLoss()

 # construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=float(cfg['lr']), momentum=float(cfg['momentum']), weight_decay=float(cfg['weight_decay']))

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg['step_size']), gamma=float(cfg['gamma']))


experiment = Experiment(model=model, criterion=criterion, optimizer=optimizer, scheduler=lr_scheduler, device=device)
# TODO make this a function call in a loop
val_acc_best = 0
for epoch in range(EPOCHS):
    logging_step = 1
    
    train_acc, loss = experiment.step(train_loader, training=True)
    val_acc, _ = experiment.step(valid_loader, training=False)
    
    if val_acc>val_acc_best:
        torch.save(experiment.model.state_dict(), RESULT_MODEL) # Check to see if this needs to be experiment.model
        val_acc_best=val_acc
        print('The new best validation accuracy is %.4f, saving model' % (val_acc_best))

    print("Epoch %d, loss: %.3f, Train acc: %.4f, Val acc: %.4f" % (epoch + 1,  loss, train_acc, val_acc))

#compute test using only the best performing model 
#(hopefully the following steps may be replaced by a testing dataset)
print(RESULT_MODEL)
checkpoint = torch.load(RESULT_MODEL)
experiment.model.load_state_dict(checkpoint)

outputs_total, labels_total, inputs_total, num_batches, test_acc = experiment.report(test_loader)
preds_total = torch.Tensor(outputs_total).argmax(dim=1)

#verify sizes
print(outputs_total.shape)
print(preds_total.shape)
print(labels_total.shape)
preds_total = preds_total.cpu().detach().numpy()

print('The accuracy on the testing set is: %.4f' % test_acc)
import time
a = time.gmtime()

#rcrm_metric_total = rcrm_metric_total / num_batches
rcrm_metric = RCRMMetric().calculate_rcr_metric(model = model, data = torch.Tensor(inputs_total).cuda(), target = torch.Tensor(labels_total).cuda(), num_components=1)
print(f"RCR Metric: {rcrm_metric}")
b = time.gmtime()
print((b-a)//60)

plot_confusion_matrix(labels_total, preds_total, class_names)
plt.tight_layout()
plt.savefig(RESULT_SAVE_PATH + MODEL_NAME + '_confusionmatrix_test.png')

#will need this to compute loss term
df_cm = pd.DataFrame(confusion_matrix(labels_total,preds_total))
df_cm.to_csv(RESULT_SAVE_PATH + MODEL_NAME + '_confusionmatrix_test.csv')

#classification report on test data

report =  classification_report(labels_total, preds_total, target_names = class_names, output_dict = True)
print(classification_report(labels_total, preds_total, target_names = class_names))

df = pd.DataFrame(report).transpose()
df.to_csv(RESULT_SAVE_PATH + MODEL_NAME + '_classificationreport.csv')

# Calculate RCR-G metric
rcrg_metric = RCRGMetric().calculate_rcr_metric(output = torch.Tensor(outputs_total).cuda(), target = torch.Tensor(preds_total).cuda())
print(f"RCR Metric: {rcrg_metric}")

#calibration metrics need output softmax

m = nn.Softmax(dim=1)
outputs_total = m(torch.Tensor(outputs_total))
outputs_total = outputs_total.cpu().detach().numpy()

plt.style.use("seaborn")

plt.rc("font", size=12)
plt.rc("axes", labelsize=12)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
plt.rc("legend", fontsize=12)

plt.rc("axes", titlesize=16)
plt.rc("figure", titlesize=16)
title = "Total Calibration Curve"

output_conf = np.max(outputs_total, axis=1)

fig = reliability_diagram(labels_total, preds_total, output_conf, num_bins=10, draw_ece=True,
                          draw_bin_importance="alpha", draw_averages=True,
                          title=title, figsize=(6, 6), dpi=100, 
                          return_fig=True)
fig.tight_layout()

fig.savefig(RESULT_SAVE_PATH + MODEL_NAME + '_allclass_calibrationcurve' + '.pdf')
plt.close()

#calibration curves and Brier Loss

for i in range(len(class_names)):
    
    labels_binary = np.zeros(len(labels_total))
    labels_binary[np.where(labels_total == i)] = 1
    
    prob_true, prob_pred = calibration_curve(labels_binary, outputs_total[:,i], n_bins=10, strategy = 'quantile')
    
    
    clf_score = brier_score_loss(labels_binary, outputs_total[:,i], pos_label=1)
    clf_score = np.round(clf_score,3)
    
    # Plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')# + str(Standard))

    # Plot model's calibration curve
    plt.plot(prob_pred, prob_true, marker = '.', label = 'Baseline Model')
    leg = plt.legend(loc = 'upper left')
    plt.xlabel('Average Predicted Probability in each bin')
    plt.ylabel('Ratio of positives')
    plt.title('Calibration Curve for ' + class_names[i] + ' with Brier Loss: ' + str(clf_score))
    plt.savefig(RESULT_SAVE_PATH + MODEL_NAME + '_Calibration-Curve_Class-' + class_names[i] + '.pdf')
    plt.close()
    
    #Plot all Confidence scores for this class
    plt.hist(outputs_total[:,i])
    plt.title('Full Histogram for ' + class_names[i])
    plt.savefig(RESULT_SAVE_PATH + MODEL_NAME + '_Histogram_Full_Class-' + class_names[i] + '.pdf')
    plt.close()
    
    #Plot only Confidence scores for this class in which this was the correct class
    outputs_total_pos = np.delete(outputs_total[:,i], np.where(labels_binary == 0))
    plt.hist(outputs_total_pos)
    plt.title('Positive only Histogram for ' + class_names[i])
    plt.savefig(RESULT_SAVE_PATH + MODEL_NAME + '_Histogram_Pos_Class-' + class_names[i] + '.pdf')
    plt.close()

    #other calibration scores

o = torch.Tensor(outputs_total)
l = torch.Tensor(labels_total)

metric1 = MulticlassCalibrationError(num_classes=len(class_names), n_bins=10, norm='l1')
ECE = metric1(o,l)
metric2 = MulticlassCalibrationError(num_classes=len(class_names), n_bins=10, norm='l2')
RMSCE = metric2(o,l)
metric3 = MulticlassCalibrationError(num_classes=len(class_names), n_bins=10, norm='max')
MCE = metric3(o,l)

print('ECE: %.4f' % (ECE))
print('RMSCE: %.4f' % (RMSCE))
print('MCE: %.4f' % (MCE))

#will need this to compute loss term
data = [['ECE', ECE], ['RMSCE', RMSCE], ['MCE', MCE]]
df_calmet = pd.DataFrame(data=data, columns=['Metric', 'Value'])
df_calmet.to_csv(RESULT_SAVE_PATH + MODEL_NAME + '_calibrationmetrics.csv')