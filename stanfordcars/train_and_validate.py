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
random.seed(1)
random_seed = random.seed(1)

from calibratedclassification.RCR import RCRMMetric, RCRGMetric

from PIL import Image
from sklearn.model_selection import train_test_split

BATCH_SIZE_train = 100#500#
BATCH_SIZE_test = 100

def plot_confusion_matrix(labels, pred_labels, classes):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, display_labels=classes)
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    plt.grid(False)
    plt.xticks(rotation=90)

# Define transformations to be applied to each image
data_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.25),  # Apply with 50% probability
    transforms.RandomApply([transforms.RandomRotation(15)], p=0.25),  # Apply with 50% probability
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.25),  # Apply with 50% probability
    transforms.ToTensor()
])

# Define the root directory of your dataset
train_dir = '/home/kyle/Desktop/Misc/Datasets/StanfordCarsClassFolders/car_data/train/'
valid_dir = '/home/kyle/Desktop/Misc/Datasets/StanfordCarsClassFolders/car_data/test/'
#test_dir = '/blue/ruogu.fang/skylastolte4444/Airplanes/Diffusion/Data/cars/test/'

# Extract class names from folder names
#class_names = [folder_name for folder_name in os.listdir(data_dir) 
#               if os.path.isdir(os.path.join(data_dir, folder_name)) 
#               and not folder_name.endswith('.txt')]

#class_names = []

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

#Create custom datasets for train and validation
train_dataset = CustomDataset(train_dir, transform=data_transform)
valid_dataset = CustomDataset(valid_dir, transform=data_transform)

#Define batch size
BATCH_SIZE = 30

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

# add directories for models and results

#generic directories
working_root = './'
model_save_path = working_root + 'models_cars/'
results_save_path = working_root + 'results_cars/'

#specific model definitions
model_name = 'resnet50_CE'#DOMINO-SSIM_REAL'
DOMINO = False
DOMINOW = False

#specific model path
results_model = results_save_path + model_name

#make all non-existing directories
isModelPath = os.path.exists(model_save_path)
isResultsPath = os.path.exists(results_save_path)
isModelResultsPath = os.path.exists(results_model)

if not isModelPath:
    os.mkdir(model_save_path)
    
if not isResultsPath:
    os.mkdir(results_save_path)

if not isModelResultsPath:
    os.mkdir(results_model)

# Domino specific settings
if DOMINO or DOMINOW:
    ##matrix_dir = working_root + 'scripts/OxfordSubDir/' #matrix_penalties_pet/'#matrix_penalties_pet/'
    #matrix_dir = '/blue/ruogu.fang/skylastolte4444/Airplanes/Diffusion/'
    ##matrix_vals = pd.read_csv(matrix_dir + 'oxfordpets_ssim_matrix_norm4.csv', header = 0, index_col=0) #'Dictionary_matrixpenalty_inv_patches_v1_1024.csv', index_col = 0) #header=None
    ##matrix_vals = pd.read_csv(matrix_dir + 'hc_matrixpenalty.csv', index_col = None, header=None)
    matrix_dir = '' # Don't have this
    matrix_vals = pd.read_csv(matrix_dir + 'cars_s64_e100_t900.csv', index_col=0, header=0)
    matrix_penalty = 3.0 * torch.from_numpy(matrix_vals.to_numpy())
    matrix_penalty = matrix_penalty.float().cuda()
    print(matrix_penalty.shape)
    
if DOMINO:
    a = 0.5
    b = 0.5

# load a pre-trained model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#model.fc.out_features = len(object_categories)
model.fc = nn.Linear(2048, len(class_names))
#model.fc = nn.Linear(512, len(object_categories))

#pretrained = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#backbone = torch.nn.Sequential(*(list(pretrained.children())[:-1]))
#model = torch.nn.Sequential(backbone, nn.Linear(2048, len(object_categories)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Specify Loss
if DOMINO:
    criterion = DOMINO_Loss()
elif DOMINOW:
    criterion = DOMINO_Loss_W()
else:
    criterion = nn.CrossEntropyLoss()

 # construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it for 1 epoch
num_epochs = 50
logging_freq = 147
save_freq = 5

val_acc_best = 0
for epoch in range(num_epochs):
    running_loss = 0.
    correct = 0.
    seen = 0.
    val_correct = 0.
    val_seen = 0.
    logging_step = 1
    
    for i, data in enumerate(train_loader, 0):##dataloaders['train'], 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        bs, c, h, w = inputs.shape
        if c == 1:
            inputs = inputs.repeat(1, 3, 1, 1).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        #outputs = outputs.logits
        if DOMINO:
            loss = criterion(outputs, labels, matrix_penalty, a, b)
        elif DOMINOW:
            loss = criterion(outputs, labels, matrix_penalty, 1)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).float().sum()
        seen += len(labels)

    for i, data in enumerate(valid_loader, 0):##dataloaders['val'], 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        bs, c, h, w = inputs.shape
        if c == 1:
            inputs = inputs.repeat(1, 3, 1, 1).float()

        outputs = model(inputs)
    
        val_correct += (outputs.argmax(dim=1) == labels).float().sum()
        val_seen += len(labels)
    
    #changed to only save models when validation improves
    val_acc = val_correct/val_seen
    if val_acc>val_acc_best:
        torch.save(model.state_dict(), model_save_path + model_name + '.pth')
        val_acc_best=val_acc
        print('The new best validation accuracy is %.4f, saving model' % (val_acc_best))

    print("Epoch %d, loss: %.3f, Train acc: %.4f, Val acc: %.4f" % (epoch + 1,  running_loss/seen, correct/seen, val_correct/val_seen))

#compute test using only the best performing model 
#(hopefully the following steps may be replaced by a testing dataset)

test_correct = 0.
test_seen = 0.
#rcrm_metric_total = 0.
num_batches = 0.

for i, data in enumerate(test_loader, 0):##dataloaders['val'], 0):
    
    torch.cuda.empty_cache()
    
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    bs, c, h, w = inputs.shape
    #print(inputs.shape)
    if c == 1:
        inputs = inputs.repeat(1, 3, 1, 1).float()

    outputs = model(inputs)
    
    test_correct += (outputs.argmax(dim=1) == labels).float().sum()
    test_seen += len(labels)
    
    # Calculate RCR-M metric
    #rcrm_metric = RCRMMetric().calculate_rcr_metric(model = model, data = torch.Tensor(inputs).cuda(), target = torch.Tensor(labels).cuda(), num_components=3)
    #print(f"RCR Metric: {rcrm_metric}")
    #rcrm_metric_total += rcrm_metric
    
    #save targets, predictions, and outputs for analysis
    if i==0:
        outputs_total = outputs.cpu().detach().numpy()
        #preds_total = outputs.argmax(dim=1)
        labels_total = labels.cpu().detach().numpy()
        inputs_total = inputs.cpu().detach().numpy()
    else:
        outputs_total = np.concatenate((outputs_total, outputs.cpu().detach().numpy()), axis=0)
        #preds_total = torch.cat((preds_total, outputs.argmax(dim=1)), dim=0)
        labels_total = np.concatenate((labels_total, labels.cpu().detach().numpy()), axis=0)
        inputs_total = np.concatenate((inputs_total, inputs.cpu().detach().numpy()), axis=0)
        
    num_batches += 1
        
    torch.cuda.empty_cache()
    
    #print(i)
    
    #if i > 945:
    #    break

preds_total = torch.Tensor(outputs_total).argmax(dim=1)

#rcrm_metric_total = rcrm_metric_total / num_batches
rcrm_metric = RCRMMetric().calculate_rcr_metric(model = model, data = torch.Tensor(inputs_total).cuda(), target = torch.Tensor(labels_total).cuda(), num_components=1)
print(f"RCR Metric: {rcrm_metric}")
        
#verify sizes
print(outputs_total.shape)
print(preds_total.shape)
print(labels_total.shape)

#outputs_total = outputs_total.cpu().detach().numpy()
preds_total = preds_total.cpu().detach().numpy()
#labels_total = labels_total.cpu().detach().numpy()

print('The accuracy on the testing set is: %.4f' % (test_correct/test_seen))

plot_confusion_matrix(labels_total, preds_total, class_names)
plt.tight_layout()
plt.savefig(results_model + '/confusionmatrix_test.png')

#will need this to compute loss term
df_cm = pd.DataFrame(confusion_matrix(labels_total,preds_total))
df_cm.to_csv(results_model + '/confusionmatrix_test.csv')

#classification report on test data

report =  classification_report(labels_total, preds_total, target_names = class_names, output_dict = True)
print(classification_report(labels_total, preds_total, target_names = class_names))

df = pd.DataFrame(report).transpose()
df.to_csv(results_model + '/classificationreport.csv')

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

fig.savefig(results_model + '/' + 'allclass_calibrationcurve' + '.pdf')
plt.close()

#calibration curves and Brier Loss

for i in range(len(class_names)):
    
    labels_binary = np.zeros(len(labels_total))
    labels_binary[np.where(labels_total == i)] = 1
    #current_label = class_names[i]
    
    prob_true, prob_pred = calibration_curve(labels_binary, outputs_total[:,i], n_bins=10, strategy = 'quantile')
    
    #print('prob_true:')
    #print(prob_true)
    
    #print('prob_pred:')
    #print(prob_pred)
    
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
    plt.savefig(results_model + '/Calibration-Curve_Class-' + class_names[i] + '.pdf')
    plt.close()
    
    #Plot all Confidence scores for this class
    plt.hist(outputs_total[:,i])
    plt.title('Full Histogram for ' + class_names[i])
    plt.savefig(results_model + '/Histogram_Full_Class-' + class_names[i] + '.pdf')
    plt.close()
    
    #Plot only Confidence scores for this class in which this was the correct class
    outputs_total_pos = np.delete(outputs_total[:,i], np.where(labels_binary == 0))
    plt.hist(outputs_total_pos)
    plt.title('Positive only Histogram for ' + class_names[i])
    plt.savefig(results_model + '/Histogram_Pos_Class-' + class_names[i] + '.pdf')
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
df_calmet.to_csv(results_model + '/calibrationmetrics.csv')