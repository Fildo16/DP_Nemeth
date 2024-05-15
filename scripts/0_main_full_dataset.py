# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:24:17 2023

@author: fifon
"""

import os
import random
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split, SubsetRandomSampler
from torchvision import datasets, transforms, models 
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from imblearn.metrics import specificity_score
from PIL import Image
import seaborn as sns
import glob
import warnings
import time
from torch.optim.lr_scheduler import StepLR

#%% HYPERPARAMETERS

num_epochs = 100
batch_size = 8
learning_rate = 0.0001


#%% DEVICE 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("GPU is available and will be used.")
else:
    print("No GPU available, using CPU.")


# %% LOAD DATA FROM PATH

# # "dataset/Car-Bike-Dataset-final_1000/class_train"

# dataset_classes = datasets.ImageFolder(root="dataset/Car-Bike-Dataset-final_1000/class_train", transform=None)

# class_names=dataset_classes.classes
# num_classes = len(class_names)
# print(class_names)
# print(len(class_names))


# base_dir = 'C:\\Users\\fifon\\anaconda3\\envs\\DP\\my_scripts\\dataset\\Car-Bike-Dataset-final_1000'
# # train_dir = base_dir + '\\class_train\\'
# train_dir = base_dir + '\\class_train_NST\\'
# test_dir = base_dir + '\\class_test\\'

# %% CUDA PATH

# Car-Bike-Dataset-final_1000
# my_lung_cancer_1000

dataset_classes = datasets.ImageFolder(root="dataset/my_lung_cancer_1000/class_train", transform=None)

class_names=dataset_classes.classes
num_classes = len(class_names)
print(class_names)
print(len(class_names))


base_dir = '/home/nemeth/nemeth-scripts/dataset/my_lung_cancer_1000'
train_dir = base_dir + '/class_train/'
test_dir = base_dir + '/class_test/'


#%% CLASSIFICATION DATAMODULE

class DataModule(pl.LightningDataModule):

    def __init__(self,  batch_size=batch_size): 
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.train_percentage = train_percentage
        self.transform = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor()])


    def setup(self, stage=None):
        
        full_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.transform)
        test_dataset = datasets.ImageFolder(root=self.test_dir, transform=self.transform)  
       

        num_classes = len(full_dataset.classes)
        class_counts = [0] * num_classes
        class_indices = [[] for _ in range(num_classes)]
            
        for idx in range(len(full_dataset)):
            _, label = full_dataset[idx]
            class_counts[label] += 1
            class_indices[label].append(idx)
        
        number_per_class = int(self.train_percentage * min(class_counts))
        
        train_size_per_class = int(0.8 * number_per_class)
        val_size_per_class = int(0.2 * number_per_class)
        
        train_indices = []
        val_indices = []
    
        for class_idx in range(num_classes):
            class_indices_for_class = class_indices[class_idx]
            num_samples = min(len(class_indices_for_class), number_per_class)
            
            train_indices.extend(class_indices_for_class[:train_size_per_class])
            val_indices.extend(class_indices_for_class[train_size_per_class:train_size_per_class + val_size_per_class])
        
        count_numbers_in_range_1 = sum(1 for number in train_indices if 0 <= number <= 699)
        count_numbers_in_range_2 = sum(1 for number in train_indices if 700 <= number <= 1399)
        
        #print("Class 1 training", count_numbers_in_range_1)
        #print("Class 2 training", count_numbers_in_range_2)
        
        #print("Train indices", train_indices)     
        #print("Val indices", val_indices)
        
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
            


        self.train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        print("Number of samples for training dataset", len(self.train_dataset.dataset))
        
        self.val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        print("Number of samples for validation dataset", len(self.val_dataset.dataset))
        
        self.test_dataset = DataLoader(test_dataset, batch_size=self.batch_size)
        print("Number of samples for testing dataset", len(self.test_dataset.dataset))
                

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self,batch_size=batch_size):
        return self.val_dataset

    def test_dataloader(self,batch_size=batch_size):
        return self.test_dataset
    


#%% ARCHITECTURE OF CNN

class ResNet18(pl.LightningModule):
    def __init__(self, num_classes=num_classes):
        super(ResNet18, self).__init__()
        
        self.resnet18 = models.resnet18(weights=None)              
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes) 

    def forward(self, x):
        return self.resnet18(x)
    

class CNN(pl.LightningModule):
    def __init__(self, num_classes=num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 =nn.BatchNorm2d(16)

        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)

        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)

        # Calculate the size of the flattened output
        self.flat_size = 64 * (224 // 4) * (224 // 4)  # Assuming two max-pooling layers with size 2

        self.flat = nn.Flatten()
        
        # Adjust the linear layer to match the correct size
        self.fc = nn.Linear(self.flat_size, num_classes)
        


    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)

        x = self.flat(x)
        x = self.fc(x)


        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    
    start = time.time() 
   
    #TRAINING WITH 100% OF THE DATASET   
    train_p = [1.0]
    
    for train_percentage in train_p:
            
       print(f"Training with {train_percentage * 100}% of the dataset")

       
       datamodule = DataModule()
       datamodule.setup()
       
       test_loader = datamodule.test_dataloader()
       val_loader = datamodule.val_dataloader()
       train_loader = datamodule.train_dataloader()
   
   
       model = ResNet18().to(device)
       #model = CNN().to(device)
       
      
       criterion = nn.CrossEntropyLoss()
       optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
       scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Adjust step_size and gamma as needed
       
      
       #EMPTY LISTS FOR SAVING VARIABLES IN EPOCHS        
       train_epochs_accs = []
       train_epoch_recall = []
       train_epoch_precision = []       
       train_epoch_specificities = []
       train_epoch_f1s = []
       train_losses = []
       
       val_epoch_accs = []  
       val_epoch_recall = []  
       val_epoch_precision = []
       val_epoch_specificities = []
       val_epoch_f1s = []
       val_losses = []
       


   
       #TRAINING
       model.train()
       for epoch in range(num_epochs):
   
           # Initialize metrics
           total_loss = 0
           total_correct = 0
           total_samples = 0
           true_positives = 0
           true_negatives = 0
           false_positives = 0
           false_negatives = 0   


           for images, labels in train_loader:
               images = images.to(device)
               labels = labels.to(device)
   
               output = model(images)
               loss = criterion(output, labels)
               
               optimizer.zero_grad()           
               loss.backward()                 
               optimizer.step()
                              
              # Calculate metrics
               _, predicted = torch.max(output, 1)
               total_loss += loss.item()
               total_correct += (predicted == labels).sum().item()
               total_samples += labels.size(0)
               
               true_positives += ((predicted == 1) & (labels == 1)).sum().item()
               true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
               false_positives += ((predicted == 1) & (labels == 0)).sum().item()
               false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
           
           # Calculate metrics for the epoch
           epoch_loss = total_loss / len(train_loader)
           accuracy = total_correct / total_samples
           recall = 0 if (true_positives + false_negatives) == 0 else true_positives / (true_positives + false_negatives)                      
           specificity = 0 if (true_negatives + false_positives) == 0 else true_negatives / (true_negatives + false_positives)
           precision = 0 if (true_positives + false_positives) == 0 else true_positives / (true_positives + false_positives)
           f1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
               
           # Append metrics to lists
           train_losses.append(epoch_loss)
           train_epochs_accs.append(accuracy)
           train_epoch_recall.append(recall)
           train_epoch_specificities.append(specificity)
           train_epoch_precision.append(precision)
           train_epoch_f1s.append(f1)
       
           # Print metrics for the epoch
           print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}")
           

           
           
           
           #VALIDATION
           model.eval() 
           with torch.no_grad(): 
                   
                   val_total_loss = 0
                   val_total_correct = 0
                   val_total_samples = 0
                   val_true_positives = 0
                   val_true_negatives = 0
                   val_false_positives = 0
                   val_false_negatives = 0
   
                   for v_images, v_labels in val_loader:
                       v_images = v_images.to(device)
                       v_labels = v_labels.to(device)
                       
                       val_output = model(v_images)
                       val_loss = criterion(val_output,v_labels)
                       
                       # Calculate metrics
                       _, val_predicted = torch.max(val_output, 1)
                       val_total_loss += val_loss.item()
                       val_total_correct += (val_predicted == v_labels).sum().item()
                       val_total_samples += v_labels.size(0)
                       val_true_positives += ((val_predicted == 1) & (v_labels == 1)).sum().item()
                       val_true_negatives += ((val_predicted == 0) & (v_labels == 0)).sum().item()
                       val_false_positives += ((val_predicted == 1) & (v_labels == 0)).sum().item()
                       val_false_negatives += ((val_predicted == 0) & (v_labels == 1)).sum().item()
           
           # Calculate metrics for the validation set           
           val_epoch_loss = val_total_loss / len(val_loader)
           val_accuracy = val_total_correct / val_total_samples
           val_recall = 0 if (val_true_positives + val_false_negatives) == 0 else val_true_positives / (val_true_positives + val_false_negatives)          
           val_specificity = 0 if (val_true_negatives + val_false_positives) == 0 else val_true_negatives / (val_true_negatives + val_false_positives)
           val_precision = 0 if (val_true_positives + val_false_positives) == 0 else val_true_positives / (val_true_positives + val_false_positives)
           val_f1 = 0 if (val_precision + val_recall) == 0 else 2 * (val_precision * val_recall) / (val_precision + val_recall)
       
           # Append metrics to lists
           val_epoch_accs.append(val_accuracy)
           val_epoch_recall.append(val_recall)
           val_epoch_precision.append(val_precision)
           val_epoch_specificities.append(val_specificity)
           val_epoch_f1s.append(val_f1)
           val_losses.append(val_epoch_loss)
       
           # Print metrics for the epoch
           print(f"Validation Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.4f}, Recall: {val_recall:.4f}, Specificity: {val_specificity:.4f}, Precision: {val_precision:.4f}, F1 Score: {val_f1:.4f}")
  
    
       #Testing
       confusion_matrix = torch.zeros(num_classes, num_classes)
             
       #TESTING
       with torch.no_grad():
           n_correct = 0
           n_samples = 0
           n_class_correct = [0 for i in range(num_classes)]
           n_class_samples = [0 for i in range(num_classes)]
           
           
           
           for images, labels in test_loader:
               images = images.to(device)
               labels = labels.to(device)
               outputs = model(images)
               # max returns (value ,index)
               _, predicted = torch.max(outputs, 1)
               n_samples += labels.size(0)
               n_correct += (predicted == labels).sum().item()
               
               for i in range(min(batch_size, labels.size(0))):
                   label = labels[i]
                   pred = predicted[i]
                 
                   if (label == pred):
                       n_class_correct[label] += 1
                   n_class_samples[label] += 1
                   confusion_matrix[label, pred] += 1
       
           test_acc = 100.0 * n_correct / n_samples
           print(f'Accuracy of the network with {train_percentage * 100}% of the dataset : {test_acc} %')
           print(confusion_matrix)
           
           t_precision = [0 for i in range(num_classes)]
           t_recall = [0 for i in range(num_classes)]
           t_f1_score = [0 for i in range(num_classes)]
           
           for i in range(num_classes):
               t_precision[i] = n_class_correct[i] / (n_class_correct[i] + confusion_matrix.sum(dim=0)[i] - confusion_matrix[i, i])
               t_recall[i] = n_class_correct[i] / (n_class_correct[i] + confusion_matrix.sum(dim=1)[i] - confusion_matrix[i, i])
               t_f1_score[i] = 2 * (t_precision[i] * t_recall[i]) / (t_precision[i] + t_recall[i] + 1e-10)

           test_average_precision = sum(t_precision) / num_classes
           test_average_recall = sum(t_recall) / num_classes
           test_average_f1_score = 2 * (test_average_precision * test_average_recall) / (test_average_precision + test_average_recall + 1e-10)
   
           print(f'Average Precision: {test_average_precision}')
           print(f'Average Recall: {test_average_recall}')
           print(f'Average F1 Score: {test_average_f1_score}')
               
           
          
           #CONFUSION MATRIX           
           shape = confusion_matrix.shape
           cm = np.zeros(shape)

           shape = confusion_matrix.shape
           cm = np.zeros(shape)
           
           for i in range(shape[0]):
               for j in range(shape[1]):
                   cm[i, j] = confusion_matrix[i, j]
           
           cm = cm.astype(int)
           
           fig, ax = plt.subplots()
           cax = ax.matshow(cm, cmap="Blues")
           
           ax.set_xlabel("Predikovaná trieda")
           ax.set_ylabel("Aktuálna trieda")
           ax.set_title(f'Matica zámen so 100% datasetom')
           fig.colorbar(cax)
           
           for i in range(shape[0]):
               for j in range(shape[1]):
                   ax.text(j, i, "{}".format(cm[i, j]), ha="center", va="center", color="black")
           
           plt.show()
       
       end = time.time()
       total_time = end - start
       minutes = int(total_time // 60)
       seconds = int(total_time % 60)

       print(f"Total time taken: {minutes} minutes and {seconds} seconds")

#%%  PLOTTING EPOCHS/ACCURACY DURING 100 % OF TRAIN PERCENTAGE

epoch_list = list(range(1, num_epochs + 1))

vaL_accuracies_per = [acc * 100 for acc in val_epoch_accs]
t_accuracies_per = [acc * 100 for acc in train_epochs_accs]

vaL_loss = [loss * 1 for loss in val_losses]
t_loss = [loss * 1 for loss in train_losses]

plt.figure(figsize=(10, 6))
sns.lineplot(x=epoch_list, y=vaL_accuracies_per, marker='o', color='b', label='Validačná množina')
sns.lineplot(x=epoch_list, y=t_accuracies_per, marker='o', color='r', label='Trénovacia množina')

plt.xlabel('Epochy', fontsize=12)
plt.ylabel('Presnosť (%)', fontsize=12)
plt.title(f"Presnosť so vzorkami NST - testovacia presnosť {round(test_acc, 2)} %  ", fontsize=14)
#plt.title(f"Presnosť so 100 % datasetom - testovacia presnosť {round(test_acc, 2)} %  ", fontsize=14)


plt.legend(loc = "upper left")
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.ylim(min(t_loss), max(t_loss))
# plt.xlim(0, num_epochs + 1)
plt.show()


#%% SAVE VARIABLES TO CSV FILE


metrics_df = pd.DataFrame({
    'epochs': epoch_list,
    'train accuracies': train_epochs_accs,
    'train recall': train_epoch_recall,
    'train precision': train_epoch_precision,
    'train spec': train_epoch_specificities,
    'train f1': train_epoch_f1s,   
    'train losses': train_losses,
    
    'val accuracies': val_epoch_accs,
    'val recall': val_epoch_recall,
    'val precision': val_epoch_precision,
    'val specificity': val_epoch_specificities,  
    'val f1': val_epoch_f1s,
    'val losses': val_losses,
    
    'test precision': test_average_precision,
    'test recall': test_average_recall,   
    'test f1': test_average_f1_score,
    'test acc': test_acc    
})


metrics_df.to_csv(f"full_LC.csv", index=False)
metrics_df.to_json("full_CB", orient="records")



