import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


import pandas as pd
import seaborn as sns
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu

import csv
import json


from extract_values import (
    extract_accuracies_acc,
    extract_accuracies_f1,
    extract_accuracies_sens,
    extract_accuracies_spec
)


#%% Values to load

number_560 = 1.0
number_320 = 0.572
number_160 = 0.287
number_80 = 0.143
number_40 = 0.073
number_20 = 0.037
number_10 = 0.019
number_5 = 0.01

train_percentage = [560,320,160,80,40,20,10,5]
seeds = [42,1024,123456]

font_subtitle = 18
font_legend = 14
font_maintitle = 20
epochs = list(range(1, 101))

#%% Full dataset training/testing

full_100 = {}
file_name_1 = f"D_LC_100"

with open(file_name_1, "r") as json_file:
    full_100 = json.load(json_file)



train_accuracy = (next(item for item in full_100 if item["train_percentage"] == number_560))["train_accuracy"]
train_accuracy = [i*100 for i in train_accuracy]
val_accuracy = (next(item for item in full_100 if item["train_percentage"] == number_560))["val_accuracy"]
val_accuracy = [i*100 for i in val_accuracy]

train_recall = (next(item for item in full_100 if item["train_percentage"] == number_560))["train_recall"]
train_recall = [i*100 for i in train_recall]
val_recall = (next(item for item in full_100 if item["train_percentage"] == number_560))["val_recall"]
val_recall = [i*100 for i in val_recall]


train_loss = (next(item for item in full_100 if item["train_percentage"] == number_560))["train_loss"]
train_loss = [i*100 for i in train_loss]
val_loss = (next(item for item in full_100 if item["train_percentage"] == number_560))["val_loss"]
val_loss = [i*100 for i in val_loss]

train_specificity = (next(item for item in full_100 if item["train_percentage"] == number_560))["train_specificity"]
train_specificity = [i*100 for i in train_specificity]
val_specificity = (next(item for item in full_100 if item["train_percentage"] == number_560))["val_specificity"]
val_specificity = [i*100 for i in val_specificity]




fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.lineplot(ax=axes[0, 0], x=epochs, y=train_accuracy, marker='o', color='orange', label='Trénovacia množina', linewidth=2)
sns.lineplot(ax=axes[0, 0], x=epochs, y=val_accuracy, marker='o', color='blue', label='Validačná množina', linewidth=2)
axes[0, 0].set_title('Presnosť', fontsize=font_subtitle)
axes[0, 0].set_ylim(0, 101) 
axes[0, 0].legend(loc="lower right", fontsize=font_legend)
axes[0, 0].set_xlabel('Epochy [-]', fontsize=font_legend)
axes[0, 0].set_ylabel('Presnosť (%)', fontsize=font_legend) 

sns.lineplot(ax=axes[0, 1], x=epochs, y=train_loss, marker='o', color='orange', label='Trénovacia množina', linewidth=2)
sns.lineplot(ax=axes[0, 1], x=epochs, y=val_loss, marker='o', color='blue', label='Validačná množina', linewidth=2)
axes[0, 1].set_title('Celková strata', fontsize=font_subtitle)
axes[0, 1].set_ylim(0, 101) 
axes[0, 1].legend(loc="lower right", fontsize=font_legend)

sns.lineplot(ax=axes[1, 0], x=epochs, y=train_recall, marker='o', color='orange',  label='Trénovacia množina', linewidth=2)
sns.lineplot(ax=axes[1, 0], x=epochs, y=val_recall, marker='o', color='blue', label='Validačná množina', linewidth=2)
axes[1, 0].set_title('Senzitivita', fontsize=font_subtitle)
axes[1, 0].set_ylim(0, 101) 
axes[1, 0].legend(loc="lower right", fontsize=font_legend)

sns.lineplot(ax=axes[1, 1], x=epochs, y=train_specificity, marker='o', color='orange',  label='Trénovacia množina', linewidth=2)
sns.lineplot(ax=axes[1, 1], x=epochs, y=val_specificity, marker='o', color='blue', label='Validačná množina', linewidth=2)
axes[1, 1].set_title('Špecifita', fontsize=font_subtitle)
axes[1, 1].set_ylim(0, 101) 
axes[1, 1].legend(loc="lower right", fontsize=font_legend)


fig.suptitle("Učenie s plným datasetom obrazov z histológie", fontsize=font_maintitle)
plt.tight_layout()
plt.show()

#%% Architecture Resnet18

test_50_1 = {}
test_50_2 = {}
test_50_3 = {}


# Iterate over the range of numbers
for seed in seeds:
    # Construct the file name
    file_name_1 = f"decreasing_LC_50_{seed}"
    file_name_2 = f"decreasing_LC_50_{seed}"
    file_name_3 = f"decreasing_LC_50_{seed}"
    
    
    # Open the JSON file
    with open(file_name_1, "r") as json_file:
        test_50_1[seed] = json.load(json_file)
    with open(file_name_2, "r") as json_file:
        test_50_2[seed] = json.load(json_file)
    with open(file_name_3, "r") as json_file:
        test_50_3[seed] = json.load(json_file)



test_acc_50 = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_acc(test_50_1, 42),
                                                        extract_accuracies_acc(test_50_2, 1024), 
                                                        extract_accuracies_acc(test_50_3, 123456))]


test_f1_50 = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_f1(test_50_1, 42), 
                                                       extract_accuracies_f1(test_50_2, 1024), 
                                                       extract_accuracies_f1(test_50_3, 123456))]


test_spec_50 = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_spec(test_50_1, 42), 
                                                         extract_accuracies_spec(test_50_2, 1024), 
                                                         extract_accuracies_spec(test_50_3, 123456))]


test_sens_50 = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_sens(test_50_1, 42), 
                                                         extract_accuracies_sens(test_50_2, 1024), 
                                                         extract_accuracies_sens(test_50_3, 123456))]

#%% Architecture ResNet50 a Resnet152


test_res50_1 = {}
test_res50_2 = {}
test_res50_3 = {}

test_res152_1 = {}
test_res152_2 = {}
test_res152_3 = {}

# Iterate over the range of numbers
for seed in seeds:
    # Construct the file name
    file_name_1 = f"decreasing_LC_50_{seed}_resnet50"
    file_name_2 = f"decreasing_LC_50_{seed}_resnet50"
    file_name_3 = f"decreasing_LC_50_{seed}_resnet50"
    
    file_name_4 = f"decreasing_LC_50_{seed}_resnet152"
    file_name_5 = f"decreasing_LC_50_{seed}_resnet152"
    file_name_6 = f"decreasing_LC_50_{seed}_resnet152"
    
    
    # Open the JSON file
    with open(file_name_1, "r") as json_file:
        test_res50_1[seed] = json.load(json_file)
    with open(file_name_2, "r") as json_file:
        test_res50_2[seed] = json.load(json_file)
    with open(file_name_3, "r") as json_file:
        test_res50_3[seed] = json.load(json_file)
    
    with open(file_name_4, "r") as json_file:
        test_res152_1[seed] = json.load(json_file)
    with open(file_name_5, "r") as json_file:
        test_res152_2[seed] = json.load(json_file)
    with open(file_name_6, "r") as json_file:
        test_res152_3[seed] = json.load(json_file)

#ResNet50
test_acc_resnet50 = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_acc(test_res50_1, 42),
                                                        extract_accuracies_acc(test_res50_2, 1024), 
                                                        extract_accuracies_acc(test_res50_3, 123456))]


test_f1_resnet50 = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_f1(test_res50_1, 42), 
                                                       extract_accuracies_f1(test_res50_2, 1024), 
                                                       extract_accuracies_f1(test_res50_3, 123456))]


test_spec_resnet50 = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_spec(test_res50_1, 42), 
                                                         extract_accuracies_spec(test_res50_2, 1024), 
                                                         extract_accuracies_spec(test_res50_3, 123456))]


test_sens_resnet50 = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_sens(test_res50_1, 42), 
                                                         extract_accuracies_sens(test_res50_2, 1024), 
                                                         extract_accuracies_sens(test_res50_3, 123456))]







#ResNet152
test_acc_resnet152 = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_acc(test_res152_1, 42),
                                                        extract_accuracies_acc(test_res152_2, 1024), 
                                                        extract_accuracies_acc(test_res152_3, 123456))]


test_f1_resnet152 = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_f1(test_res152_1, 42), 
                                                       extract_accuracies_f1(test_res152_2, 1024), 
                                                       extract_accuracies_f1(test_res152_3, 123456))]


test_spec_resnet152 = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_spec(test_res152_1, 42), 
                                                         extract_accuracies_spec(test_res152_2, 1024), 
                                                         extract_accuracies_spec(test_res152_3, 123456))]


test_sens_resnet152 = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_sens(test_res152_1, 42), 
                                                         extract_accuracies_sens(test_res152_2, 1024), 
                                                         extract_accuracies_sens(test_res152_3, 123456))]





fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.lineplot(ax=axes[0, 0], x=train_percentage, y=test_acc_50, marker='o', color='orange', label='ResNet18', linewidth=2)
sns.lineplot(ax=axes[0, 0], x=train_percentage, y=test_acc_resnet50, marker='o', color='blue', label='ResNet50', linewidth=2)
sns.lineplot(ax=axes[0, 0], x=train_percentage, y=test_acc_resnet152, marker='o', color='skyblue', label='ResNet152', linewidth=2)
axes[0, 0].set_title('Presnosť', fontsize=font_subtitle)
axes[0, 0].set_xscale('log')
axes[0, 0].set_xticks(train_percentage)
axes[0, 0].set_xticklabels([str(val) for val in train_percentage])
axes[0, 0].set_ylim(50, 100) 
axes[0, 0].legend(loc="lower right", fontsize=font_legend)
axes[0, 0].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[0, 0].set_ylabel('Presnosť (%)', fontsize=font_legend)  


sns.lineplot(ax=axes[0, 1], x=train_percentage, y=test_f1_50, marker='o', color='orange', label='ResNet18', linewidth=2)
sns.lineplot(ax=axes[0, 1], x=train_percentage, y=test_f1_resnet50, marker='o', color='blue', label='ResNet50', linewidth=2)
sns.lineplot(ax=axes[0, 1], x=train_percentage, y=test_f1_resnet152, marker='o', color='skyblue', label='ResNet152', linewidth=2)
axes[0, 1].set_title('F1 Skóre', fontsize=font_subtitle)
axes[0, 1].set_xscale('log')
axes[0, 1].set_xticks(train_percentage)
axes[0, 1].set_xticklabels([str(val) for val in train_percentage])
axes[0, 1].set_ylim(50, 100) 
axes[0, 1].legend(loc="lower right", fontsize=font_legend)
axes[0, 1].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[0, 1].set_ylabel('F1 Skóre (%)', fontsize=font_legend)  
# for i, value in enumerate(test_f1_50):
#     axes[0, 1].annotate(f'{value:.1f}', (train_percentage[i], test_f1_50[i]), textcoords="offset points", xytext=(0,-10), ha='center')
# for i, value in enumerate(test_DA_transform_f1):
#     axes[0, 1].annotate(f'{value:.1f}', (train_percentage[i], test_DA_transform_f1[i]), textcoords="offset points", xytext=(0,10), ha='center')



sns.lineplot(ax=axes[1, 0], x=train_percentage, y=test_spec_50, marker='o', color='orange', label='ResNet18', linewidth=2)
sns.lineplot(ax=axes[1, 0], x=train_percentage, y=test_spec_resnet50, marker='o', color='blue', label='ResNet50', linewidth=2)
sns.lineplot(ax=axes[1, 0], x=train_percentage, y=test_spec_resnet152, marker='o', color='skyblue', label='ResNet152', linewidth=2)
axes[1, 0].set_title('Špecifita', fontsize=font_subtitle)
axes[1, 0].set_xscale('log')
axes[1, 0].set_xticks(train_percentage)
axes[1, 0].set_xticklabels([str(val) for val in train_percentage])
axes[1, 0].set_ylim(50, 100) 
axes[1, 0].legend(loc="lower right", fontsize=font_legend)
axes[1, 0].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[1, 0].set_ylabel('Špecificita (%)', fontsize=font_legend)  
# for i, value in enumerate(test_spec_50):
#     axes[1, 0].annotate(f'{value:.1f}', (train_percentage[i], test_spec_50[i]), textcoords="offset points", xytext=(0,-10), ha='center')
# for i, value in enumerate(test_DA_transform_spec):
#     axes[1, 0].annotate(f'{value:.1f}', (train_percentage[i], test_DA_transform_spec[i]), textcoords="offset points", xytext=(0,10), ha='center')



sns.lineplot(ax=axes[1, 1], x=train_percentage, y=test_sens_50, marker='o', color='orange', label='ResNet18', linewidth=2)
sns.lineplot(ax=axes[1, 1], x=train_percentage, y=test_sens_resnet50, marker='o', color='blue', label='ResNet50', linewidth=2)
sns.lineplot(ax=axes[1, 1], x=train_percentage, y=test_sens_resnet152, marker='o', color='skyblue', label='ResNet152', linewidth=2)
axes[1, 1].set_title('Senzitivita', fontsize=font_subtitle)
axes[1, 1].set_xscale('log')
axes[1, 1].set_xticks(train_percentage)
axes[1, 1].set_xticklabels([str(val) for val in train_percentage])
axes[1, 1].set_ylim(50, 100) 
axes[1, 1].legend(loc="lower right", fontsize=font_legend)
axes[1, 1].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[1, 1].set_ylabel('Senzitivita (%)', fontsize=font_legend)  
# for i, value in enumerate(test_sens_50):
#     axes[1, 1].annotate(f'{value:.1f}', (train_percentage[i], test_sens_50[i]), textcoords="offset points", xytext=(0,-10), ha='center')
# for i, value in enumerate(test_DA_transform_sens):
#     axes[1, 1].annotate(f'{value:.1f}', (train_percentage[i], test_DA_transform_sens[i]), textcoords="offset points", xytext=(0,10), ha='center')


fig.suptitle("Dataset obrazov z histológie - Porovnanie architektúr ResNet", fontsize=font_maintitle)
plt.tight_layout()
plt.show()

#%% Data augmentation online

data_DA_transform_1 = {}
data_DA_transform_2 = {}
data_DA_transform_3 = {}


# Iterate over the range of numbers
for seed in seeds:
    # Construct the file name
    file_name_1 = f"DA_online_LC_{seed}"
    file_name_2 = f"DA_online_LC_{seed}"
    file_name_3 = f"DA_online_LC_{seed}"
    
    
    # Open the JSON file
    with open(file_name_1, "r") as json_file:
        data_DA_transform_1[seed] = json.load(json_file)
    with open(file_name_2, "r") as json_file:
        data_DA_transform_2[seed] = json.load(json_file)
    with open(file_name_3, "r") as json_file:
        data_DA_transform_3[seed] = json.load(json_file)


test_DA_transform_acc = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_acc(data_DA_transform_1, 42), 
                                                                  extract_accuracies_acc(data_DA_transform_2, 1024), 
                                                                  extract_accuracies_acc(data_DA_transform_3, 123456))]

test_DA_transform_f1 = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip( extract_accuracies_f1(data_DA_transform_1, 42), 
                                                                 extract_accuracies_f1(data_DA_transform_2, 1024), 
                                                                 extract_accuracies_f1(data_DA_transform_3, 123456))]


test_DA_transform_spec = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip( extract_accuracies_spec(data_DA_transform_1, 42), 
                                                                   extract_accuracies_spec(data_DA_transform_2, 1024), 
                                                                   extract_accuracies_spec(data_DA_transform_3, 123456))]



test_DA_transform_sens = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_sens(data_DA_transform_1, 42), 
                                                                   extract_accuracies_sens(data_DA_transform_2, 1024), 
                                                                   extract_accuracies_sens(data_DA_transform_3, 123456))]



fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.lineplot(ax=axes[0, 0], x=train_percentage, y=test_acc_50, marker='o', color='orange', label='Bez dátovej augmentácie', linewidth=2)
sns.lineplot(ax=axes[0, 0], x=train_percentage, y=test_DA_transform_acc, marker='o', color='blue', label='Dátová augmentácia online', linewidth=2)
axes[0, 0].set_title('Presnosť', fontsize=font_subtitle)
axes[0, 0].set_xscale('log')
axes[0, 0].set_xticks(train_percentage)
axes[0, 0].set_xticklabels([str(val) for val in train_percentage])
axes[0, 0].set_ylim(50, 100) 
axes[0, 0].legend(loc="lower right", fontsize=font_legend)
axes[0, 0].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[0, 0].set_ylabel('Presnosť (%)', fontsize=font_legend)  
# for i, value in enumerate(test_acc_50):
#     axes[0, 0].annotate(f'{value:.1f}', (train_percentage[i], test_acc_50[i]), textcoords="offset points", xytext=(0,-10), ha='center')
# for i, value in enumerate(test_DA_transform_acc):
#     axes[0, 0].annotate(f'{value:.1f}', (train_percentage[i], test_DA_transform_acc[i]), textcoords="offset points", xytext=(0,10), ha='center')


sns.lineplot(ax=axes[0, 1], x=train_percentage, y=test_f1_50, marker='o', color='orange', label='Bez dátovej augmentácie', linewidth=2)
sns.lineplot(ax=axes[0, 1], x=train_percentage, y=test_DA_transform_f1, marker='o', color='blue', label='Dátová augmentácia online', linewidth=2)
axes[0, 1].set_title('F1 Skóre', fontsize=font_subtitle)
axes[0, 1].set_xscale('log')
axes[0, 1].set_xticks(train_percentage)
axes[0, 1].set_xticklabels([str(val) for val in train_percentage])
axes[0, 1].set_ylim(50, 100) 
axes[0, 1].legend(loc="lower right", fontsize=font_legend)
axes[0, 1].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[0, 1].set_ylabel('F1 Skóre (%)', fontsize=font_legend)  
# for i, value in enumerate(test_f1_50):
#     axes[0, 1].annotate(f'{value:.1f}', (train_percentage[i], test_f1_50[i]), textcoords="offset points", xytext=(0,-10), ha='center')
# for i, value in enumerate(test_DA_transform_f1):
#     axes[0, 1].annotate(f'{value:.1f}', (train_percentage[i], test_DA_transform_f1[i]), textcoords="offset points", xytext=(0,10), ha='center')



sns.lineplot(ax=axes[1, 0], x=train_percentage, y=test_spec_50, marker='o', color='orange', label='Bez dátovej augmentácie', linewidth=2)
sns.lineplot(ax=axes[1, 0], x=train_percentage, y=test_DA_transform_spec, marker='o', color='blue', label='Dátová augmentácia online', linewidth=2)
axes[1, 0].set_title('Špecifita', fontsize=font_subtitle)
axes[1, 0].set_xscale('log')
axes[1, 0].set_xticks(train_percentage)
axes[1, 0].set_xticklabels([str(val) for val in train_percentage])
axes[1, 0].set_ylim(50, 100) 
axes[1, 0].legend(loc="lower right", fontsize=font_legend)
axes[1, 0].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[1, 0].set_ylabel('Špecificita (%)', fontsize=font_legend)  
# for i, value in enumerate(test_spec_50):
#     axes[1, 0].annotate(f'{value:.1f}', (train_percentage[i], test_spec_50[i]), textcoords="offset points", xytext=(0,-10), ha='center')
# for i, value in enumerate(test_DA_transform_spec):
#     axes[1, 0].annotate(f'{value:.1f}', (train_percentage[i], test_DA_transform_spec[i]), textcoords="offset points", xytext=(0,10), ha='center')



sns.lineplot(ax=axes[1, 1], x=train_percentage, y=test_sens_50, marker='o', color='orange', label='Bez dátovej augmentácie', linewidth=2)
sns.lineplot(ax=axes[1, 1], x=train_percentage, y=test_DA_transform_sens, marker='o', color='blue', label='Dátová augmentácia online', linewidth=2)
axes[1, 1].set_title('Senzitivita', fontsize=font_subtitle)
axes[1, 1].set_xscale('log')
axes[1, 1].set_xticks(train_percentage)
axes[1, 1].set_xticklabels([str(val) for val in train_percentage])
axes[1, 1].set_ylim(50, 100) 
axes[1, 1].legend(loc="lower right", fontsize=font_legend)
axes[1, 1].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[1, 1].set_ylabel('Senzitivita (%)', fontsize=font_legend)  
# for i, value in enumerate(test_sens_50):
#     axes[1, 1].annotate(f'{value:.1f}', (train_percentage[i], test_sens_50[i]), textcoords="offset points", xytext=(0,-10), ha='center')
# for i, value in enumerate(test_DA_transform_sens):
#     axes[1, 1].annotate(f'{value:.1f}', (train_percentage[i], test_DA_transform_sens[i]), textcoords="offset points", xytext=(0,10), ha='center')



fig.suptitle("Dataset obrazov z histológie - Dátová augmentácia online", fontsize=font_maintitle)
plt.tight_layout()
plt.show()



#%% Data augmentation offline

data_DA_offline_1 = {}
data_DA_offline_2 = {}
data_DA_offline_3 = {}


# Iterate over the range of numbers
for seed in seeds:
    # Construct the file name
    file_name_1 = f"DA_offline_LC_{seed}"
    file_name_2 = f"DA_offline_LC_{seed}"
    file_name_3 = f"DA_offline_LC_{seed}"
    
    
    # Open the JSON file
    with open(file_name_1, "r") as json_file:
        data_DA_offline_1[seed] = json.load(json_file)
    with open(file_name_2, "r") as json_file:
        data_DA_offline_2[seed] = json.load(json_file)
    with open(file_name_3, "r") as json_file:
        data_DA_offline_3[seed] = json.load(json_file)


test_DA_offline_acc = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_acc(data_DA_offline_1, 42), 
                                                                  extract_accuracies_acc(data_DA_offline_2, 1024), 
                                                                  extract_accuracies_acc(data_DA_offline_3, 123456))]

test_DA_offline_f1 = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip( extract_accuracies_f1(data_DA_offline_1, 42), 
                                                                 extract_accuracies_f1(data_DA_offline_2, 1024), 
                                                                 extract_accuracies_f1(data_DA_offline_3, 123456))]


test_DA_offline_spec = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip( extract_accuracies_spec(data_DA_offline_1, 42), 
                                                                   extract_accuracies_spec(data_DA_offline_2, 1024), 
                                                                   extract_accuracies_spec(data_DA_offline_3, 123456))]



test_DA_offline_sens = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_sens(data_DA_offline_1, 42), 
                                                                   extract_accuracies_sens(data_DA_offline_2, 1024), 
                                                                   extract_accuracies_sens(data_DA_offline_3, 123456))]


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.lineplot(ax=axes[0, 0], x=train_percentage, y=test_acc_50, marker='o', color='orange', label='Bez dátovej augmentácie', linewidth=2)
sns.lineplot(ax=axes[0, 0], x=train_percentage, y=test_DA_offline_acc, marker='o', color='blue', label='Dátová augmentácia offline', linewidth=2)
axes[0, 0].set_title('Presnosť', fontsize=font_subtitle)
axes[0, 0].set_xscale('log')
axes[0, 0].set_xticks(train_percentage)
axes[0, 0].set_xticklabels([str(val) for val in train_percentage])
axes[0, 0].set_ylim(50, 100) 
axes[0, 0].legend(loc="lower right", fontsize=font_legend)
axes[0, 0].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[0, 0].set_ylabel('Presnosť (%)', fontsize=font_legend)  

sns.lineplot(ax=axes[0, 1], x=train_percentage, y=test_f1_50, marker='o', color='orange', label='Bez dátovej augmentácie', linewidth=2)
sns.lineplot(ax=axes[0, 1], x=train_percentage, y=test_DA_offline_f1, marker='o', color='blue', label='Dátová augmentácia offline', linewidth=2)
axes[0, 1].set_title('F1 Skóre', fontsize=font_subtitle)
axes[0, 1].set_xscale('log')
axes[0, 1].set_xticks(train_percentage)
axes[0, 1].set_xticklabels([str(val) for val in train_percentage])
axes[0, 1].set_ylim(50, 100) 
axes[0, 1].legend(loc="lower right", fontsize=font_legend)
axes[0, 1].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[0, 1].set_ylabel('F1 Skóre (%)', fontsize=font_legend)  

sns.lineplot(ax=axes[1, 0], x=train_percentage, y=test_spec_50, marker='o', color='orange', label='Bez dátovej augmentácie', linewidth=2)
sns.lineplot(ax=axes[1, 0], x=train_percentage, y=test_DA_offline_spec, marker='o', color='blue', label='Dátová augmentácia offline', linewidth=2)
axes[1, 0].set_title('Špecifita', fontsize=font_subtitle)
axes[1, 0].set_xscale('log')
axes[1, 0].set_xticks(train_percentage)
axes[1, 0].set_xticklabels([str(val) for val in train_percentage])
axes[1, 0].set_ylim(50, 100) 
axes[1, 0].legend(loc="lower right", fontsize=font_legend)
axes[1, 0].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[1, 0].set_ylabel('Špecifita (%)', fontsize=font_legend)  

sns.lineplot(ax=axes[1, 1], x=train_percentage, y=test_sens_50, marker='o', color='orange', label='Bez dátovej augmentácie', linewidth=2)
sns.lineplot(ax=axes[1, 1], x=train_percentage, y=test_DA_offline_sens, marker='o', color='blue', label='Dátová augmentácia offline', linewidth=2)
axes[1, 1].set_title('Senzitivita', fontsize=font_subtitle)
axes[1, 1].set_xscale('log')
axes[1, 1].set_xticks(train_percentage)
axes[1, 1].set_xticklabels([str(val) for val in train_percentage])
axes[1, 1].set_ylim(50, 100) 
axes[1, 1].legend(loc="lower right", fontsize=font_legend)
axes[1, 1].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[1, 1].set_ylabel('Senzitivita (%)', fontsize=font_legend)

fig.suptitle("Dataset obrazov z histológie - Dátová augmentácia offline", fontsize=font_maintitle)
plt.tight_layout()
plt.show()

#%% NST

seeds = [42,1024,123456] 

data_NST_1 = {}
data_NST_2 = {}
data_NST_3 = {}

data_NST_phase2_1 = {}
data_NST_phase2_2 = {}
data_NST_phase2_3 = {}


# Iterate over the range of numbers
for seed in seeds:
    # Construct the file name
    file_name_1 = f"NST_LC_{seed}"
    file_name_2 = f"NST_LC_{seed}"
    file_name_3 = f"NST_LC_{seed}"
    file_name_4 = f"NST_LC_{seed}_phase2"
    file_name_5 = f"NST_LC_{seed}_phase2"
    file_name_6 = f"NST_LC_{seed}_phase2"
    
    
    # Open the JSON file
    with open(file_name_1, "r") as json_file:
        data_NST_1[seed] = json.load(json_file)
    with open(file_name_2, "r") as json_file:
        data_NST_2[seed] = json.load(json_file)
    with open(file_name_3, "r") as json_file:
        data_NST_3[seed] = json.load(json_file)

    with open(file_name_4, "r") as json_file:
         data_NST_phase2_1[seed] = json.load(json_file)
    with open(file_name_5, "r") as json_file:
         data_NST_phase2_2[seed] = json.load(json_file)
    with open(file_name_6, "r") as json_file:
         data_NST_phase2_3[seed] = json.load(json_file)


#accuracy
data_NST_acc1 =[(next(item for item in data_NST_1[42] if item["train_percentage"] == number_560))["test_acc"],
            (next(item for item in data_NST_1[42] if item["train_percentage"] == number_320))["test_acc"], 
            (next(item for item in data_NST_1[42] if item["train_percentage"] == number_160))["test_acc"], 
            (next(item for item in data_NST_1[42] if item["train_percentage"] == number_80))["test_acc"], 
            (next(item for item in data_NST_1[42] if item["train_percentage"] == number_40))["test_acc"], 
            (next(item for item in data_NST_phase2_1[42] if item["train_percentage"] == number_20))["test_acc"], 
            (next(item for item in data_NST_phase2_1[42] if item["train_percentage"] == number_10))["test_acc"], 
            (next(item for item in data_NST_phase2_1[42] if item["train_percentage"] == number_5))["test_acc"]]
data_NST_acc1 = [acc * 100 for acc in data_NST_acc1]

data_NST_acc2 = [(next(item for item in data_NST_2[1024] if item["train_percentage"] == number_560))["test_acc"],
            (next(item for item in data_NST_2[1024] if item["train_percentage"] == number_320))["test_acc"], 
            (next(item for item in data_NST_2[1024] if item["train_percentage"] == number_160))["test_acc"], 
            (next(item for item in data_NST_2[1024] if item["train_percentage"] == number_80))["test_acc"], 
            (next(item for item in data_NST_2[1024] if item["train_percentage"] == number_40))["test_acc"], 
            (next(item for item in data_NST_phase2_1[1024] if item["train_percentage"] == number_20))["test_acc"], 
            (next(item for item in data_NST_phase2_1[1024] if item["train_percentage"] == number_10))["test_acc"], 
            (next(item for item in data_NST_phase2_1[1024] if item["train_percentage"] == number_5))["test_acc"]]
data_NST_acc2 = [acc * 100 for acc in data_NST_acc2]

data_NST_acc3 = [(next(item for item in data_NST_3[123456] if item["train_percentage"] == number_560))["test_acc"],
            (next(item for item in data_NST_3[123456] if item["train_percentage"] == number_320))["test_acc"], 
            (next(item for item in data_NST_3[123456] if item["train_percentage"] == number_160))["test_acc"], 
            (next(item for item in data_NST_3[123456] if item["train_percentage"] == number_80))["test_acc"], 
            (next(item for item in data_NST_3[123456] if item["train_percentage"] == number_40))["test_acc"], 
            (next(item for item in data_NST_phase2_1[123456] if item["train_percentage"] == number_20))["test_acc"], 
            (next(item for item in data_NST_phase2_1[123456] if item["train_percentage"] == number_10))["test_acc"], 
            (next(item for item in data_NST_phase2_1[123456] if item["train_percentage"] == number_5))["test_acc"]]
data_NST_acc3 = [acc * 100 for acc in data_NST_acc3]


#f1
data_NST_f1_1 =[(next(item for item in data_NST_1[42] if item["train_percentage"] == number_560))["test_f1"],
            (next(item for item in data_NST_1[42] if item["train_percentage"] == number_320))["test_f1"], 
            (next(item for item in data_NST_1[42] if item["train_percentage"] == number_160))["test_f1"], 
            (next(item for item in data_NST_1[42] if item["train_percentage"] == number_80))["test_f1"], 
            (next(item for item in data_NST_1[42] if item["train_percentage"] == number_40))["test_f1"], 
            (next(item for item in data_NST_phase2_1[42] if item["train_percentage"] == number_20))["test_f1"], 
            (next(item for item in data_NST_phase2_1[42] if item["train_percentage"] == number_10))["test_f1"], 
            (next(item for item in data_NST_phase2_1[42] if item["train_percentage"] == number_5))["test_f1"]]
data_NST_f1_1 = [acc * 100 for acc in data_NST_f1_1]

data_NST_f1_2 = [(next(item for item in data_NST_2[1024] if item["train_percentage"] == number_560))["test_f1"],
            (next(item for item in data_NST_2[1024] if item["train_percentage"] == number_320))["test_f1"], 
            (next(item for item in data_NST_2[1024] if item["train_percentage"] == number_160))["test_f1"], 
            (next(item for item in data_NST_2[1024] if item["train_percentage"] == number_80))["test_f1"], 
            (next(item for item in data_NST_2[1024] if item["train_percentage"] == number_40))["test_f1"], 
            (next(item for item in data_NST_phase2_1[1024] if item["train_percentage"] == number_20))["test_f1"], 
            (next(item for item in data_NST_phase2_1[1024] if item["train_percentage"] == number_10))["test_f1"], 
            (next(item for item in data_NST_phase2_1[1024] if item["train_percentage"] == number_5))["test_f1"]]
data_NST_f1_2 = [acc * 100 for acc in data_NST_f1_2]

data_NST_f1_3 = [(next(item for item in data_NST_3[123456] if item["train_percentage"] == number_560))["test_f1"],
            (next(item for item in data_NST_3[123456] if item["train_percentage"] == number_320))["test_f1"], 
            (next(item for item in data_NST_3[123456] if item["train_percentage"] == number_160))["test_f1"], 
            (next(item for item in data_NST_3[123456] if item["train_percentage"] == number_80))["test_f1"], 
            (next(item for item in data_NST_3[123456] if item["train_percentage"] == number_40))["test_f1"], 
            (next(item for item in data_NST_phase2_1[123456] if item["train_percentage"] == number_20))["test_f1"], 
            (next(item for item in data_NST_phase2_1[123456] if item["train_percentage"] == number_10))["test_f1"], 
            (next(item for item in data_NST_phase2_1[123456] if item["train_percentage"] == number_5))["test_f1"]]

data_NST_f1_3 = [acc * 100 for acc in data_NST_f1_3]

#specificity
data_NST_spec_1 =[(next(item for item in data_NST_1[42] if item["train_percentage"] == number_560))["test_spec"],
            (next(item for item in data_NST_1[42] if item["train_percentage"] == number_320))["test_spec"], 
            (next(item for item in data_NST_1[42] if item["train_percentage"] == number_160))["test_spec"], 
            (next(item for item in data_NST_1[42] if item["train_percentage"] == number_80))["test_spec"], 
            (next(item for item in data_NST_1[42] if item["train_percentage"] == number_40))["test_spec"], 
            (next(item for item in data_NST_phase2_1[42] if item["train_percentage"] == number_20))["test_spec"], 
            (next(item for item in data_NST_phase2_1[42] if item["train_percentage"] == number_10))["test_spec"], 
            (next(item for item in data_NST_phase2_1[42] if item["train_percentage"] == number_5))["test_spec"]]
data_NST_spec_1 = [acc * 100 for acc in data_NST_spec_1]

data_NST_spec_2 = [(next(item for item in data_NST_2[1024] if item["train_percentage"] == number_560))["test_spec"],
            (next(item for item in data_NST_2[1024] if item["train_percentage"] == number_320))["test_spec"], 
            (next(item for item in data_NST_2[1024] if item["train_percentage"] == number_160))["test_spec"], 
            (next(item for item in data_NST_2[1024] if item["train_percentage"] == number_80))["test_spec"], 
            (next(item for item in data_NST_2[1024] if item["train_percentage"] == number_40))["test_spec"], 
            (next(item for item in data_NST_phase2_1[1024] if item["train_percentage"] == number_20))["test_spec"], 
            (next(item for item in data_NST_phase2_1[1024] if item["train_percentage"] == number_10))["test_spec"], 
            (next(item for item in data_NST_phase2_1[1024] if item["train_percentage"] == number_5))["test_spec"]]
data_NST_spec_2 = [acc * 100 for acc in data_NST_spec_2]

data_NST_spec_3 = [(next(item for item in data_NST_3[123456] if item["train_percentage"] == number_560))["test_spec"],
            (next(item for item in data_NST_3[123456] if item["train_percentage"] == number_320))["test_spec"], 
            (next(item for item in data_NST_3[123456] if item["train_percentage"] == number_160))["test_spec"], 
            (next(item for item in data_NST_3[123456] if item["train_percentage"] == number_80))["test_spec"], 
            (next(item for item in data_NST_3[123456] if item["train_percentage"] == number_40))["test_spec"], 
            (next(item for item in data_NST_phase2_1[123456] if item["train_percentage"] == number_20))["test_spec"], 
            (next(item for item in data_NST_phase2_1[123456] if item["train_percentage"] == number_10))["test_spec"], 
            (next(item for item in data_NST_phase2_1[123456] if item["train_percentage"] == number_5))["test_spec"]]

data_NST_spec_3 = [acc * 100 for acc in data_NST_spec_3]



#sensitivity
data_NST_sens_1 =[(next(item for item in data_NST_1[42] if item["train_percentage"] == number_560))["test_sens"],
            (next(item for item in data_NST_1[42] if item["train_percentage"] == number_320))["test_sens"], 
            (next(item for item in data_NST_1[42] if item["train_percentage"] == number_160))["test_sens"], 
            (next(item for item in data_NST_1[42] if item["train_percentage"] == number_80))["test_sens"], 
            (next(item for item in data_NST_1[42] if item["train_percentage"] == number_40))["test_sens"], 
            (next(item for item in data_NST_phase2_1[42] if item["train_percentage"] == number_20))["test_sens"], 
            (next(item for item in data_NST_phase2_1[42] if item["train_percentage"] == number_10))["test_sens"], 
            (next(item for item in data_NST_phase2_1[42] if item["train_percentage"] == number_5))["test_sens"]]
data_NST_sens_1 = [acc * 100 for acc in data_NST_sens_1]

data_NST_sens_2 = [(next(item for item in data_NST_2[1024] if item["train_percentage"] == number_560))["test_sens"],
            (next(item for item in data_NST_2[1024] if item["train_percentage"] == number_320))["test_sens"], 
            (next(item for item in data_NST_2[1024] if item["train_percentage"] == number_160))["test_sens"], 
            (next(item for item in data_NST_2[1024] if item["train_percentage"] == number_80))["test_sens"], 
            (next(item for item in data_NST_2[1024] if item["train_percentage"] == number_40))["test_sens"], 
            (next(item for item in data_NST_phase2_1[1024] if item["train_percentage"] == number_20))["test_sens"], 
            (next(item for item in data_NST_phase2_1[1024] if item["train_percentage"] == number_10))["test_sens"], 
            (next(item for item in data_NST_phase2_1[1024] if item["train_percentage"] == number_5))["test_sens"]]
data_NST_sens_2 = [acc * 100 for acc in data_NST_sens_2]

data_NST_sens_3 = [(next(item for item in data_NST_3[123456] if item["train_percentage"] == number_560))["test_sens"],
            (next(item for item in data_NST_3[123456] if item["train_percentage"] == number_320))["test_sens"], 
            (next(item for item in data_NST_3[123456] if item["train_percentage"] == number_160))["test_sens"], 
            (next(item for item in data_NST_3[123456] if item["train_percentage"] == number_80))["test_sens"], 
            (next(item for item in data_NST_3[123456] if item["train_percentage"] == number_40))["test_sens"], 
            (next(item for item in data_NST_phase2_1[123456] if item["train_percentage"] == number_20))["test_sens"], 
            (next(item for item in data_NST_phase2_1[123456] if item["train_percentage"] == number_10))["test_sens"], 
            (next(item for item in data_NST_phase2_1[123456] if item["train_percentage"] == number_5))["test_sens"]]

data_NST_sens_3 = [acc * 100 for acc in data_NST_sens_3]



test_NST_acc = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(data_NST_acc1, data_NST_acc2, data_NST_acc3)]
test_NST_f1 = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(data_NST_f1_1, data_NST_f1_2, data_NST_f1_3)]
test_NST_spec = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(data_NST_spec_1, data_NST_spec_2, data_NST_spec_3)]
test_NST_sens = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(data_NST_sens_1, data_NST_sens_2, data_NST_sens_3)]


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.lineplot(ax=axes[0, 0], x=train_percentage, y=test_acc_50, marker='o', color='orange', label='Bez dátovej augmentácie', linewidth=2)
sns.lineplot(ax=axes[0, 0], x=train_percentage, y=test_NST_acc, marker='o', color='blue', label='NST', linewidth=2)
axes[0, 0].set_title('Presnosť', fontsize=font_subtitle)
axes[0, 0].set_xscale('log')
axes[0, 0].set_xticks(train_percentage)
axes[0, 0].set_xticklabels([str(val) for val in train_percentage])
axes[0, 0].set_ylim(50, 100) 
axes[0, 0].legend(loc="lower right", fontsize=font_legend)
axes[0, 0].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[0, 0].set_ylabel('Presnosť (%)', fontsize=font_legend)

sns.lineplot(ax=axes[0, 1], x=train_percentage, y=test_f1_50, marker='o', color='orange', label='Bez dátovej augmentácie', linewidth=2)
sns.lineplot(ax=axes[0, 1], x=train_percentage, y=test_NST_f1, marker='o', color='blue',label='NST', linewidth=2)
axes[0, 1].set_title('F1 Skóre', fontsize=font_subtitle)
axes[0, 1].set_xscale('log')
axes[0, 1].set_xticks(train_percentage)
axes[0, 1].set_xticklabels([str(val) for val in train_percentage])
axes[0, 1].set_ylim(50, 100) 
axes[0, 1].legend(loc="lower right", fontsize=font_legend)
axes[0, 1].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[0, 1].set_ylabel('F1 Skóre (%)', fontsize=font_legend)  

sns.lineplot(ax=axes[1, 0], x=train_percentage, y=test_spec_50, marker='o', color='orange', label='Bez dátovej augmentácie', linewidth=2)
sns.lineplot(ax=axes[1, 0], x=train_percentage, y=test_NST_spec, marker='o', color='blue', label='NST', linewidth=2)
axes[1, 0].set_title('Špecifita', fontsize=font_subtitle)
axes[1, 0].set_xscale('log')
axes[1, 0].set_xticks(train_percentage)
axes[1, 0].set_xticklabels([str(val) for val in train_percentage])
axes[1, 0].set_ylim(50, 100) 
axes[1, 0].legend(loc="lower right", fontsize=font_legend)
axes[1, 0].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[1, 0].set_ylabel('Špecificita (%)', fontsize=font_legend) 

sns.lineplot(ax=axes[1, 1], x=train_percentage, y=test_sens_50, marker='o', color='orange', label='Bez dátovej augmentácie', linewidth=2)
sns.lineplot(ax=axes[1, 1], x=train_percentage, y=test_NST_sens, marker='o', color='blue', label='NST', linewidth=2)
axes[1, 1].set_title('Senzitivita', fontsize=font_subtitle)
axes[1, 1].set_xscale('log')
axes[1, 1].set_xticks(train_percentage)
axes[1, 1].set_xticklabels([str(val) for val in train_percentage])
axes[1, 1].set_ylim(50, 100) 
axes[1, 1].legend(loc="lower right", fontsize=font_legend)
axes[1, 1].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[1, 1].set_ylabel('Senzitivita (%)', fontsize=font_legend) 

fig.suptitle("Dataset obrazov z histológie - NST", fontsize=font_maintitle)
plt.tight_layout()
plt.show()


#%% Transfer learning

seeds = [42,1024,123456] 

data_TL_1 = {}
data_TL_2 = {}
data_TL_3 = {}


# Iterate over the range of numbers
for seed in seeds:
    # Construct the file name
    file_name_1 = f"TL_LC_{seed}"
    file_name_2 = f"TL_LC_{seed}"
    file_name_3 = f"TL_LC_{seed}"
    
    
    # Open the JSON file
    with open(file_name_1, "r") as json_file:
        data_TL_1[seed] = json.load(json_file)
    with open(file_name_2, "r") as json_file:
        data_TL_2[seed] = json.load(json_file)
    with open(file_name_3, "r") as json_file:
        data_TL_3[seed] = json.load(json_file)


test_TL_acc = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_acc(data_TL_1, 42), 
                                                                  extract_accuracies_acc(data_TL_2, 1024), 
                                                                  extract_accuracies_acc(data_TL_3, 123456))]

test_TL_f1 = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip( extract_accuracies_f1(data_TL_1, 42), 
                                                                 extract_accuracies_f1(data_TL_2, 1024), 
                                                                 extract_accuracies_f1(data_TL_3, 123456))]


test_TL_spec = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip( extract_accuracies_spec(data_TL_1, 42), 
                                                                   extract_accuracies_spec(data_TL_2, 1024), 
                                                                   extract_accuracies_spec(data_TL_3, 123456))]



test_TL_sens = [(s1 + s2 + s3) / 3 for s1, s2, s3 in zip(extract_accuracies_sens(data_TL_1, 42), 
                                                                   extract_accuracies_sens(data_TL_2, 1024), 
                                                                   extract_accuracies_sens(data_TL_3, 123456))]


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.lineplot(ax=axes[0, 0], x=train_percentage, y=test_acc_50, marker='o', color='orange', label='Bez transferového učenia', linewidth=2)
sns.lineplot(ax=axes[0, 0], x=train_percentage, y=test_TL_acc, marker='o', color='blue', label='Transferové učenie', linewidth=2)
axes[0, 0].set_title('Presnosť', fontsize=font_subtitle)
axes[0, 0].set_xscale('log')
axes[0, 0].set_xticks(train_percentage)
axes[0, 0].set_xticklabels([str(val) for val in train_percentage])
axes[0, 0].set_ylim(50, 100) 
axes[0, 0].legend(loc="lower right", fontsize=font_legend)
axes[0, 0].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[0, 0].set_ylabel('Presnosť (%)', fontsize=font_legend)
# for i, value in enumerate(test_acc_50):
#     axes[0, 0].annotate(f'{value:.1f}', (train_percentage[i], test_acc_50[i]), textcoords="offset points", xytext=(0,-10), ha='center')
# for i, value in enumerate(test_TL_acc):
#     axes[0, 0].annotate(f'{value:.1f}', (train_percentage[i], test_TL_acc[i]), textcoords="offset points", xytext=(0,10), ha='center')


sns.lineplot(ax=axes[0, 1], x=train_percentage, y=test_f1_50, marker='o', color='orange', label='Bez transferového učenia', linewidth=2)
sns.lineplot(ax=axes[0, 1], x=train_percentage, y=test_TL_f1, marker='o', color='blue', label='Transferové učenie', linewidth=2)
axes[0, 1].set_title('F1 Skóre', fontsize=font_subtitle)
axes[0, 1].set_xscale('log')
axes[0, 1].set_xticks(train_percentage)
axes[0, 1].set_xticklabels([str(val) for val in train_percentage])
axes[0, 1].set_ylim(50, 100) 
axes[0, 1].legend(loc="lower right", fontsize=font_legend)
axes[0, 1].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[0, 1].set_ylabel('F1 Skóre (%)', fontsize=font_legend)  
# for i, value in enumerate(test_f1_50):
#     axes[0, 1].annotate(f'{value:.1f}', (train_percentage[i], test_f1_50[i]), textcoords="offset points", xytext=(0,-10), ha='center')
# for i, value in enumerate(test_TL_f1):
#     axes[0, 1].annotate(f'{value:.1f}', (train_percentage[i], test_TL_f1[i]), textcoords="offset points", xytext=(0,10), ha='center')



sns.lineplot(ax=axes[1, 0], x=train_percentage, y=test_spec_50, marker='o', color='orange', label='Bez transferového učenia', linewidth=2)
sns.lineplot(ax=axes[1, 0], x=train_percentage, y=test_TL_spec, marker='o', color='blue', label='Transferové učenie', linewidth=2)
axes[1, 0].set_title('Špecifita', fontsize=font_subtitle)
axes[1, 0].set_xscale('log')
axes[1, 0].set_xticks(train_percentage)
axes[1, 0].set_xticklabels([str(val) for val in train_percentage])
axes[1, 0].set_ylim(50, 100) 
axes[1, 0].legend(loc="lower right", fontsize=font_legend)
axes[1, 0].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[1, 0].set_ylabel('Špecificita (%)', fontsize=font_legend) 
# for i, value in enumerate(test_spec_50):
#     axes[1, 0].annotate(f'{value:.1f}', (train_percentage[i], test_spec_50[i]), textcoords="offset points", xytext=(0,-10), ha='center')
# for i, value in enumerate(test_DA_transform_spec):
#     axes[1, 0].annotate(f'{value:.1f}', (train_percentage[i], test_TL_spec[i]), textcoords="offset points", xytext=(0,10), ha='center')



sns.lineplot(ax=axes[1, 1], x=train_percentage, y=test_sens_50, marker='o', color='orange', label='Bez transferového učenia', linewidth=2)
sns.lineplot(ax=axes[1, 1], x=train_percentage, y=test_TL_sens, marker='o', color='blue', label='Transferové učenie', linewidth=2)
axes[1, 1].set_title('Senzitivita', fontsize=font_subtitle)
axes[1, 1].set_xscale('log')
axes[1, 1].set_xticks(train_percentage)
axes[1, 1].set_xticklabels([str(val) for val in train_percentage])
axes[1, 1].set_ylim(50, 100) 
axes[1, 1].legend(loc="lower right", fontsize=font_legend)
axes[1, 1].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[1, 1].set_ylabel('Senzitivita (%)', fontsize=font_legend)  
axes[1, 1].set_xlabel('Počet obrazov na triedu [-]', fontsize=font_legend)
axes[1, 1].set_ylabel('Senzitivita (%)', fontsize=font_legend) 
# for i, value in enumerate(test_sens_50):
#     axes[1, 1].annotate(f'{value:.1f}', (train_percentage[i], test_sens_50[i]), textcoords="offset points", xytext=(0,-10), ha='center')
# for i, value in enumerate(test_TL_sens):
#     axes[1, 1].annotate(f'{value:.1f}', (train_percentage[i], test_TL_sens[i]), textcoords="offset points", xytext=(0,10), ha='center')



fig.suptitle("Dataset obrazov z histológie - Transferové učenie", fontsize=font_maintitle)
plt.tight_layout()
plt.show()


#%% Groupped plotting


# techniques = ['DA online', 'DA offline', 'NST', '50 epoch', 'Transfer learning', 'Dropout']
# test_results = [test_DA_transform_acc, test_DA_offline_acc, test_NST_acc, test_acc_50, test_TL_acc, test_drop_acc]

#accuracy

# techniques_1 = ['50 epoch', 'Transfer learning', 'Dropout']
# test_results_1 = [test_acc_50, test_TL_acc, test_drop_acc]
# colors_1 = ['orange', 'blue', '#aec7e8']


# techniques_2 = ['50 epoch', 'DA online', 'DA offline', 'NST',]
# test_results_2 = [test_acc_50, test_DA_transform_acc, test_DA_offline_acc, test_NST_acc]
# colors_2 = ['orange', '#2ca02c', '#E2F0CB', '#d62728' ] 

#f1s
techniques_1 = ['ResNet18', 'Transfer learning']
test_results_1 = [test_f1_50, test_TL_f1]
colors_1 = ['orange', 'blue']


techniques_2 = ['ResNet18', 'DA online', 'DA offline', 'NST',]
test_results_2 = [test_f1_50, test_DA_transform_f1, test_DA_offline_f1, test_NST_f1]
colors_2 = ['orange', '#2ca02c', '#E2F0CB', '#d62728' ] 

bar_width = 0.18



# Create figure and axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 8))

# First subplot
for i, result in enumerate(test_results_1):
    x = [val + i * bar_width for val in range(len(train_percentage))]
    bars = ax1.bar(x, result, width=bar_width, label=techniques_1[i], color=colors_1[i])
    for bar in bars:
        height = bar.get_height()
        ax1.annotate('{:.1f}'.format(height),
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),  
                      textcoords="offset points",
                      ha='center', va='bottom')

ax1.set_xlabel('Počet obrazov na triedu [-]', fontsize=font_subtitle)
ax1.set_ylabel('F1 skóre (%)', fontsize=font_subtitle) 
ax1.set_title('Techniky na úpravu algoritmu', fontsize=font_maintitle)
ax1.set_xticks([val + bar_width * (len(test_results_1) - 1) / 2 for val in range(len(train_percentage))])
ax1.set_xticklabels([str(val) for val in train_percentage])
ax1.legend(loc='lower right')
ax1.invert_xaxis()
ax1.grid(False)

# Second subplot
for i, result in enumerate(test_results_2):
    x = [val + i * bar_width for val in range(len(train_percentage))]
    bars = ax2.bar(x, result, width=bar_width, label=techniques_2[i], color=colors_2[i])
    for bar in bars:
        height = bar.get_height()
        ax2.annotate('{:.1f}'.format(height),
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),  
                      textcoords="offset points",
                      ha='center', va='bottom')

ax2.set_xlabel('Počet obrazov na triedu [-]', fontsize=font_subtitle)
ax2.set_ylabel('F1 skóre (%)', fontsize=font_subtitle) 
ax2.set_title('Techniky na úpravu datasetu', fontsize=font_maintitle)
ax2.set_xticks([val + bar_width * (len(test_results_2) - 1) / 2 for val in range(len(train_percentage))])
ax2.set_xticklabels([str(val) for val in train_percentage])
ax2.legend(loc='lower right')
ax2.grid(False)
ax2.invert_xaxis()

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('lightgray')  # You can use 'none' to make it invisible
ax1.spines['bottom'].set_color('lightgray')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('lightgray')
ax2.spines['bottom'].set_color('lightgray')

fig.suptitle("Dataset obrazov z histológie", fontsize=font_maintitle)

plt.tight_layout(pad = 1.2)
plt.show()




