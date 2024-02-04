import pandas as pd
import numpy as np
from scipy import stats
import os
from PIL import Image
import random

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

BASE_DIR = 'c:\\Users\\Zhenzhen\\Desktop\\University\\M.Sc. Tübingen\\2023-2024 WS\\Data Science Project\\temp'

df = pd.read_csv('survey.csv', sep=';')

#################################
## DATA PREPROCESSING
#################################

# Define preprocessing function
def preprocess(data):
    # Remove participants who took less than 100 seconds to complete the survey
    data = data[data['duration'] > 100]
    # Remove irrelevant columns
    data = data.drop(data.columns[0:7], axis=1)
    data = data.drop(data.columns[819:], axis=1)
    # Replace all -77 with NaN
    data = data.replace(-77, np.nan)

    # Reshaping the data
    # Identify all v_66_... columns
    cols = [col for col in data.columns if col.startswith('v_66_')]
    data_list = []
    # Loop over each v_66_ column and calculate the means
    for col in cols:
        row = {
            'image': col,
            'mean': data[col].mean(),
            'stuttgart_yes': data[data['v_56'] == 1][col].mean(),
            'stuttgart_no': data[data['v_56'] == 2][col].mean(),
            'commute': data[data['v_67'] == 1][col].mean(),
            'recreation': data[data['v_67'] == 2][col].mean(),
            'equally': data[data['v_67'] == 3][col].mean(),
            'rarely': data[data['v_67'] == 4][col].mean()
        }
        data_list.append(row)

    # Create a DataFrame from the list of dictionaries
    data = pd.DataFrame(data_list)

    return data

df_new = preprocess(df)



#################################
## ADD SCORES TO EACH IMAGE
#################################
REF_DIR = os.path.join(BASE_DIR, r'Images\sampled_edges.csv')
ref = pd.read_csv(REF_DIR, sep=';', header=None)
ref = ref.drop(ref.columns[3:8], axis=1)
ref.columns = ['cluster_num', 'survey_id', 'image_name']
ref['survey_id'] = ref['survey_id'].astype(str)

scores = df_new.drop(df_new.columns[2:8], axis=1)
scores['image'] = scores['image'].str.split('_').str[-1]

base_scores = pd.merge(ref, scores, left_on='survey_id', right_on='image', how='left')
base_scores = base_scores.drop('image', axis=1)



#################################
## HYPOTHESIS TESTING
#################################

# Define hypothesis testing function
def hypothesis_testing(data, column1, column2):
    # H0: The mean of the columns are equal
    # Sig. level
    alpha = 0.05
    
    # Perform an Independent Samples t-test
    t_stat, p_val = stats.ttest_ind(data[column1], data[column2],
                                    equal_var=False, nan_policy='omit')

    # Print the results
    print('t-statistic: ', t_stat)
    print('p-value: ', p_val)
    if p_val < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')

        
hypothesis_testing(df_new, 'mean', 'stuttgart_yes')
# t-statistic:  -2.8888115017587555
# p-value:  0.003926827267292419
# Reject the null hypothesis

hypothesis_testing(df_new, 'stuttgart_yes', 'stuttgart_no')
# t-statistic:  3.37113226546661
# p-value:  0.0007689703590340253
# Reject the null hypothesis

hypothesis_testing(df_new, 'mean', 'recreation')
# t-statistic:  -2.134236249734194
# p-value:  0.03298612528163634
# Reject the null hypothesis

hypothesis_testing(df_new, 'commute', 'recreation')
# t-statistic:  -2.55847448447655
# p-value:  0.010607484225381554
# Reject the null hypothesis



#################################
## 
## SURVEY MODEL
##
#################################

#################################
## PARAMETER SETTINGS
#################################

IMAGE_DIR = os.path.join(BASE_DIR, r'Images\survey_images')

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TYPE = "regression"
NUM_CLASSES = 1
LR = 3e-5  # Already trained model --> smaller learning rate
NUM_EPOCHS = 15

IMG_SIZE = (224, 224)
BATCH_SIZE = 64

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]



#################################
## CREATING DATASETS & DATALOADERS
#################################

# Train, validation, test split
train_data, temp_test_data = train_test_split(base_scores, test_size=0.3, random_state=SEED)
valid_data, test_data = train_test_split(temp_test_data, test_size=2/3, random_state=SEED)

# Define dataset class
class SurveyDataset(Dataset):
    def __init__(self, data, img_dir, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 2])
        image = Image.open(img_name) #.convert('RGB')
        label = self.data.iloc[idx]['mean']

        if self.transform:
            image = self.transform(image)

        return image, label
    
   
# Define the transformations
train_transform = transforms.Compose([transforms.Resize(size=IMG_SIZE),
                                      transforms.RandomVerticalFlip(p=0.3),
                                      transforms.RandomRotation(degrees=15),
                                      transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.9,1)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=MEAN, std=STD)
                                      ])

test_transform = transforms.Compose([transforms.Resize(size=IMG_SIZE),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=MEAN, std=STD)
                                     ])

# Instantiate the datasets
train_dataset = SurveyDataset(data=train_data, img_dir=IMAGE_DIR, transform=train_transform)
valid_dataset = SurveyDataset(data=valid_data, img_dir=IMAGE_DIR, transform=test_transform)
test_dataset = SurveyDataset(data=test_data, img_dir=IMAGE_DIR, transform=test_transform)

# Instantiate the dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)



#################################
## LOAD PRETRAINED MODEL
#################################



