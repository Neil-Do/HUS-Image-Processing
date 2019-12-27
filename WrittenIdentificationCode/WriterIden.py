#from google.colab import drive
#drive.mount('/content/drive')
import os
import glob
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
#os.chdir('/content/drive/My Drive/Dataset/preprocess-image/')
PATH_OF_DATA= '/media/neil-do/Intersection/MegaSync/MEGAsync/Study/Artificial Intelligence/Dataset/iam-handwriting-top50/preprocess-image/'

# Create sentence writer mapping
#Dictionary with form and writer mapping
d = {}
with open('/media/neil-do/Intersection/MegaSync/MEGAsync/Study/Artificial Intelligence/Dataset/iam-handwriting-top50/forms_for_parsing.txt') as f:
    for line in f:
        key = line.split(' ')[0]
        writer = line.split(' ')[1]
        d[key] = writer

count = 0
'''
for img_path in glob.glob(PATH_OF_DATA + '*.png'):
    img_name = img_path.split('/')[-1]
    name, extension = img_name.split('.')
    count += 1
    if count
'''
print('Finish image name and writer hash map from forms_for_parsing')


# Create array of file names and corresponding target writer names
tmp = []
target_list = []
path_to_files = os.path.join(PATH_OF_DATA, '*')
for filename in sorted(glob.glob(path_to_files)):
    tmp.append(filename)
    image_name = filename.split('/')[-1]
    file, ext = os.path.splitext(image_name)
    parts = file.split('-')
    form = parts[0] + '-' + parts[1]
    for key in d:
        if key == form:
            target_list.append(str(d[form]))

img_files = np.asarray(tmp) # list of filenames
img_targets = np.asarray(target_list) # list of writer associate with img_files has same index
print('Finish create list of image files and associate writer')


# Visualizing sample image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
from sklearn.preprocessing import LabelEncoder

# Visualizing the data
for filename in img_files[:3]:
    img=mpimg.imread(filename)
    plt.figure(figsize=(10,10))
    #plt.imshow(img, cmap ='gray')

# Label Encode writer names for one hot encoding later
encoder = LabelEncoder()
encoder.fit(img_targets)
encoded_Y = encoder.transform(img_targets)

print(img_files[:5], img_targets[:5], encoded_Y[:5])


#split into test train and validation in ratio 4:1:1

from sklearn.model_selection import train_test_split
train_files, rem_files, train_targets, rem_targets = train_test_split(
        img_files, encoded_Y, train_size=0.66, random_state=52, shuffle= True)

validation_files, test_files, validation_targets, test_targets = train_test_split(
        rem_files, rem_targets, train_size=0.5, random_state=22, shuffle=True)

print(train_files.shape, validation_files.shape, test_files.shape)
print(train_targets.shape, validation_targets.shape, test_targets.shape)






print('Fin generator')




__all__ = ['DeepWriter', 'deepwriter']

class DeepWriter(nn.Module):

    def __init__(self, num_classes=10):
        super(DeepWriter, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = DeepWriter(50)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

nb_epoch = 8

samples_per_epoch = 3268
nb_val_samples = 842
print('Fin create Model')
