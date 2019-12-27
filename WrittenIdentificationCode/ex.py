#split into test train and validation in ratio 4:1:1

import pickle
from PIL import Image
import numpy as np
import sklearn
import torch

img_files_in = open('img_files.dat', 'rb')
encoded_Y_in = open('encoded_Y.dat', 'rb')

img_files = pickle.load(img_files_in)
encoded_Y = pickle.load(encoded_Y_in)

img_files_in.close()
encoded_Y_in.close()

from sklearn.model_selection import train_test_split
train_files, rem_files, train_targets, rem_targets = train_test_split(
        img_files, encoded_Y, train_size=0.66, random_state=52, shuffle= True)

validation_files, test_files, validation_targets, test_targets = train_test_split(
        rem_files, rem_targets, train_size=0.5, random_state=22, shuffle=True)

# print(train_files.size, validation_files.size, test_files.size)
# print(train_targets.size, validation_targets.size, test_targets.size)
# print(train_files)

def generate_data(samples, target_files):
    data = []
    print(samples)
    count = 0
    for i in range(len(samples)):
        im = Image.open(samples[i])
        img_np = np.array(im)
        targets_np = np.array(target_files[i])
        data.append([img_np, targets_np])
        count += 1
        if count % 100 == 0:
            print(count)
    return data


print('Create dataset.')
train_dataset = generate_data(train_files, train_targets)
validation_dataset = generate_data(validation_files, validation_targets)
test_dataset = generate_data(test_files, test_targets)

print('Finish create dataset.')
print('start save dataset')

train_dataset_out = open('train_dataset2.dat', 'wb')
validation_dataset_out = open('validation_dataset2.dat', 'wb')
test_dataset_out = open('test_dataset2.dat', 'wb')

pickle.dump(train_dataset, train_dataset_out)
pickle.dump(validation_dataset, validation_dataset_out)
pickle.dump(test_dataset, test_dataset_out)

validation_dataset_out.close()
test_dataset_out.close()
train_dataset_out.close()
print('finish save dataset')


print('Finish generate data')
batch_size = 100
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
print('Finish Load dataset')
