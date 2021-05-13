import torch
import cv2
import os
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, data_root_dir='/content/drive/MyDrive/Colab Notebooks/ASL-main/COMP5329S1A2Dataset', train=True):
        self.data_root_dir = data_root_dir
        self.train = train

        # some parameter deine
        # assuming same from the spec, have 20 classes
        self.N_class = 20
        # train/valid split percentage, value equals using train/total train set
        train_valid_percentage = 0.85

        self.train_csv = pd.read_csv(os.path.join(self.data_root_dir, 'train.csv'))

        # read data form file, make binary array for all classes
        self.train_csv_image_names = self.train_csv[:]['ImageID']
        
        #self.train_csv_label = self.train_csv[:]['Labels'].str.split(' ')
        
        train_csv_label = self.train_csv[:]['Labels'].str.split(' ')
        np_train_csv_label = np.zeros((train_csv_label.shape[0],self.N_class), dtype=int)
        for outter_it in range(train_csv_label.shape[0]):
            for inner_it in train_csv_label[outter_it]:
                np_train_csv_label[outter_it][int(inner_it)] = 1

        self.train_csv_label = np_train_csv_label

        #class_mask = self.train_csv_label.sum(axis=0)
        #print(class_mask)

        self.Total_N_train_sample = self.train_csv_image_names.shape[0]
        self.train_valid_plit = int(train_valid_percentage * self.Total_N_train_sample)

        # set the training data images and labels
        if self.train == True:
            print(f"Number of training images: {self.train_valid_plit}")
            self.image_names = list(self.train_csv_image_names[:self.train_valid_plit])
            self.labels = list(self.train_csv_label[:self.train_valid_plit])
            # define the training transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((200, 200)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
            ])
        # set the validation data images and labels
        else:
            print(f"Number of validation images: {self.Total_N_train_sample-self.train_valid_plit}")
            self.image_names = list(self.train_csv_image_names[self.train_valid_plit:])
            self.labels = list(self.train_csv_label[self.train_valid_plit:])
            # define the validation transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((200, 200)),
                transforms.ToTensor(),
            ])
        

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        print(f"{self.image_names[index]}")
        
        #trian/val set
        image = cv2.imread(f"/content/COMP5329S1A2Dataset/data/{self.image_names[index]}")
        #cv2.imshow(f"{self.image_names[index]}", image)
        #print(image)

        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        targets = self.labels[index]
        
        return  torch.tensor(image, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)