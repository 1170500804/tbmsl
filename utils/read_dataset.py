import torch
import os
from datasets import dataset
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms

def read_dataset(input_size, batch_size, root, set, regression=False):
    if set == 'CUB':
        print('Loading CUB trainset')
        trainset = dataset.CUB(input_size=input_size, root=root, is_train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False)
        print('Loading CUB testset')
        testset = dataset.CUB(input_size=input_size, root=root, is_train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)
    elif set == 'CAR':
        print('Loading car trainset')
        trainset = dataset.STANFORD_CAR(input_size=input_size, root=root, is_train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False)
        print('Loading car testset')
        testset = dataset.STANFORD_CAR(input_size=input_size, root=root, is_train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)
    elif set == 'Aircraft':
        print('Loading Aircraft trainset')
        trainset = dataset.FGVC_aircraft(input_size=input_size, root=root, is_train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False)
        print('Loading Aircraft testset')
        testset = dataset.FGVC_aircraft(input_size=input_size, root=root, is_train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)
    elif set == 'Floor_Ele':
        print('Loading Floor_Ele trainset')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                normalize]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                normalize])
        #TODO: change image folder
        image_folder = '/home/liushuai/Streetview_Irma/images'
        train_data = 'elevation_train.csv'
        val_data = 'elevation_val.csv'
        train_dataset = Floor_Ele(train_data, image_folder, transform=train_transform,
                                  regression=regression)
        val_dataset = Floor_Ele(val_data, image_folder, transform=val_transform, regression=regression)

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        testloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    else:
        print('Please choose supported dataset')
        os._exit()

    return trainloader, testloader

class Floor_Ele(Dataset):
    def __init__(self, csv_path, img_path, transform=None, regression = False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.regression = regression

        # self.classes = np.sort(self.df['first_floor_elevation_ft'].unique())
        self.classes = [0., 1., 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5,
                         4., 4.5, 5., 6., 7., 8., 8.5, 9., 10.,
                         11., 12., 13., 14.]  # No dataset alone has all the clases therefore hard define this here

        self.img_path = img_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df.iloc[idx]['filename']
        image = Image.open(os.path.join(self.img_path, img_name))
        elevation = self.df.iloc[idx]['first_floor_elevation_ft']
        if self.regression:
            label = torch.from_numpy(np.asarray(elevation))
        else:
            label = np.flatnonzero(
                self.classes == elevation)  # Flatnonzero is the sane version of np.where which does not return weird tuples
            label = label.squeeze()

        if (self.transform):
            image = self.transform(image)

        return (image, label)