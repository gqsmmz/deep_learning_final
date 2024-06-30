# data.py
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet
import zipfile
import urllib.request
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms


class SimCLRTransform:
    def __init__(self, size=32):
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

def get_cifar10_train_loader(batch_size=256, num_workers=4, download=True, data_scale=1.0):
    if data_scale <= 0 or data_scale > 1:
        raise ValueError("Data scale must be a positive number less than or equal to 1.")

    # Calculate the number of samples based on the scale
    total_samples = int(len(datasets.CIFAR10(root='./data', train=True, download=download)) * data_scale)

    transform = SimCLRTransform()  # Replace with your actual transformation

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=download)

    # Subset the dataset based on the calculated number of samples
    train_dataset.data = train_dataset.data[:total_samples]
    train_dataset.targets = train_dataset.targets[:total_samples]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return train_loader


mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

cifar100_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

def get_cifar100_train_loader(batch_size=256, num_workers=4, download=True):
    train_dataset = datasets.CIFAR100(root='./data', train=True, transform=cifar100_transform, download=download)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader

def get_cifar100_test_loader(batch_size=256, num_workers=4, download=True):
    # eval_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    # ])
    eval_dataset = datasets.CIFAR100(root='./data', train=False, transform=cifar100_transform, download=download)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return eval_loader


def tiny_imagenet_prepare(data_dir='./data/tiny-imagenet'):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)
    else:
        print('Tiny ImageNet zip file already exists.')

    extract_path = os.path.join(data_dir, 'tiny-imagenet-200')
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    else:
        print('Tiny ImageNet directory already exists.')
    return extract_path

def get_tiny_imagenet_loader(batch_size=256, num_workers=4, download=True):
    # data_dir = tiny_imagenet_prepare()  # Ensure data is downloaded and extracted
    data_dir='./data/tiny-imagenet/tiny-imagenet-200'
    transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]),
    ])
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader



if __name__=='__main__':
    tiny_imagenet_prepare(data_dir='./data/tiny-imagenet')