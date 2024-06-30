# train.py
import argparse
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import models
import os
from model import SimCLR, train_supervised
from torch.utils.tensorboard import SummaryWriter
from data import (
    get_cifar100_train_loader,
    get_cifar10_train_loader,
    get_tiny_imagenet_loader
)
from model import SimCLR
from simclr_train import train

def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet-18 model.')
    parser.add_argument('--train_type', type=str, choices=['selfsupervised', 'supervised'], default='selfsupervised', help='Training type')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet'], default='cifar10', help='Dataset to use')
    parser.add_argument('--pretrained', type=str, choices=['true', 'false'], default='true', help='Use pretrained model or not')
    parser.add_argument('--data_scale', type=float, default=1.0, help='Scale of the dataset to use (fraction)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Determine the number of classes based on the dataset
    if args.dataset == 'cifar100': #监督学习
        num_classes = 100
        train_loader = get_cifar100_train_loader(args.batch_size)
    elif args.dataset == 'cifar10':   #自监督学习
        num_classes = 10
        train_loader = get_cifar10_train_loader(args.batch_size)
    elif args.dataset == 'imagenet': #监督学习
        # Adjust the number of classes according to the ImageNet dataset
        num_classes = 200
        train_loader = get_tiny_imagenet_loader(args.batch_size)

    pretrained = True if args.pretrained.lower() == 'true' else False

    # TensorBoard writer path and naming
    log_dir = os.path.join('runs', 'train_sup')
    os.makedirs(log_dir, exist_ok=True)
    log_name = f"{args.train_type}_{args.dataset}_pretrained_{args.pretrained}_batch_size_{args.batch_size}_epochs_{args.epochs}_lr_{args.learning_rate}"
    writer = SummaryWriter(log_dir=os.path.join(log_dir, log_name))


    if args.train_type == 'selfsupervised':
        base_encoder = eval('resnet18')
        model = SimCLR(base_encoder).cuda()
        train()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # base_model = models.resnet18(pretrained=False)
        # model = SimCLR(base_model).to(device)  ##恒等化最后一层线性层后（512，1000）变为（512，512），加上映射层（512，out_dim），out_dim目前设为128
        # train_simclr(model,train_loader,epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,device=device,writer=writer)

        # # Save SimCLR model checkpoint
        # checkpoint_name = f'selfsupervised_{args.dataset}_pretrained_{args.pretrained}_data_scale_{args.data_scale}_batch_size_{args.batch_size}_epochs_{args.epochs}_lr_{args.learning_rate}.pth'
        # checkpoint_path = os.path.join('checkpoints', 'selfsupervised', checkpoint_name)
        # torch.save(model.state_dict(), checkpoint_path)
        # print(f'Saved SimCLR model checkpoint at: {checkpoint_path}')

    elif args.train_type == 'supervised':
        if pretrained:
            print("初始化参数pretrained=true")
            model = models.resnet18(pretrained=pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)  #把最后一层调整为训练数据集的类别数，最后一层需要训练，其他层代入预训练参数
            model = model.to(device)
        else:
            print("初始化参数pretrained=false")
            model = models.resnet18(pretrained=pretrained, num_classes=num_classes).to(device)  #只需要把最后一层换成类别数，都不设参数
        
        train_supervised(model,train_loader,epochs=args.epochs,learning_rate=args.learning_rate,device=device,
                         dataset=args.dataset,pretrained=args.pretrained,data_scale=args.data_scale,batch_size=args.batch_size,
                         writer=writer)

        # # Save supervised ResNet model checkpoint
        checkpoint_name = f'supervised_{args.dataset}_pretrained_{args.pretrained}_data_scale_{args.data_scale}_batch_size_{args.batch_size}_epochs_{args.epochs}_lr_{args.learning_rate}.pth'
        checkpoint_path = os.path.join('checkpoints', 'supervised', checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saved SimCLR model checkpoint at: {checkpoint_path}')

if __name__ == '__main__':
    main()
