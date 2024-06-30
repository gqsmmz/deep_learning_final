# evaluate.py
import argparse
import torch
from data import get_cifar100_test_loader, get_cifar100_train_loader, get_cifar10_train_loader, get_tiny_imagenet_loader
from model import SimCLR, load_resnet
from torch.utils.tensorboard import SummaryWriter
import os
from torchvision import datasets, transforms, models
from evaluate import supervised_linear_classifier
import torch.nn as nn
from simclr_train import SimCLR
from simclr_test import LinModel
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10,CIFAR100
from simclr_test import finetune
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model.')
    parser.add_argument('--train_type', type=str, choices=['selfsupervised', 'supervised'], default='selfsupervised', help='Type of model to evaluate')
    parser.add_argument('--dataset', type=str, choices=['cifar100', 'cifar100', 'imagenet'], default='cifar10', help='Dataset to use for evaluation')
    parser.add_argument('--model_path', type=str, default='try1.pth', help='Path to the model checkpoint')
    parser.add_argument('--train_dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'imagenet'], help='Dataset used for training the model')
    parser.add_argument('--test_num_epochs', type=int, default=10, help='Number of epochs for fine-tuning on CIFAR-100')
    parser.add_argument('--test_learning_rate', type=float, default=1e-3, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for testing')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Determine the number of classes based on the training dataset
    if args.train_dataset == 'cifar100':
        num_classes = 100
    elif args.train_dataset == 'cifar10':
        num_classes = 10
    elif args.train_dataset == 'imagenet':
        num_classes = 200  # Assuming Tiny ImageNet has 200 classes

    # Set up data loaders based on the evaluation dataset
    if args.dataset == 'cifar100':  #检验都用这个
        eval_loader = get_cifar100_test_loader(args.batch_size)
        train_loader = get_cifar100_train_loader(args.batch_size)
    elif args.dataset == 'cifar10':
        # eval_loader = get_cifar10_test_loader(args.batch_size)
        # train_loader = get_cifar10_train_loader(args.batch_size)
        train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor()])
        test_transform = transforms.ToTensor()
        data_dir='./data'
        train_set = CIFAR10(root=data_dir, train=True, transform=train_transform, download=False)
        test_set = CIFAR10(root=data_dir, train=False, transform=test_transform, download=False)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True)
        eval_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)




    # TensorBoard writer path and naming
    log_dir = os.path.join('runs', 'eval_sup')
    os.makedirs(log_dir, exist_ok=True)
    log_name = f"{args.train_type}_testdataset_{args.dataset}_traindataset_{args.train_dataset}_test_epochs_{args.test_num_epochs}_lr_{args.test_learning_rate}_bs_{args.batch_size}"
    print("log name",log_name)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, log_name))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if args.train_type == 'selfsupervised':

        # # Load Pre-trained ResNet-18 from SimCLR
        # base_model = models.resnet18(pretrained=False)  # Load without pretrained weights
        # simclr_model = SimCLR(base_model).to(device)
        # simclr_model.load_state_dict(torch.load(args.model_path))
        # simclr_model.eval()

        # # linear_classifier = LinearClassifier(simclr_model.encoder).to(device)
        # evaluate_linear_classifier(simclr_model,train_loader,eval_loader,device=device,writer=writer)
        finetune()



    elif args.train_type == 'supervised':
        # resnet_model = models.resnet18(pretrained=True).to(device)
        # resnet_model.eval()
        # for param in resnet_model.parameters(): #固定参数
        #     print(param.names)  #咋都是nan
        #     param.requires_grad = False

        resnet_model = models.resnet18(pretrained=False, num_classes=num_classes).to(device)
        resnet_model.load_state_dict(torch.load(args.model_path))
        # resnet_model.eval()
        for param in resnet_model.parameters(): #固定参数
            print(param.names)  #咋都是nan
            param.requires_grad = False
        fc_inputs = resnet_model.fc.in_features # 保持与前面第一步中的代码一致
        resnet_model.fc = nn.Sequential(         #
        nn.Linear(fc_inputs, 100),  #
        nn.LogSoftmax(dim=1))
        resnet_model = resnet_model.to('cuda')

        supervised_linear_classifier(resnet_model,train_loader, eval_loader, num_classes,num_epochs=args.test_num_epochs, lr=args.test_learning_rate, device='cuda',writer=writer)




if __name__ == '__main__':
    main()
