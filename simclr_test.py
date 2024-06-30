import hydra
from omegaconf import DictConfig
import logging
import numpy as np
from PIL import Image
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SubsetRandomSampler,Dataset
from torchvision.datasets import CIFAR10,CIFAR100,ImageFolder
from torchvision import transforms
from torchvision.models import resnet18, resnet34
from model import SimCLR
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter




logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    
class LinModel(nn.Module):
    """Linear wrapper of encoder."""
    def __init__(self, encoder: nn.Module, feature_dim: int, n_classes: int):
        super().__init__()
        self.enc = encoder
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.lin = nn.Linear(self.feature_dim, self.n_classes)

    def forward(self, x):
        return self.lin(self.enc(x))


def run_epoch(model, dataloader, epoch, optimizer=None, scheduler=None,writer=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    if writer:
        writer = writer
    else:
        writer = SummaryWriter()  # TensorBoard writer
    
    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    loader_bar = tqdm(dataloader)
    for x, y in loader_bar:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        acc = (logits.argmax(dim=1) == y).float().mean()
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc.item(), x.size(0))
        if optimizer:  #训练
            loader_bar.set_description("Train epoch {}, loss: {:.4f}, acc: {:.4f}"
                                       .format(epoch, loss_meter.avg, acc_meter.avg))
            writer.add_scalar('finetune_Loss/train', loss_meter.avg, epoch + 1)
            writer.add_scalar('finetune_Accuracy/train', acc_meter.avg, epoch + 1)
        else:  #测试
            loader_bar.set_description("Test epoch {}, loss: {:.4f}, acc: {:.4f}"
                                       .format(epoch, loss_meter.avg, acc_meter.avg))
            writer.add_scalar('test_Loss/train', loss_meter.avg, epoch + 1)
            writer.add_scalar('test_Accuracy/train', acc_meter.avg, epoch + 1)

    return loss_meter.avg, acc_meter.avg


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


@hydra.main(config_path='./simclr.yml')  #注意这里路径要写清楚，不然识别不到
def finetune(args: DictConfig) -> None:
    train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor()])
    test_transform = transforms.ToTensor()

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    #采用自带的Cifar100
    train_set = CIFAR100(root='/home/datadisk3/lss/zhangli_final/mission_1/data', train=True, download=True, transform=transform)
    # 选择数据集的子集
    # n_samples = len(train_set)
    # subset_indices = np.random.choice(n_samples, size=1000, replace=False)  # 选择 10000 个样本
    # subset_sampler = SubsetRandomSampler(subset_indices)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32) #, sampler=subset_sampler)

    # test_set = CIFAR100(root='/home/datadisk3/lss/zhangli_final/mission_1/data', train=False, download=True, transform=transform)  #transform可以用一般的
    # n_samples_test = len(test_set)
    # test_subset_indices = np.random.choice(n_samples_test, size=1000, replace=False)  # 选择 1000 个样本
    # test_subset_sampler = SubsetRandomSampler(test_subset_indices)
    # test_loader = DataLoader(test_set, batch_size=32, sampler=test_subset_sampler)

    test_set = CIFAR100(root='/home/datadisk3/lss/zhangli_final/mission_1/data', train=False, download=True, transform=transform)  #transform可以用一般的
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)  

    # train_set = CIFAR10(root=data_dir, train=True, transform=train_transform, download=False)
    # test_set = CIFAR10(root=data_dir, train=False, transform=test_transform, download=False)

    # # n_classes = 10
    # # indices = np.random.choice(len(train_set), 10*n_classes, replace=False)
    # # sampler = SubsetRandomSampler(indices)
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Prepare model
    base_encoder = eval(args.backbone)
    pre_model = SimCLR(base_encoder, projection_dim=args.projection_dim).cuda()

    pre_model.load_state_dict(torch.load('/home/datadisk3/lss/zhangli_final/mission_1/logs_imagenetreal/SimCLR/imagenet/simclr_resnet18_epoch250.pt'))
    # pre_model.load_state_dict(torch.load('/home/datadisk3/lss/zhangli_final/mission_1_1/SimCLR-CIFAR10-master/logs_imagenet/SimCLR/cifar10/simclr_resnet18_epoch50.pt'))
    # pre_model.load_state_dict(torch.load('/home/datadisk3/lss/zhangli_final/mission_1_1/SimCLR-CIFAR10-master/logs/SimCLR/cifar10/simclr_resnet18_epoch1000.pt'))
    # pre_model.load_state_dict(torch.load('/home/datadisk3/lss/zhangli_final/mission_1/logs/SimCLR/cifar10/simclr_{}_epoch{}.pt'.format(args.backbone, args.load_epoch)))
    model = LinModel(pre_model.enc, feature_dim=pre_model.feature_dim, n_classes=len(train_set.targets))
    

    # # Load pretrained SimCLR model
    # pretrained_path = '/home/datadisk3/lss/zhangli_final/mission_1/logs/SimCLR/cifar10/simclr_lin_resnet18_best.pth'
    # if os.path.exists(pretrained_path):
    #     model.load_state_dict(torch.load(pretrained_path))
    #     logger.info(f"Loaded pretrained model from {pretrained_path}")
    # else:
    #     logger.warning(f"Pretrained model checkpoint '{pretrained_path}' not found.")

    model = model.cuda()

    # Fix encoder
    model.enc.requires_grad = False
    parameters = [param for param in model.parameters() if param.requires_grad is True]  # trainable parameters.
    # optimizer = Adam(parameters, lr=0.001)

    optimizer = torch.optim.SGD(
        parameters,
        0.2,   # lr = 0.1 * batch_size / 256, see section B.6 and B.7 of SimCLR paper.
        momentum=args.momentum,
        weight_decay=0.,
        nesterov=True)

    # cosine annealing lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            args.learning_rate,  # lr_lambda computes multiplicative factor
            1e-3))

    optimal_loss, optimal_acc = 1e5, 0.  #设置初始值
    
    log_dir = os.path.join('runs', 'eval_sup')
    os.makedirs(log_dir, exist_ok=True)
    # log_name = "selfsupervised_cifar10_cifar100_lr_0.075_bs_32"
    log_name = "selfsupervised_imagenet2_cifar100_lr_0.075_bs_32"
    writer = SummaryWriter(log_dir=os.path.join(log_dir, log_name))

    for epoch in range(1, args.finetune_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, epoch, optimizer, scheduler,writer=writer)
        test_loss, test_acc = run_epoch(model, test_loader, epoch,writer=writer)

        # if epoch==24:
        #     optimal_loss=train_loss
        if train_loss < optimal_loss:  
            optimal_loss = train_loss
            optimal_acc = test_acc
            logger.info("==> New best results")
            torch.save(model.state_dict(), 'simclr_lin_{}_best.pth'.format(args.backbone))

    logger.info("Best Test Acc: {:.4f}".format(optimal_acc))
    writer.close()


if __name__ == '__main__':
    finetune()


