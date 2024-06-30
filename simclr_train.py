import hydra
from omegaconf import DictConfig
import logging

import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import CIFAR10,ImageFolder
from torchvision.models import resnet18, resnet34
from torchvision import transforms
import os
from model import SimCLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler

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


class CIFAR10Pair(CIFAR10):
    """Generate mini-batche pairs on CIFAR10 training set."""
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # .convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target  # stack a positive pair

from torchvision.transforms.functional import to_pil_image
class TinyImageNetPair(Dataset):
    """Generate mini-batch pairs on Tiny ImageNet training set."""
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root, transform=None)  # Disable transform here
        self.transform = transform
        
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        
        # Convert to PIL Image if it's a tensor
        if isinstance(img, torch.Tensor):
            img = to_pil_image(img)
        
        # Apply transformations
        imgs = [self.transform(img), self.transform(img)]
        
        return torch.stack(imgs), target  # stack a positive pair
    
    def __len__(self):
        return len(self.dataset)

def nt_xent(x, t=0.5):
    x = F.normalize(x, dim=1)
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
    x_scale = x_scores / t   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  # target of 2k element is 2k+1
    targets[1::2] -= 1  # target of 2k+1 element is 2k
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# color distortion composed by color jittering and color dropping.
# See Section A of SimCLR: https://arxiv.org/abs/2002.05709
def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


@hydra.main(config_path='./simclr.yml')
def train(args: DictConfig) -> None:
    assert torch.cuda.is_available()
    cudnn.benchmark = True

    train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          get_color_distortion(s=0.5),
                                          transforms.ToTensor()])
    data_dir = hydra.utils.to_absolute_path(args.data_dir)  # get absolute path of data dir
    ##cifar10预训练
    train_set = CIFAR10Pair(root=data_dir,
                            train=True,
                            transform=train_transform,
                            download=True)
    
    n_samples = len(train_set)
    subset_indices = np.random.choice(n_samples, size=int(n_samples * 0.5), replace=False)
    subset_sampler = SubsetRandomSampler(subset_indices)

    train_loader = DataLoader(train_set,
                            batch_size=args.batch_size,
                            sampler=subset_sampler,
                            num_workers=args.workers,
                            drop_last=True)

    # train_loader = DataLoader(train_set,
    #                           batch_size=args.batch_size,
    #                           shuffle=True,
    #                           num_workers=args.workers,
    #                           drop_last=True)
    
    ##tiny-imagenet预训练
    # train_set = TinyImageNetPair(root=os.path.join(data_dir, 'train'), transform=train_transform)

    # train_loader = DataLoader(train_set,
    #                           batch_size=args.batch_size,
    #                           shuffle=True,
    #                           num_workers=args.workers,
    #                           drop_last=True)
    
    # Prepare model
    assert args.backbone in ['resnet18', 'resnet34']
    base_encoder = eval(args.backbone)
    model = SimCLR(base_encoder).cuda()

    #     # Load pretrained parameters
    # pretrained_path = 'simclr_resnet18_epoch100.pt'  # Update with your path
    # if os.path.exists(pretrained_path):
    #     model.load_state_dict(torch.load(pretrained_path))
    #     logger.info(f"Loaded pretrained model from {pretrained_path}")
    # else:
    #     logger.warning(f"Pretrained model checkpoint '{pretrained_path}' not found.")

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)

    # cosine annealing lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            args.learning_rate,  # lr_lambda computes multiplicative factor
            1e-3))

    # TensorBoard writer path and naming
    log_dir = os.path.join('runs', 'train_sup')
    os.makedirs(log_dir, exist_ok=True)
    log_name = "selfsupervised_cifar10/change1_lr_0.6_bs_512"
    # log_name = "selfsupervised_cifar10_lr_0.6_bs_512"
    writer = SummaryWriter(log_dir=os.path.join(log_dir, log_name))

    # SimCLR training
    model.train()
    for epoch in range(1, args.epochs + 1):
        loss_meter = AverageMeter("SimCLR_loss")
        train_bar = tqdm(train_loader)
        for x, y in train_bar:
            sizes = x.size()
            x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4]).cuda(non_blocking=True)

            optimizer.zero_grad()
            feature, rep = model(x)
            loss = nt_xent(rep, args.temperature)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), x.size(0))
            train_bar.set_description("Train epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))
        
        writer.add_scalar('pretrain_imagenet/loss', loss_meter.avg, epoch + 1)

        # save checkpoint very log_interval epochs
        if epoch >= args.log_interval and epoch % args.log_interval == 0:
            logger.info("==> Save checkpoint. Train epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))
            torch.save(model.state_dict(), 'simclr_{}_epoch{}.pt'.format(args.backbone, epoch))


if __name__ == '__main__':
    train()


