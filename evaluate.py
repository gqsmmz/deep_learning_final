import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tqdm
from torchvision import models
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from data import get_cifar100_train_loader, get_cifar100_test_loader
import numpy as np 
from torch.optim.lr_scheduler import LambdaLR


def supervised_linear_classifier(base_model, train_loader, test_loader, in_classes, num_epochs=20, lr=1e-3, device='cuda',writer=None):
    
    linear_model=base_model
    # linear_model = LinearClassifier(base_model, in_classes=in_classes).to(device)  #in_class就是resnet最后输出的大小，在此模型上再新增一个线性层。
    #因为simclr的fc层是恒等层，所以输出就是512，不用传入in_classes。但是监督学习的fc层输出是in_classes，所以要传入。

    if writer:
        writer = writer
    else:
        writer = SummaryWriter()  # TensorBoard writer



    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(linear_model.fc.parameters(), lr=0.01, momentum=0.9)  #original
    # optimizer = optim.Adam(linear_model.fc.parameters(), lr=lr)  #只训练新增的线性层
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 每10个epoch减少学习率到原来的0.1倍
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  #original

    

    def train(linear_model,epoch,optimizer,index=1):
        linear_model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            optimizer.zero_grad()
            outputs = linear_model(torch.squeeze(inputs, 1))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
            print(batch_idx+1,'/', len(train_loader),'epoch: %d' % epoch, '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if index==1:#第一次微调
            writer.add_scalar('finetuned_1_pretrain_imagenet/loss', train_loss/(batch_idx+1), epoch + 1)
            writer.add_scalar('finetuned_1_pretrain_imagenet/accuracy', 100.*correct/total, epoch + 1)
        else:
            writer.add_scalar('finetuned_2_pretrain_imagenet/loss', train_loss/(batch_idx+1), epoch + 1)
            writer.add_scalar('finetuned_2_pretrain_imagenet/accuracy', 100.*correct/total, epoch + 1)
    
    for epoch in range(num_epochs):
        train(linear_model,epoch,optimizer)

    torch.save(linear_model.state_dict(),'imagenet_10.pkl') 
    from torchvision import models
    net = models.resnet18(pretrained=False, num_classes=200).to(device)
    
    fc_inputs = net.fc.in_features # 保持与前面第一步中的代码一致
    net.fc = nn.Sequential(         #
        nn.Linear(fc_inputs, 100),  #
        nn.LogSoftmax(dim=1)
    )
 
    net.load_state_dict(torch.load('imagenet_10.pkl')) 
    # net=linear_model
    models=net.modules()
    for p in models:
        if p._get_name()!='Linear':
            print(p._get_name())
            p.requires_grad_=False
    
    net = net.to('cuda')
            
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #减小 lr,这里把所有参数拿进去训了
    for epoch in range(30):
        train(net,epoch,optimizer,index=2)
    writer.close()


    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(torch.squeeze(inputs, 1))
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(batch_idx,'/',len(test_loader),'test: ', 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        

