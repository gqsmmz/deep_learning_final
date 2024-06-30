import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.functional as F

# SimCLR Model and Projection Head
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),  #（512，512）
            nn.ReLU(inplace=True),
            nn.Linear(2048, out_dim) #（512，128）
        )

    def forward(self, x):
        return self.net(x)
    
class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.enc = base_encoder(pretrained=False)  # load model from torchvision.models without pretrained weights.
        self.feature_dim = self.enc.fc.in_features

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.enc.maxpool = nn.Identity()
        self.enc.fc = nn.Identity()  # remove final fully connected layer.

        # Add MLP projection.
        self.projection_dim = projection_dim
        self.projector = nn.Sequential(nn.Linear(self.feature_dim, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, projection_dim))

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection


# 监督学习训练函数
def train_supervised(model, train_loader, epochs=100, learning_rate=1e-3, device="cpu", 
                     dataset="cifar10",pretrained="true",data_scale="1.0",batch_size=256,writer=None):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss()

    if writer:
        writer = writer
    else:
        writer = SummaryWriter()  # TensorBoard writer
    
    # 创建保存检查点的文件夹
    checkpoint_name=f'supervised_{dataset}_pretrained_{pretrained}_data_scale_{data_scale}_batch_size_{batch_size}_epochs_{epochs}_lr_{learning_rate}'
    checkpoint_path = os.path.join('checkpoints', 'supervised', checkpoint_name)
    if checkpoint_path:
        os.makedirs(checkpoint_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()  # 每个epoch后更新学习率

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        writer.add_scalar('Loss/supervised_train', avg_loss, epoch + 1)
        writer.add_scalar('Accuracy/supervised_train', accuracy, epoch + 1)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

                # 每10个epoch保存一次模型
        if checkpoint_path and (epoch + 1) % 10 == 0:
            checkpoint_file = os.path.join(checkpoint_path, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_file)
            print(f'Checkpoint saved: {checkpoint_file}')

    writer.close()




def load_resnet(model_path, device, num_classes):
    model = resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Adjusting the last fully connected layer
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model


