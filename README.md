# deep_learning_final

# 1 数据集下载

运行`python data.py`便可下载tiny-imagenet数据集，其他数据集在进行如下训练和测试的时候，通过调用函数即可自动下载。

# 2 算法思路：

本项目采取的思路是通过自监督或者监督学习的方式预训练resnet18网络，再使用线性分类协议（Linear Classification Protocol）对每个预训练结果进行性能评估。其中该协议是去掉预训练后resnet18的投影层，取其特征层，然后新增两层线性层，并在用于微调的数据集上仅对新增的线性层进行微调训练，最后用微调数据集的测试集计算测试损失和准确率。

实验结果性能主要查看：预训练期间的训练损失曲线；线性分类协议评估时的微调训练损失曲线、微调训练准确率曲线，测试损失曲线、测试准确率曲线。

# 3 实验结果

### 3.1 三个方法的比较

#### 自监督（resnet18，在cifar10上训练，在cifar100上评估）：

pretrained=False，训练时的batch_size 为 512，初始学习率 learning_rate 为 0.6。使用了随机梯度下降 (SGD)作为优化器，使用了 LambdaLR 调度器根据给定的 lambda 函数动态调整学习率，且 lambda 函数使用余弦退火学习率调整方案，并设置为从初始学习率降到最低学习率 1e-3。微调时的batch_size 为 32，初始学习率 learning_rate 为 0.075，参数更新和学习率下降策略和预训练时一样。测试时的batch size也为32.

```
python train.py 

python test.py --train_type supervised --dataset cifar100 --test_num_epochs 100 --train_dataset cifar10 --model_path ***.pth
```
最终测试损失为 1.5800, 测试准确率为 68.96%。网络参数文件下载链接：https://drive.google.com/file/d/1NDzFZsoRcj6Sg4uxfybiiII0INHYGLRH/view?usp=drive_link

#### 监督（resnet18，在imagenet上训练，在cifar100上评估）：

pretrained=True，预训练时batch size为128，初始学习率 learning_rate 为 0.001。采用周期为50的余弦退火学习率调整策略。微调时epoch 为 30，batch_size 为 128，分为两个阶段。第一阶段采用 0.01 的初始学习率learning_rate、周期为 50 的余弦退火学习率调整策略、SGD 参数更新方法。第二阶段采用 0.001的初始学习率 learning_rate、周期为 50 的余弦退火学习率调整策略、SGD 参数更新方法。

```
python train.py --train_type supervised --dataset imagenet --epochs 0 --pretrained true

python test.py --train_type supervised --dataset cifar100 --test_num_epochs 5 --train_dataset imagenet --model_path ***.pth
```

最终测试损失 0.711，测试准确率为 80.260%。网络参数文件下载链接：https://drive.google.com/file/d/1ex41aaMlJfZ5XiCNQf9KSsgI_4F1Gpy0/view?usp=drive_link

#### 从零开始训练的监督（resneet18，pretrained=False，在cifar100上训练，在cifar100上测验）

pretrained=False，预训练时epoch为50，batch size为128，初始学习率 learning_rate 为 0.001。采用周期为50的余弦退火学习率调整策略。微调时epoch 为 5，batch_size 为 128，分为两个阶段。第一阶段采用 0.01 的初始学习率learning_rate、周期为 50 的余弦退火学习率调整策略、SGD 参数更新方法。第二阶段采用 0.001的初始学习率 learning_rate、周期为 50 的余弦退火学习率调整策略、SGD 参数更新方法。

```
python train.py --train_type supervised --dataset cifar100 --epochs 100 --pretrained false

python test.py --train_type supervised --dataset cifar100 --test_num_epochs 100 --train_dataset cifar100 --model_path ***.pth
```

在 cifar100 的测试集上的 Loss 为 1.201，准确率为 68.480%。网络参数文件下载链接：https://drive.google.com/file/d/1depRuN4D4NSICYGN7XVopH32wk6n2X7Q/view?usp=drive_link

### 3.2 探索不同超参数组合的自监督训练：

在不同规模的CIFAR-10数据集以及其他超参数组合上进行自监督预训练，并评测其在CIFAR-100数据集上的性能。用data_scale表示选取cifar10数据集的规模（1.0就是整个数据集），还可以修改batch size、epochs、初始learning_rate，用网格搜索方法探索最佳的参数组合方式。

手动更改simclr_train.py里关于cifar10数据集的规模参数以及其他超参数，本任务中主要做了如下超参数探索：使用50%的预训练数据集cifar10，batch size=256，初始learning_rate=0.3，选择预训练epoch为250的参数文件。

网络参数文件下载链接：https://drive.google.com/file/d/1VT9VcjWZuk0iyPsw0895VMIhfZ_2vTH2/view?usp=drive_link






