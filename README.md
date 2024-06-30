# deep_learning_final

# 数据集下载

运行`python data.py`便可下载tiny-imagenet数据集，其他数据集在进行如下训练和测试的时候，通过调用函数即可自动下载。

# 算法思路：

本项目采取的思路是通过自监督或者监督学习的方式预训练resnet18网络，再使用线性分类协议（Linear Classification Protocol）对每个预训练结果进行性能评估。其中该协议是去掉预训练后resnet18的投影层，取其特征层，然后新增两层线性层，并在用于微调的数据集上仅对新增的线性层进行微调训练，最后用微调数据集的测试集计算测试损失和准确率。

实验结果性能主要查看：预训练期间的训练损失曲线；线性分类协议评估时的微调训练损失曲线、微调训练准确率曲线，测试损失曲线、测试准确率曲线。

### 三个实验的比较

#### 自监督（resnet18，在cifar10上训练，在cifar100上评估）：

batch_size 为 512，初始学习率 learning_rate 为 0.6。使用了随机梯度下降 (SGD)作为优化器，使用了 LambdaLR 调度器根据给定的 lambda 函数动态调整学习率，且 lambda 函数使用余弦退火学习率调整方案，并设置为从初始学习率降到最低学习率 1e-

```
python train.py 

python test.py --train_type supervised --dataset cifar100 --test_num_epochs 100 --train_dataset cifar10 --model_path ***.pth
```

#### 监督（resnet18，在imagenet上训练，在cifar100上评估）：

```
python train.py --train_type supervised --dataset imagenet --epochs 0 --pretrained true

python test.py --train_type supervised --dataset cifar100 --test_num_epochs 5 --train_dataset imagenet --model_path ***.pth
```

#### 从零开始训练的监督（resneet18，pretrained=False，在cifar100上训练，在cifar100上测验）

python train.py --train_type supervised --dataset cifar100 --epochs 100 --pretrained false

python test.py --train_type supervised --dataset cifar100 --test_num_epochs 100 --train_dataset cifar100 --model_path ***.pth

### 探索不同超参数组合的自监督训练：

在不同规模的CIFAR-10数据集以及其他超参数组合上进行自监督预训练，并评测其在CIFAR-100数据集上的性能。用data_scale表示选取cifar10数据集的规模（1.0就是整个数据集），还可以修改batch_size、epochs、learning_rate，用网格搜索方法探索最佳的参数组合方式。

手动更改simclr_train.py里关于cifar10数据集的规模参数以及其他超参数，本任务中主要做了两个超参数组合探索：

1）




