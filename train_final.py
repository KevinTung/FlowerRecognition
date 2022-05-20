from requests import patch
from jittor.lr_scheduler import CosineAnnealingLR, MultiStepLR
from jittor.dataset import Dataset
from jittor import transform
from jittor import nn
import jittor as jt
import json
import os
from jimm.loss import CrossEntropy
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import torch
import seaborn as sn
import matplotlib.pyplot as plt
from mlp_mixer import MLPMixer_S_16
from vision_transformer import vit_small_patch_K_16_224
from xxlimited import new
from email.mime import image
from vision_transformer import trunc_normal_
sys.path.append('../../Jittor-Image-Models')  # for importing jimm

# ,CosineAnnealingWarmRestarts
batch_size = 16
x_set_name = "rawPic"
y_set_name = "edge_new"
expID = 'vit-k=8'
data_path = './outputs-final/'+str(expID)
pic_path = './outputs-final/'+str(expID)+'/confusion_matrix'
model_path = './outputs-final/'+str(expID)+'/model'
print("ExpID:", expID)
print("Datapath:", data_path)
print("PicPath:", pic_path)
if not os.path.isdir(data_path):
    os.mkdir(data_path)
if not os.path.isdir(pic_path):
    os.mkdir(pic_path)


# import matplotlib.pyplot as plt
jt.flags.use_cuda = 1

# ============== ./tools/util.py ================== #

# ============== ./models/model.py ================== #


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def execute(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size,
                      stride=patch_size, padding=patch_size//2),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim,
                              padding=(int)((kernel_size - 1) // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for i in range(depth)],
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.classifier = nn.Sequential(
            nn.Linear(dim, n_classes)
        )

    def execute(self, x):
        embedding = self.embedding(x)
        embedding = self.blocks(embedding)
        embedding = self.avgpool(embedding)     # feature vector!!!!
        embedding = jt.flatten(embedding, 1)
        out = self.classifier(embedding)
        return out


def ConvMixer_768_32(num_classes: int = 1000, **kwargs):
    model = ConvMixer(dim=768, depth=32,
                      kernel_size=7, patch_size=7,
                      n_classes=num_classes,
                      **kwargs)
    return model

# ============== ./models/MoCo(MAE,MaskFeat,....).py ================== #

# ========== ./tools/pretrain.py ================= #

# ========== ./tools/train.py ================= #


def train_one_epoch(model, x_ds,label_ds, criterion, optimizer, epoch, accum_iter, scheduler, training_datas):
    #modify train_loader to x_ds and label_ds; when setting pbar, we use x_ds 
    model.train()
    total_acc = 0
    total_num = 0
    losses = []
    pbar = tqdm(x_ds, desc=f'Epoch {epoch} [TRAIN]')
    for index, (value1, value2) in enumerate(zip(x_ds, label_ds)):
        output = model(value1[0])
        labels=model.patch_embed(value2[0])
        #Block1->2; transform y so that it matchs model's output
        B = labels.shape[0]
        embed_dim = 768
        cls_token = jt.zeros((1, 1, embed_dim))
        cls_token = trunc_normal_(cls_token, std=.02)
        _, i, j = cls_token.shape
        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = cls_token.expand((B, i, j))
        labels = jt.contrib.concat((cls_tokens, labels), dim=1)
        
        #output and labels finished 
        loss = criterion(output, labels)
        optimizer.backward(loss)
        if (i + 1) % accum_iter == 0 or i + 1 == len(x_ds):
            optimizer.step(loss)
        # pred = np.argmax(output.data, axis=1)
        # acc = np.sum(pred == labels.data)
        # total_acc += acc
        # total_num += labels.shape[0]
        losses.append(loss.data[0])
        # pbar.set_description(f'Epoch {epoch} loss={sum(losses) / len(losses):.2f} '
        #                      f'acc={total_acc / total_num:.2f}')
        pbar.set_description(f'Epoch {epoch} loss={sum(losses) / len(losses):.2f}')
        
        
    # for i, (images, labels) in enumerate(pbar):
    #     output = model(images)
    #     loss = criterion(output, labels)

    #     optimizer.backward(loss)
    #     if (i + 1) % accum_iter == 0 or i + 1 == len(train_loader):
    #         optimizer.step(loss)
    #     pred = np.argmax(output.data, axis=1)
    #     acc = np.sum(pred == labels.data)
    #     total_acc += acc
    #     total_num += labels.shape[0]
    #     losses.append(loss.data[0])

    #     pbar.set_description(f'Epoch {epoch} loss={sum(losses) / len(losses):.2f} '
    #                          f'acc={total_acc / total_num:.2f}')
    scheduler.step()
    # training_datas.append({"Epoch": epoch, "loss": sum(
    #     losses) / len(losses), "acc": total_acc / total_num})
    training_datas.append({"Epoch": epoch, "loss": sum(
        losses) / len(losses)})
    return sum(losses)/len(losses)


def valid_one_epoch(model, val_loader, epoch, valid_datas):
    model.eval()
    total_acc = 0
    total_num = 0

    # confusion matrix
    cm = np.zeros(shape=(102, 103))

    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [VALID]')
    for i, (images, labels) in enumerate(pbar):
        output = model(images)
        pred = np.argmax(output.data, axis=1)
        for j in range(0, labels.data.shape[0]):
            # print("label:{} pred:{}".format(labels.data[j],pred[j]))
            if(pred[j] > 101):
                cm[labels.data[j]][102] += 1
            else:
                cm[labels.data[j]][pred[j]] += 1

        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]

        pbar.set_description(f'Epoch {epoch} acc={total_acc / total_num:.2f}')

    f, ax = plt.subplots(figsize=(16, 9))
    ax = sn.heatmap(cm, annot=False, fmt='.20g')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.savefig(
        pic_path+'/cm_{expID}_{epoch}.jpg'.format(expID=expID, epoch=epoch))
    # print("epoch {epoch} completed.".format(epoch=epoch))

    acc = total_acc / total_num
    valid_datas.append({"Epoch": epoch, "acc": acc})
    plt.close(f)

    return acc

# ========== ./datasets/dataloader.py =============== #


def output_data_to_file(training_datas):
    # print("IN OUTPUT",data_path+'/train_each_step.txt',training_datas)
    # 2 ways to write, to ensure that data don't lose
    f1 = open(data_path+'/train_each_step.txt', 'a')
    f1.write(json.dumps(training_datas[len(training_datas)-1]))
    f1.write('\n')
    f1.close()

    # f2 = open(data_path+'/valid_each_step.txt', 'a')
    # f2.write(json.dumps(valid_datas[len(valid_datas)-1]))
    # f2.write('\n')
    # f2.close()


data_transforms = {
    'train': transform.Compose([
        transform.Resize((256, 256)),
        transform.RandomCrop((224, 224)),       # 从中心开始裁剪
        transform.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
        transform.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # 均值，标准差
    ]),
    'valid': transform.Compose([
        transform.Resize((224, 224)),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
    ]),
    'test': transform.Compose([
        transform.Resize((224, 224)),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
    ])
}


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        super().__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def execute(self, x: jt.Var, target: jt.Var) -> jt.Var:
        print("In execute", x.shape, target.shape)
        if target.ndim <= 1:
            target = target.broadcast(x, [1])
            target = (target.index(1) == target).unary(op='float32')
        # logprobs = x - x.exp().sum(1).log()
        print("In execute 2", x.shape, target.shape)
        logprobs = nn.log_softmax(x, dim=-1)
        nll_loss = -(logprobs * target).sum(1)
        smooth_loss = -logprobs.mean(dim=-1)
        print("In execute 3", x.shape, target.shape)
        a = self.smoothing * smooth_loss
        loss = self.confidence * nll_loss
        return loss.mean()



data_dir = '../dataAugmentation'
# image_datasets = {x: jt.dataset.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
#                   ['train', 'valid', 'test']}
# traindataset = image_datasets['train'].set_attrs(
#     batch_size=batch_size, shuffle=True)
# validdataset = image_datasets['valid'].set_attrs(batch_size=64, shuffle=False)
# testdataset = image_datasets['test'].set_attrs(batch_size=1, shuffle=False)

# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
# train_num = len(traindataset)
# val_num = len(validdataset)
# test_num = len(testdataset)


# ========== ./run.py =============== #

jt.set_global_seed(537)

# single attribute test
# optimized configuration (need to check)
model = vit_small_patch_K_16_224(k=8)
criterion = nn.MSELoss()  # How to Choose Alpha?
optimizer = nn.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
optimizer = nn.Adam(model.parameters(), lr=0.0005, weight_decay=5e-5)
scheduler = CosineAnnealingLR(optimizer, 20, 5e-6)

# freeze model
# k = 3
epochs = 300
# best_acc = 0.0
best_loss = 10000 
best_epoch = 0

training_datas = []
valid_datas = []

# print(model.blocks)

# goal: build dataset

# print(yourds)
# Single X dataset (720 pics)

# Y dataset(720 pics)
data_transforms = {
    'train': transform.Compose([
        transform.Resize((256, 256)),
        transform.RandomCrop((224, 224)),       # 从中心开始裁剪
        transform.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
        transform.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # 均值，标准差
    ]),
    'valid': transform.Compose([
        transform.Resize((224, 224)),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
    ]),
    'test': transform.Compose([
        transform.Resize((224, 224)),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
    ])
}

new_image_datasets = {x: jt.dataset.ImageFolder(os.path.join(data_dir, x),transform.Compose([
        transform.ToTensor()
    ])) for x in
                      [x_set_name, y_set_name]}
x_ds = new_image_datasets[x_set_name].set_attrs(
    batch_size=batch_size, shuffle=False)
label_ds = new_image_datasets[y_set_name].set_attrs(
    batch_size=batch_size, shuffle=False)

# for index, (value1, value2) in enumerate(zip(x_ds, label_ds)):
    
#     print("CHECK DATA...",index,value1[0].shape)
# for index, (value1, value2) in enumerate(zip(x_ds, label_ds)):
    
#     print('before shape',value1[0].shape, value2[0].shape) #need to eval()
#     x1=value1[0]#.transpose(0,3,1,2)
#     print("X1.shape",x1.shape)
#     x1 = model(x1)
#     y=value2[0]#.transpose(0,3,1,2)
#     y=model.patch_embed(y)
#     #Block1->2
#     B = y.shape[0]
#     embed_dim = 768
#     cls_token = jt.zeros((1, 1, embed_dim))
#     cls_token = trunc_normal_(cls_token, std=.02)
#     _, i, j = cls_token.shape
#     # stole cls_tokens impl from Phil Wang, thanks
#     cls_tokens = cls_token.expand((B, i, j))
#     y = jt.contrib.concat((cls_tokens, y), dim=1)
#     # print("after shape",x1.shape,y.shape)
    
#     # print("FUCK")
#     # print("Y",y)
#     print('loss:',criterion(x1,y))

    #print(criterion(value1[0], value2[0]))
# for i in range(len(x_ds)):
#     x_ds[i] = (x_ds[i][0],label_ds[i][0])
# print(x_ds[1000])

# for i, (images, labels) in enumerate(x_ds):
#     print(images.shape,labels.shape)


# method1: use pytorch;

# image_datasets =  np.stack([new_image_datasets['X_data'],new_image_datasets['edge']])
# print(image_datasets.shape)

# for (i,j) in traindataset:
#     traindataset
#     Dataset.collate_batch()


# train with k layer freezed
for epoch in range(epochs):
    loss = train_one_epoch(model, x_ds,label_ds, criterion, optimizer,
                    epoch, 1, scheduler, training_datas)
    # acc = valid_one_epoch(model, x_ds,label_ds, epoch, valid_datas)
    output_data_to_file(training_datas) #not output now

    # if acc > best_acc:
    #     best_acc = acc
    #     best_epoch = epoch
    #     model.save(f'{model_path}-{epoch}-{acc:.2f}.pkl')
    print("Loss:",loss,"Best Loss",best_loss)
    if loss < best_loss:
        best_loss = loss
        best_epoch = epoch
        model.save(f'{model_path}-{epoch}-{loss:.2f}.pkl')


# save model


# test task load model
# initialize partial model


# evaluation


f1 = open(data_path+'/train_total.txt', 'a')
# f2 = open(data_path+'/valid_total.txt', 'a')
f1.write(json.dumps(training_datas))
# f2.write(json.dumps(valid_datas))
f1.close()
# f2.close()
# f = open(data_path+'/best.txt', 'a')
# f.write(best_acc)
# f.write(best_epoch)
# f.close()
# print(best_acc, best_epoch)
