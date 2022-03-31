import os
import sys
import json
import jittor as jt
from jittor import nn
from jittor import transform
from jittor.dataset import Dataset
from jittor.lr_scheduler import CosineAnnealingLR, MultiStepLR #,CosineAnnealingWarmRestarts

import sys
sys.path.append('../../Jittor-Image-Models') #for importing jimm
from jimm.loss import CrossEntropy, LabelSmoothingCrossEntropy
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import seaborn as sn
import matplotlib.pyplot as plt
from mlp_mixer import MLPMixer_S_16
from vision_transformer import vit_small_patch16_224
expID = 'vit_lbl_cos_7'
data_path = './outputs/'+str(expID)
pic_path='./outputs/'+str(expID)+'/confusion_matrix'
model_path='./outputs/'+str(expID)+'/model'
print("ExpID:",expID)
print("Datapath:",data_path)
print("PicPath:",pic_path)
if not os.path.isdir(data_path):
    os.mkdir(data_path)
if not os.path.isdir(pic_path):
    os.mkdir(pic_path)

expID = 'lbl_smth'
data_path = './outputs/'+str(expID)
if not os.path.isdir(data_path):
    os.mkdir(data_path)

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
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, padding=patch_size//2),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=(int)((kernel_size - 1) // 2)),
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
    model = ConvMixer(dim = 768, depth = 32, 
                    kernel_size=7, patch_size=7,
                    n_classes=num_classes,
                    **kwargs)
    return model

# ============== ./models/MoCo(MAE,MaskFeat,....).py ================== # 

# ========== ./tools/pretrain.py ================= # 

# ========== ./tools/train.py ================= # 

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, accum_iter, scheduler,training_datas):
    model.train()
    total_acc = 0
    total_num = 0
    losses = []
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    for i, (images, labels) in enumerate(pbar):
        # print(images.shape)
        output = model(images)
        loss = criterion(output, labels)

        optimizer.backward(loss)
        if (i + 1) % accum_iter == 0 or i + 1 == len(train_loader):
            optimizer.step(loss)

        # print(output)
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]
        losses.append(loss.data[0])

        pbar.set_description(f'Epoch {epoch} loss={sum(losses) / len(losses):.2f} '
                             f'acc={total_acc / total_num:.2f}')
    scheduler.step()
    training_datas.append({"Epoch":epoch,"loss":sum(losses) / len(losses),"acc":total_acc / total_num})

def valid_one_epoch(model, val_loader, epoch,valid_datas):
    model.eval()
    total_acc = 0
    total_num = 0

    #confusion matrix
    cm=np.zeros(shape=(102,103))

    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [VALID]')
    for i, (images, labels) in enumerate(pbar):
        output = model(images)
        pred = np.argmax(output.data, axis=1)
        for j in range(0,labels.data.shape[0]):
            # print("label:{} pred:{}".format(labels.data[j],pred[j]))
            if(pred[j]>101):
                cm[labels.data[j]][102]+=1
            else:
                cm[labels.data[j]][pred[j]]+=1


        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]

        pbar.set_description(f'Epoch {epoch} acc={total_acc / total_num:.2f}')

    f, ax = plt.subplots(figsize=(16, 9))
    ax = sn.heatmap(cm, annot=False, fmt='.20g')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.savefig(pic_path+'/cm_{expID}_{epoch}.jpg'.format(expID=expID,epoch=epoch))
    # print("epoch {epoch} completed.".format(epoch=epoch))

    acc = total_acc / total_num
    valid_datas.append({"Epoch":epoch,"acc":acc})
    plt.close(f)

    return acc

# ========== ./datasets/dataloader.py =============== # 

def output_data_to_file(training_datas,valid_datas):
    f1 = open(data_path+'/train_each_step.txt', 'a') #2 ways to write, to ensure that data don't lose
    f1.write(json.dumps(training_datas[len(training_datas)-1]))
    f1.write('\n')
    f1.close()

    f2 = open(data_path+'/valid_each_step.txt', 'a') 
    f2.write(json.dumps(valid_datas[len(valid_datas)-1]))
    f2.write('\n')
    f2.close()
data_transforms = {
    'train': transform.Compose([
        transform.Resize((256,256)),
        transform.RandomCrop((224, 224)),       # 从中心开始裁剪
        transform.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
        transform.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # 均值，标准差
    ]),
    'valid': transform.Compose([
        transform.Resize((224,224)),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
    ]),
    'test': transform.Compose([
        transform.Resize((224,224)),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
    ])
}

batch_size = 16
data_dir = './data'
image_datasets = {x: jt.dataset.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                ['train', 'valid', 'test']}
traindataset = image_datasets['train'].set_attrs(batch_size=batch_size, shuffle=True)
validdataset = image_datasets['valid'].set_attrs(batch_size=64, shuffle=False)
testdataset = image_datasets['test'].set_attrs(batch_size=1, shuffle=False)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
train_num = len(traindataset)
val_num = len(validdataset)
test_num = len(testdataset)

# ========== ./run.py =============== # 

jt.set_global_seed(537)
# model = ConvMixer_768_32(num_classes=102)
model = vit_small_patch16_224()
# model = MLPMixer_S_16(num_classes=102)
# criterion = nn.CrossEntropyLoss()
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
# optimizer = nn.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
# scheduler = MultiStepLR(optimizer, milestones=[40, 80, 160, 240], gamma=0.2) #learning rate decay

jt.set_global_seed(648)
model = ConvMixer_768_32()
# criterion = nn.CrossEntropyLoss()
criterion = LabelSmoothingCrossEntropy(smoothing=0.1) #How to Choose Alpha? 
optimizer = nn.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = MultiStepLR(optimizer, milestones=[40, 80, 160, 240], gamma=0.2) #learning rate decay
# scheduler = CosineAnnealingLR(optimizer, 15, 1e-5)
# scheduler = CosineAnnealingWarmRestarts(optimizer, 5, T_mult=2, eta_min=1e-6, last_epoch=- 1, verbose=False)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=- 1, verbose=False)

# slow decay
optimizer = nn.Adam(model.parameters(), lr=0.0005, weight_decay=5e-5)
# scheduler = CosineAnnealingLR(optimizer, 20, 1e-7)
# scheduler = CosineAnnealingLR(optimizer, 50, 0)
scheduler = CosineAnnealingLR(optimizer, 20, 5e-6)
epochs = 300
best_acc = 0.0
best_epoch = 0

training_datas = []
valid_datas = []
for epoch in range(epochs):
    train_one_epoch(model, traindataset, criterion, optimizer, epoch, 1, scheduler,training_datas)
    acc = valid_one_epoch(model, validdataset, epoch,valid_datas)
    output_data_to_file(training_datas,valid_datas)

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        model.save(f'{model_path}-{epoch}-{acc:.2f}.pkl')


f1 = open(data_path+'/train_total.txt', 'a')
f2 = open(data_path+'/valid_total.txt', 'a')
f1.write(json.dumps(training_datas))
f2.write(json.dumps(valid_datas))
f1.close()
f2.close()
f = open(data_path+'/best.txt', 'a')
f.write(best_acc)
f.write(best_epoch)
f.close()
print(best_acc, best_epoch)

