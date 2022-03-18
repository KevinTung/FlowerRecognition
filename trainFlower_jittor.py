import os
import jittor as jt
from jittor import nn
from jittor import transform
from jittor.dataset import Dataset
from jittor.lr_scheduler import CosineAnnealingLR, MultiStepLR

import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

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

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, accum_iter, scheduler):
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

def valid_one_epoch(model, val_loader, epoch):
    model.eval()
    total_acc = 0
    total_num = 0

    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [VALID]')
    for i, (images, labels) in enumerate(pbar):
        output = model(images)
        pred = np.argmax(output.data, axis=1)

        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]

        pbar.set_description(f'Epoch {epoch} acc={total_acc / total_num:.2f}')

    acc = total_acc / total_num
    return acc

# ========== ./datasets/dataloader.py =============== # 

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

jt.set_global_seed(648)
model = ConvMixer_768_32()
criterion = nn.CrossEntropyLoss()
optimizer = nn.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = MultiStepLR(optimizer, milestones=[40, 80, 160, 240], gamma=0.2) #learning rate decay
# scheduler = CosineAnnealingLR(optimizer, 15, 1e-5)


epochs = 300
best_acc = 0.0
best_epoch = 0
for epoch in range(epochs):
    train_one_epoch(model, traindataset, criterion, optimizer, epoch, 1, scheduler)
    acc = valid_one_epoch(model, validdataset, epoch)
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        # model.save(f'ConvMixer-{epoch}-{acc:.2f}.pkl')

print(best_acc, best_epoch)