import json
import matplotlib.pyplot as plt
import numpy as np
import os

expID = ['conv_cos_lbl_2','mlp_128_4_cos_lbl','vit_lb_cos_lr1e-4']

for item in expID:
    loss = []
    epoch = []
    train_acc = []
    val_acc = []

    data_path = './outputs/'+str(item)
    # out_path = './loss_pic'
    out_path='./data_sol/loss_pic'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    f = open(data_path+"/train_each_step.txt")
    while 1:
        line = f.readline()
        if not line:
            break
        it = json.loads(line)
        loss.append(it["loss"])
        epoch.append(it["Epoch"])
        train_acc.append(it["acc"])
    f.close()

    f = open(data_path+"/valid_each_step.txt")
    while 1:
        line = f.readline()
        if not line:
            break
        it = json.loads(line)
        val_acc.append(it["acc"])
    f.close()

    # loss
    fig, ax1 = plt.subplots(figsize=(16, 9))
    lns1 = ax1.plot(np.arange(len(epoch)), loss, label="Loss", color='khaki')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('training loss')

    # acc
    ax2 = ax1.twinx()
    lns2 = ax2.plot(np.arange(len(epoch)), train_acc,
                    label="train acc", color='blue')
    lns3 = ax2.plot(np.arange(len(epoch)), val_acc,
                    label="val acc", color='hotpink')

    ax2.set_xlabel('epoch')
    ax2.set_ylabel('acc')

    lns = lns2+lns3+lns1
    labels = ["train acc", "val acc", "loss"]
    plt.legend(lns, labels, loc='best')
    plt.title("{}".format(item))
    plt.savefig(out_path+'/loss_{}.png'.format(item))
