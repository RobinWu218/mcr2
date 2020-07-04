import argparse
import os

import numpy as np
from torch.utils.data import DataLoader
from augmentloader import AugmentLoader
from torch.optim import SGD
import torch.nn as nn

import train_func as tf
from loss import MaximalCodingRateReduction
import utils



parser = argparse.ArgumentParser(description='Sequential Learning with Cross Entropy')
parser.add_argument('--arch', type=str, default='resnet18',
                    help='architecture for deep neural network (default: resnet18)')
parser.add_argument('--data', type=str, default='cifar10',
                    help='dataset for training (default: CIFAR10)')
parser.add_argument('--cpb', type=int, default=10,
                    help='number of classes in each learning batch (default: 10)')
parser.add_argument('--epo', type=int, default=500,
                    help='number of epochs for training (default: 500)')
parser.add_argument('--bs', type=int, default=1000,
                    help='input batch size for training (default: 1000)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--mom', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--lcr', type=float, default=0.,
                    help='label corruption ratio (default: 0)')
parser.add_argument('--lcs', type=int, default=10,
                    help='label corruption seed for index randomization (default: 10)')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--transform', type=str, default='default',
                    help='transform applied to trainset (default: default')
parser.add_argument('--save_dir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='base directory for saving PyTorch model. (default: ./data/)')
parser.add_argument('--pretrain_dir', type=str, default=None,
                    help='load pretrained checkpoint for assigning labels')
parser.add_argument('--pretrain_epo', type=int, default=None,
                    help='load pretrained epoch for assigning labels')
args = parser.parse_args()


## Pipelines Setup
model_dir = os.path.join(args.save_dir,
               'seqsupce_{}+{}_cpb{}_epo{}_bs{}_lr{}_mom{}_wd{}_lcr{}{}'.format(
                    args.arch, args.data, args.cpb, args.epo, args.bs, args.lr, args.mom, 
                    args.wd, args.lcr, args.tail))
headers = ["label_batch_id", "epoch", "step", "loss"]
utils.init_pipeline(model_dir,headers)
utils.save_params(model_dir, vars(args))

## per model functions
def lr_schedule(epoch, optimizer):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 400:
        lr = args.lr * 0.01
    elif epoch >= 200:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


## Prepare for Training
transforms = tf.load_transforms(args.transform)
trainset = tf.load_trainset(args.data, transforms, path=args.data_dir)
#trainset = tf.corrupt_labels(trainset, args.lcr, args.lcs)
if args.pretrain_dir is not None:
    net, _ = tf.load_checkpoint(args.pretrain_dir, args.pretrain_epo)
    utils.update_params(model_dir, args.pretrain_dir)
else:
    net = tf.load_architectures_ce(args.arch, trainset.num_classes)
assert (trainset.num_classes % args.cpb == 0),"Number of classes not divisible by cpb"
classes = np.unique(trainset.targets)
class_batch_num = trainset.num_classes//args.cpb
class_batch_list = classes.reshape(class_batch_num,args.cpb)

#trainloader = DataLoader(trainset, batch_size=args.bs, drop_last=True, num_workers=4)
criterion = nn.CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)


## Training
for label_batch_id in range(class_batch_num):
    subtrainset = tf.get_subset(class_batch_list[label_batch_id,:],trainset)
    trainloader = DataLoader(subtrainset, batch_size=args.bs, drop_last=True, num_workers=4)
    print("training starts on label batch:{}".format(label_batch_id))
    os.makedirs(os.path.join(model_dir, 'checkpoints','labelbatch{}'.format(label_batch_id)))
    for epoch in range(args.epo):
        lr_schedule(epoch, optimizer)
        for step, (batch_imgs, batch_lbls) in enumerate(trainloader):
            features = net(batch_imgs.cuda())
            loss = criterion(features, batch_lbls.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            utils.save_state(model_dir, label_batch_id, epoch, step, loss.item())
        utils.save_ckpt(model_dir, net, epoch,label_batch_id)
print("training complete.")
