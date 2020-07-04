import argparse
import os
from tqdm import tqdm
import torch

import numpy as np
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import cluster
import train_func as tf
import utils

def calc_acc(test_features, test_labels):
    _, test_pred = torch.max(test_features, 1)
    # test_pred = test_pred.values.detach()
    acc = utils.compute_accuracy(test_pred.numpy(), test_labels.numpy())
    print("Test Acc: {}".format(acc))
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation for Sequential Learning with CE')
    parser.add_argument('--model_dir', type=str, help='base directory for saving PyTorch model.')
    parser.add_argument('--epoch', type=int, default=None, help='which epoch for evaluation')
    parser.add_argument('--label_batch', type=int, default=None, help='which label batch for evaluation')
    parser.add_argument('--cpb', type=int, default=10, help='number of classes in each learning batch (default: 10)')
    
    parser.add_argument('--save', action='store_true', help='save labels')
    parser.add_argument('--data_dir', default='./data/', help='path to dataset')
    args = parser.parse_args()

    print("evaluate using label_batch: {}".format(args.label_batch))

    params = utils.load_params(args.model_dir)
    # get train features and labels
    train_transforms = tf.load_transforms('test')
    trainset = tf.load_trainset(params['data'], train_transforms, train=True, path=args.data_dir)
    if 'lcr' in params.keys(): # supervised corruption case
        trainset = tf.corrupt_labels(trainset, params['lcr'], params['lcs'])
    new_labels = trainset.targets
    assert (trainset.num_classes % args.cpb == 0),"Number of classes not divisible by cpb"
    ## load model
    net, epoch = tf.load_checkpoint_ce(args.model_dir, trainset.num_classes, args.epoch, eval_=True, label_batch_id=args.label_batch)
    net = net.cuda().eval()
    
    classes = np.unique(trainset.targets)
    class_batch_num = trainset.num_classes//args.cpb
    class_batch_list = classes.reshape(class_batch_num,args.cpb)

    # get test features and labels
    test_transforms = tf.load_transforms('test')
    testset = tf.load_trainset(params['data'], test_transforms, train=False)
    subtestset = tf.get_subset(class_batch_list[0,:],testset)
    testloader = DataLoader(subtestset, batch_size=200)
    test_features, test_labels = tf.get_features(net, testloader)

    calc_acc(test_features, test_labels)


