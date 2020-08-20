import argparse
import os
from tqdm import tqdm


import numpy as np
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import cluster
import train_func as tf
import utils


def svm(args, train_features, train_labels, test_features, test_labels):
    svm = LinearSVC(verbose=0, random_state=10)
    svm.fit(train_features, train_labels)
    acc_train = svm.score(train_features, train_labels)
    acc_test = svm.score(test_features, test_labels)
    print("SVM: {}".format(acc_test))
    return acc_train, acc_test


def knn(args, train_features, train_labels, test_features, test_labels):
    """Perform k-Nearest Neighbor classification using cosine similaristy as metric.

    Options:
        k (int): top k features for kNN
    
    """
    sim_mat = train_features @ test_features.T
    topk = sim_mat.topk(k=args.k, dim=0)
    topk_pred = train_labels[topk.indices]
    test_pred = topk_pred.mode(0).values.detach()
    acc = utils.compute_accuracy(test_pred.numpy(), test_labels.numpy())
    print("kNN: {}".format(acc))
    return acc


def nearsub(args, train_features, train_labels, test_features, test_labels, classes_num=None):
    """Perform nearest subspace classification.
    
    Options:
        n_comp (int): number of components for PCA or SVD
    
    """
    scores_pca = []
    scores_svd = []
    num_classes = train_labels.numpy().max() + 1 # should be correct most of the time
    if classes_num is not None:
        num_classes = classes_num
    features_sort, _ = utils.sort_dataset(train_features.numpy(), train_labels.numpy(), 
                                          num_classes=num_classes, stack=False)
    fd = features_sort[0].shape[1]
    for j in range(num_classes):
        pca = PCA(n_components=args.n_comp).fit(features_sort[j]) 
        pca_subspace = pca.components_.T
        mean = np.mean(features_sort[j], axis=0)
        pca_j = (np.eye(fd) - pca_subspace @ pca_subspace.T) \
                        @ (test_features.numpy() - mean).T
        score_pca_j = np.linalg.norm(pca_j, ord=2, axis=0)

        svd = TruncatedSVD(n_components=args.n_comp).fit(features_sort[j])
        svd_subspace = svd.components_.T
        svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) \
                        @ (test_features.numpy()).T
        score_svd_j = np.linalg.norm(svd_j, ord=2, axis=0)
        
        scores_pca.append(score_pca_j)
        scores_svd.append(score_svd_j)
    test_predict_pca = np.argmin(scores_pca, axis=0)
    test_predict_svd = np.argmin(scores_svd, axis=0)
    acc_pca = utils.compute_accuracy(test_predict_pca, test_labels.numpy())
    acc_svd = utils.compute_accuracy(test_predict_svd, test_labels.numpy())
    print('PCA: {}'.format(acc_pca))
    print('SVD: {}'.format(acc_svd))
    return acc_svd

def kmeans(args, train_features, train_labels, test_features, test_labels):
    """Perform KMeans clustering. 
    
    Options:
        n (int): number of clusters used in KMeans.

    """
    return cluster.kmeans(args, train_features, train_labels)

def ensc(args, train_features, train_labels, test_features, test_labels):
    """Perform Elastic Net Subspace Clustering.
    
    Options:
        gam (float): gamma parameter in EnSC
        tau (float): tau parameter in EnSC

    """
    return cluster.ensc(args, train_features, train_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--model_dir', type=str, help='base directory for saving PyTorch model.')
    parser.add_argument('--svm', help='evaluate using SVM', action='store_true')
    parser.add_argument('--knn', help='evaluate using kNN measuring cosine similarity', action='store_true')
    parser.add_argument('--nearsub', help='evaluate using Nearest Subspace', action='store_true')
    parser.add_argument('--kmeans', help='evaluate using KMeans', action='store_true')
    parser.add_argument('--ensc', help='evaluate using Elastic Net Subspace Clustering', action='store_true')
    parser.add_argument('--epoch', type=int, default=None, help='which epoch for evaluation')
    parser.add_argument('--label_batch', type=int, default=None, help='which label batch for evaluation')
    parser.add_argument('--cpb', type=int, default=10, help='number of classes in each learning batch (default: 10)')
    
    parser.add_argument('--k', type=int, default=5, help='top k components for kNN')
    parser.add_argument('--n', type=int, default=10, help='number of clusters for cluster (default: 10)')
    parser.add_argument('--gam', type=int, default=300, 
                        help='gamma paramter for subspace clustering (default: 100)')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='tau paramter for subspace clustering (default: 1.0)')
    parser.add_argument('--n_comp', type=int, default=30, help='number of components for PCA (default: 30)')
    parser.add_argument('--save', action='store_true', help='save labels')
    parser.add_argument('--data_dir', default='./data/', help='path to dataset')
    args = parser.parse_args()

    ## load model
    params = utils.load_params(args.model_dir)
    train_transforms = tf.load_transforms('test')
    trainset = tf.load_trainset(params['data'], train_transforms, train=True, path=args.data_dir)
    test_transforms = tf.load_transforms('test')
    testset = tf.load_trainset(params['data'], test_transforms, train=False)
    assert (trainset.num_classes % args.cpb == 0),"Number of classes not divisible by cpb"
    classes = np.unique(trainset.targets)
    class_batch_num = trainset.num_classes//args.cpb
    
    if 'lcr' in params.keys(): # supervised corruption case
        trainset = tf.corrupt_labels(trainset, params['lcr'], params['lcs'])
    new_labels = trainset.targets
    class_batch_list = classes.reshape(class_batch_num,args.cpb)

    for label_batch_id in range(class_batch_num):
        net, epoch = tf.load_checkpoint(args.model_dir, args.epoch, eval_=True, label_batch_id=label_batch_id)
        net = net.cuda().eval()
        print("Learning Session: {}".format(label_batch_id))
        # get train features and labels
        accs = []
        for task_id in range(label_batch_id+1):
            print("Currently evaluating on Task id: {}".format(task_id))
            subtrainset = tf.get_subset(class_batch_list[:task_id+1,:].flatten(),trainset)
            # subtrainset = tf.get_subset(class_batch_list[task_id,:],trainset)
            # print("subset train size: {}".format(len(subtrainset)))
            trainloader = DataLoader(subtrainset, batch_size=200)
            train_features, train_labels = tf.get_features(net, trainloader)
            # print("train feature size: {}".format(train_features.numpy().shape))
            # get test features and labels
            subtestset = tf.get_subset(class_batch_list[task_id,:],testset)
            # print("subset test size: {}".format(len(subtestset)))
            testloader = DataLoader(subtestset, batch_size=200)
            test_features, test_labels = tf.get_features(net, testloader)
            # print("test feature size: {}".format(test_features.numpy().shape))

            # test_labels = test_labels % args.cpb
            # train_labels = train_labels % args.cpb
            if args.svm:
                accs.append(svm(args, train_features, train_labels, test_features, test_labels)[0])
            if args.knn:
                accs.append(knn(args, train_features, train_labels, test_features, test_labels))
            if args.nearsub:
                accs.append(nearsub(args, train_features, train_labels, test_features, test_labels))
            if args.kmeans:
                accs.append(kmeans(args, train_features, train_labels))
            if args.ensc:
                accs.append(ensc(args, train_features, train_labels))
        
        print("Average Incremental Accuracy: {}".format(np.mean(np.array(accs))))

    
    # net, epoch = tf.load_checkpoint(args.model_dir, args.epoch, eval_=True, label_batch_id=0)
    # net = net.cuda().eval()
    # print("Learning Session: {}".format(0))
    # # get train features and labels
    # accs = []
    # for task_id in range(class_batch_num):
    #     print("Currently evaluating on Task id: {}".format(task_id))
    #     subtrainset = tf.get_subset(class_batch_list[:task_id+1,:].flatten(),trainset)
    #     # subtrainset = tf.get_subset(class_batch_list[task_id,:],trainset)
    #     # print("subset train size: {}".format(len(subtrainset)))
    #     trainloader = DataLoader(subtrainset, batch_size=200)
    #     train_features, train_labels = tf.get_features(net, trainloader)
    #     # print("train feature size: {}".format(train_features.numpy().shape))
    #     # get test features and labels
    #     subtestset = tf.get_subset(class_batch_list[task_id,:],testset)
    #     # print("subset test size: {}".format(len(subtestset)))
    #     testloader = DataLoader(subtestset, batch_size=200)
    #     test_features, test_labels = tf.get_features(net, testloader)
    #     # print("test feature size: {}".format(test_features.numpy().shape))

    #     # test_labels = test_labels % args.cpb
    #     # train_labels = train_labels % args.cpb
    #     if args.svm:
    #         accs.append(svm(args, train_features, train_labels, test_features, test_labels)[0])
    #     if args.knn:
    #         accs.append(knn(args, train_features, train_labels, test_features, test_labels))
    #     if args.nearsub:
    #         accs.append(nearsub(args, train_features, train_labels, test_features, test_labels))
    #     if args.kmeans:
    #         accs.append(kmeans(args, train_features, train_labels))
    #     if args.ensc:
    #         accs.append(ensc(args, train_features, train_labels))
    
    # print("Average Incremental Accuracy: {}".format(np.mean(np.array(accs))))
