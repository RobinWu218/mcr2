import argparse
import os
from tqdm import tqdm


import numpy as np
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from loss_cpu import MaximalCodingRateReduction
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
    """Perform k-Nearest Neighbor classification using cosine similarity as metric.
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


def nearsub(args, train_features, train_labels, test_features, test_labels):
    """Perform nearest subspace classification.
    
    Options:
        n_comp (int): number of components for PCA or SVD
    
    """
    scores_pca = []
    scores_svd = []
    num_classes = train_labels.numpy().max() + 1 # should be correct most of the time
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
    return acc_pca, acc_svd

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



# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--model_dir', type=str, help='base directory for saving PyTorch model.')
parser.add_argument('--svm', help='evaluate using SVM', action='store_true')
parser.add_argument('--knn', help='evaluate using kNN measuring cosine similarity', action='store_true')
parser.add_argument('--nearsub', help='evaluate using Nearest Subspace', action='store_true')
parser.add_argument('--kmeans', help='evaluate using KMeans', action='store_true')
parser.add_argument('--ensc', help='evaluate using Elastic Net Subspace Clustering', action='store_true')
parser.add_argument('--epoch', type=int, default=None, help='which epoch for evaluation')

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

print("Start Evaluation")

if not os.path.exists(args.data_dir):
    raise NameError("No dataset found in directory {}".format(args.data_dir))

dataset_list = os.listdir(args.data_dir)

dataset_list.remove('omniglot')
dataset_list.remove('vgg-flowers')
dataset_list.remove('protein_atlas')
# dataset_list = dataset_list[8:]
print(dataset_list)

criterion = MaximalCodingRateReduction(gam1=1., gam2=1., eps=0.5)
res = np.zeros((len(dataset_list),len(dataset_list)),dtype=dict)

for i, source_ds_name in enumerate(dataset_list):
    print("Use pretrained network on: {}".format(source_ds_name))
    pretrained_model_dir = args.model_dir.split("{}")[0]+source_ds_name+args.model_dir.split("{}")[1]
    ## load model
    params = utils.load_params(pretrained_model_dir)
    net, epoch = tf.load_checkpoint(pretrained_model_dir, args.epoch, eval_=True)
    net = net.cuda().eval()

    for j, target_ds_name in enumerate(dataset_list):
        stats_dict = {}
        # get train features and labels
        train_transforms = tf.load_transforms('transfer')#('test')
        trainset = tf.load_trainset(target_ds_name, train_transforms, train=True, path=args.data_dir)
        if 'lcr' in params.keys(): # supervised corruption case
            trainset = tf.corrupt_labels(trainset, params['lcr'], params['lcs'])
        new_labels = trainset.targets
        trainloader = DataLoader(trainset, batch_size=200)
        print("Target task on: {}".format(target_ds_name))
        train_features, train_labels = tf.get_features(net, trainloader, verbose=False)

        # get test features and labels
        test_transforms = tf.load_transforms('transfer')#('test')
        testset = tf.load_trainset(target_ds_name, test_transforms, train=False, path=args.data_dir)
        testloader = DataLoader(testset, batch_size=200)
        test_features, test_labels = tf.get_features(net, testloader, verbose=False)
        
        trainloader_for_mcr = DataLoader(trainset, batch_size=1000, drop_last=True, num_workers=4)
        for step, (batch_imgs, batch_lbls) in enumerate(trainloader_for_mcr):
            batch_features = net(batch_imgs.cuda()).cpu().detach()
            loss, loss_empi, loss_theo = criterion(batch_features, batch_lbls, num_classes=trainset.num_classes)
            stats_dict['mcr_loss'] = loss.item()
            print("mcr loss: {}".format(loss.item()))
            break
        if args.svm:
            acc_svm = svm(args, train_features, train_labels, test_features, test_labels)
            stats_dict['svm_train_acc'] = acc_svm[0]
            stats_dict['svm_test_acc'] = acc_svm[1]
        if args.knn:
            acc_knn = knn(args, train_features, train_labels, test_features, test_labels)
            stats_dict['knn_test_acc'] = acc_knn
        if args.nearsub:
            acc_nearsub = nearsub(args, train_features, train_labels, test_features, test_labels)
            stats_dict['nearsub_test_acc_pca'] = acc_nearsub[0]
            stats_dict['nearsub_test_acc_svd'] = acc_nearsub[1]
        if args.kmeans:
            res_kmeans = kmeans(args, train_features, train_labels)
            stats_dict['kmeans_acc'] = res_kmeans[0]
        if args.ensc:
            res_ensc = ensc(args, train_features, train_labels)
            stats_dict['ensc_acc'] = res_ensc[0]
        res[i,j] = stats_dict
print("evaluation completed")
np.save("transfer_results_1",res)

# source_ds_name = dataset_list[0]
# print("Use pretrained network on: {}".format(source_ds_name))
# pretrained_model_dir = args.model_dir.split("{}")[0]+source_ds_name+args.model_dir.split("{}")[1]
# ## load model
# params = utils.load_params(pretrained_model_dir)
# net, epoch = tf.load_checkpoint(pretrained_model_dir, args.epoch, eval_=True)
# net = net.cuda().eval()

# target_ds_name = dataset_list[1]
# stats_dict = {}
# # get train features and labels
# train_transforms = tf.load_transforms('transfer')#('test')
# trainset = tf.load_trainset(target_ds_name, train_transforms, train=True, path=args.data_dir)
# if 'lcr' in params.keys(): # supervised corruption case
#     trainset = tf.corrupt_labels(trainset, params['lcr'], params['lcs'])
# new_labels = trainset.targets
# trainloader = DataLoader(trainset, batch_size=200)
# print("Target task on: {}".format(target_ds_name))
# train_features, train_labels = tf.get_features(net, trainloader, verbose=False)

# # get test features and labels
# test_transforms = tf.load_transforms('transfer')#('test')
# testset = tf.load_trainset(target_ds_name, test_transforms, train=False, path=args.data_dir)
# testloader = DataLoader(testset, batch_size=200)
# test_features, test_labels = tf.get_features(net, testloader, verbose=False)

# loss, loss_empi, loss_theo = criterion(train_features, train_labels, num_classes=trainset.num_classes)
# stats_dict['mcr_loss'] = loss.item()
# if args.svm:
#     acc_svm = svm(args, train_features, train_labels, test_features, test_labels)
#     stats_dict['svm_train_acc'] = acc_svm[0]
#     stats_dict['svm_test_acc'] = acc_svm[1]
# if args.knn:
#     acc_knn = knn(args, train_features, train_labels, test_features, test_labels)
#     stats_dict['knn_test_acc'] = acc_knn
# if args.nearsub:
#     acc_nearsub = nearsub(args, train_features, train_labels, test_features, test_labels)
#     stats_dict['nearsub_test_acc_pca'] = acc_nearsub[0]
#     stats_dict['nearsub_test_acc_svd'] = acc_nearsub[1]
# if args.kmeans:
#     res_kmeans = kmeans(args, train_features, train_labels)
#     stats_dict['kmeans_acc'] = res_kmeans[0]
# if args.ensc:
#     res_ensc = ensc(args, train_features, train_labels)
#     stats_dict['ensc_acc'] = res_ensc[0]
# print(stats_dict)
            