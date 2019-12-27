import torch
import torch.nn
import torch.nn.init
import torchvision, torchtext
import argparse
import sklearn.datasets
import numpy as np
import random

def weight_init(m):
    '''
    This function is used to initialize the netwok weights
    '''

    if isinstance(m,torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m,torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m,torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)

def get_mnist():
    ''' 
    This function returns the MNIST dataset in training, validation, test splits.
    '''

    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
        torchvision.transforms.Normalize((0.0,), (1.0,))]))
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
        torchvision.transforms.Normalize((0.0,), (1.0,))]))
    
    return trainset, testset

def get_fashion_mnist():
    ''' 
    This function returns the MNIST dataset in training, validation, test splits. Parameters:
    - percentage_validation (float) : Percentage of the original training set sent to validation 
    '''

    trainset = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
        torchvision.transforms.Normalize((0.0,), (1.0,))]))
    testset = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
        torchvision.transforms.Normalize((0.0,), (1.0,))]))
    
    return trainset, testset

class DatasetRegression(torch.utils.data.Dataset):

    def __init__(self,X,Y):
        self.data = X
        self.targets = Y
    
    def __getitem__(self,index):
        data, target = self.data[index], self.targets[index]
        return data, target
    
    def __len__(self):
        return len(self.targets)

def get_california_housing(percentage_test=0.3):

    X, Y = sklearn.datasets.fetch_california_housing(data_home='../data/CaliforniaHousing/', \
        download_if_missing=True,return_X_y=True)
    
    # We remove the houses with prices higher than 500,000 dollars
    idx_drop = Y >= 5
    X, Y = X[~idx_drop], np.log(Y[~idx_drop])

    # We shuffle the inputs and outputs before assigning train/test 
    tmp = list(zip(X,Y))
    random.shuffle(tmp)
    X, Y = zip(*tmp)
    X, Y = torch.FloatTensor(X), torch.FloatTensor(Y)
    X = (X - torch.mean(X,0)) / torch.std(X,0)

    # Split between training / testing
    splitpoint_test = int(len(Y) * (1.0 - percentage_test))
    X_train, Y_train = X[:splitpoint_test], Y[:splitpoint_test]
    X_test, Y_test = X[splitpoint_test:], Y[splitpoint_test:]

    # Generate and return the datasets
    trainset = DatasetRegression(X_train,Y_train)
    testset = DatasetRegression(X_test,Y_test)
    return trainset, testset

def get_TREC():

    TEXT = torchtext.data.Field(tokenize = 'spacy')
    LABEL = torchtext.data.LabelField()

    trainset, testset = torchtext.datasets.TREC.splits(TEXT, LABEL, fine_grained=False, root='../data/')

    MAX_VOCAB_SIZE = 25_000
    TEXT.build_vocab(trainset, 
                    max_size = MAX_VOCAB_SIZE, 
                    vectors = "glove.6B.100d", 
                    unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(trainset)

    return trainset, testset, TEXT, LABEL

def get_data(dataset='mnist'):
    '''
    This function returns the training and validation set from MNIST
    '''

    if dataset == 'mnist':
        return get_mnist()
    elif dataset == 'fashion_mnist':
        return get_fashion_mnist()
    elif dataset == 'california_housing':
        return get_california_housing()
    elif dataset == 'trec':
        return get_TREC()

def get_args():
    '''
    This function returns the arguments from terminal and set them to display
    '''

    parser = argparse.ArgumentParser(
        description = 'Run convex IB Lagrangian on MNIST dataset (with Pytorch)',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--logs_dir', default = '../results/logs/',
        help = 'folder to output the logs')
    parser.add_argument('--figs_dir', default = '../results/figures/',
        help = 'folder to output the images')
    parser.add_argument('--models_dir', default = '../results/models/',
        help = 'folder to save the models')
    parser.add_argument('--n_epochs', type = int, default = 100,
        help = 'number of training epochs')
    parser.add_argument('--beta', type = float, default = 0.0,
        help = 'Lagrange multiplier (only for train_model)')
    parser.add_argument('--n_betas', type = int, default = 50,
        help = 'Number of Lagrange multipliers (only for study behavior)')
    parser.add_argument('--beta_lim_min', type = float, default = 0.0,
        help = 'minimum value of beta for the study of the behavior')
    parser.add_argument('--beta_lim_max', type = float, default = 1.0,
        help = 'maximum value of beta for the study of the behavior')  
    parser.add_argument('--u_func_name', choices = ['pow', 'exp'], default = 'exp',
        help = 'monotonically increasing, strictly convex function')
    parser.add_argument('--hyperparameter', type = float, default = 1.0,
        help = 'hyper-parameter of the h function (e.g., alpha in the power and eta in the exponential case)')
    parser.add_argument('--example_clusters', action = 'store_true', default = False,
        help = 'plot example of the clusters obtained (only for study behavior of power with alpha 1, otherwise change the number of clusters to show)')
    parser.add_argument('--K', type = int, default = 2,
        help = 'Dimensionality of the bottleneck varaible')
    parser.add_argument('--logvar_kde', type = float, default = -1.0,
        help = 'initial log variance of the KDE estimator')
    parser.add_argument('--logvar_t', type = float, default = -1.0,
        help = 'initial log varaince of the bottleneck variable')
    parser.add_argument('--sgd_batch_size', type = int, default = 128,
        help = 'mini-batch size for the SGD on the error')
    parser.add_argument('--mi_batch_size', type = int, default = 1000,
        help = 'mini-batch size for the I(X;T) estimation')
    parser.add_argument('--same_batch', action = 'store_true', default = False,
        help = 'use the same mini-batch for the SGD on the error and I(X;T) estimation')
    parser.add_argument('--dataset', choices = ['mnist', 'fashion_mnist', 'california_housing','trec'], default = 'mnist',
        help = 'dataset where to run the experiments. Classification: MNIST or Fashion MNIST. Regression: California housing.')
    parser.add_argument('--optimizer_name', choices = ['sgd', 'rmsprop', 'adadelta', 'adagrad', 'adam', 'asgd'], default = 'adam',
        help = 'optimizer')
    parser.add_argument('--learning_rate', type = float, default = 0.0001,
        help = 'initial learning rate')
    parser.add_argument('--learning_rate_drop', type = float, default = 0.6,
        help = 'learning rate decay rate (step LR every learning_rate_steps)')
    parser.add_argument('--learning_rate_steps', type = int, default = 10,
        help = 'number of steps (epochs) before decaying the learning rate')
    parser.add_argument('--train_logvar_t', action = 'store_true', default = False,
        help = 'train the log(variance) of the bottleneck variable')
    parser.add_argument('--eval_rate', type = int, default = 20,
        help = 'evaluate I(X;T), I(T;Y) and accuracies every eval_rate epochs')
    parser.add_argument('--visualize', action = 'store_true', default = False,
        help = 'visualize the results every eval_rate epochs')
    parser.add_argument('--verbose', action = 'store_true', default = False,
        help = 'report the results every eval_rate epochs')

    return parser.parse_args()
