from convexIB import ConvexIB
from utils import get_data
from utils import get_args
from visualization import plot_behavior, plot_clustering
import torch
import os
import numpy as np
import multiprocessing
import math

torch.set_num_threads(1)

# Obtain the arguments
args = get_args()

# Obtain the data
dataset_name = args.dataset
if dataset_name == 'mnist':
    trainset, validationset = get_data(dataset_name)
    n_x = 784
    n_y = 10
    network_type = 'mlp_mnist'
    maxIXY = np.log2(10)
    problem_type = 'classification'
    TEXT = None
elif dataset_name == 'fashion_mnist':
    trainset, validationset = get_data(dataset_name)
    n_x = (28,28)
    n_y = 10
    network_type = 'conv_net_fashion_mnist'
    maxIXY = np.log2(n_y)
    problem_type = 'classification'
    TEXT = None
elif dataset_name == 'california_housing':
    trainset, validationset = get_data(dataset_name)
    n_x = 8
    n_y = 1
    network_type = 'mlp_california_housing'
    varY = torch.var(trainset.targets)
    HY = 0.5 * math.log(varY.item() * 2.0 * math.pi * math.e) / math.log(2)
    maxIXY = 0.72785 / math.log(2) # Estimation by training with only the cross entropy and getting the result after training 
    problem_type = 'regression'
    TEXT = None
elif dataset_name == 'trec':
    trainset, validationset, TEXT, LABEL = get_data(dataset_name)
    n_x = len(TEXT.vocab)
    n_y = len(LABEL.vocab)
    network_type = 'conv_net_trec'
    maxIXY = np.log2(n_y)
    problem_type = 'classification'

# Create the folders
args.logs_dir = os.path.join(args.logs_dir,dataset_name) + '/'
args.figs_dir = os.path.join(args.figs_dir,dataset_name) + '/'
args.models_dir = os.path.join(args.models_dir,dataset_name) + '/'
os.makedirs(args.logs_dir) if not os.path.exists(args.logs_dir) else None
os.makedirs(args.figs_dir) if not os.path.exists(args.figs_dir) else None
os.makedirs(args.models_dir) if not os.path.exists(args.models_dir) else None

# Create specific folders for the function chosen 
args.logs_dir += args.u_func_name + '_' + str(round(args.hyperparameter,2)).replace('.', '-') + '/'
args.figs_dir += args.u_func_name + '_' + str(round(args.hyperparameter,2)).replace('.', '-') + '/'
args.models_dir += args.u_func_name + '_' + str(round(args.hyperparameter,2)).replace('.', '-') + '/'
os.makedirs(args.logs_dir) if not os.path.exists(args.logs_dir) else None
os.makedirs(args.figs_dir) if not os.path.exists(args.figs_dir) else None
os.makedirs(args.models_dir) if not os.path.exists(args.models_dir) else None

# Range of the lagrange multiplier
betas = np.linspace(args.beta_lim_min,args.beta_lim_max,args.n_betas)

# Function for multiprocessing
def train_and_save(beta):

    print("--- Studying Non-Linear IB behavior with beta = " + str(round(beta,3)) + " ---")
    # Train the network
    convex_IB = ConvexIB(n_x = n_x, n_y = n_y, problem_type = problem_type, network_type = network_type, K = args.K, beta = beta, logvar_t = args.logvar_t, 
        logvar_kde = args.logvar_kde, train_logvar_t = args.train_logvar_t, u_func_name = args.u_func_name, hyperparameter = args.hyperparameter, TEXT=TEXT)
    convex_IB.fit(trainset, validationset, n_epochs = args.n_epochs, learning_rate = args.learning_rate,
        learning_rate_drop = args.learning_rate_drop, learning_rate_steps = args.learning_rate_steps, sgd_batch_size = args.sgd_batch_size,
        mi_batch_size = args.mi_batch_size, same_batch = args.same_batch, eval_rate = args.eval_rate, optimizer_name = args.optimizer_name,
        verbose = args.verbose, visualization = args.visualize, logs_dir = args.logs_dir, figs_dir = args.figs_dir)

    # Save the network
    name_base = "K-" + str(args.K) + "-B-" + str(round(args.beta,3)).replace('.', '-') \
        + "-Tr-" + str(bool(args.train_logvar_t)) + '-'
    torch.save(convex_IB.state_dict(), args.models_dir + name_base + 'model')


# For all betas...
with multiprocessing.Pool(8) as p:
    p.map(train_and_save, betas)

# Visualize the comparison
plot_behavior(args.logs_dir,args.figs_dir,args.K,betas,args.train_logvar_t,maxIXY,args.u_func_name,args.hyperparameter)
if problem_type == 'classification':
    plot_clustering(args.logs_dir,args.figs_dir,betas,args.K,args.train_logvar_t,example=args.example_clusters)