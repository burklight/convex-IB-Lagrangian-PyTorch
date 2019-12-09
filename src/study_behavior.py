from convexIB import ConvexIB
from utils import get_data
from utils import get_args
from visualization import plot_behavior, plot_clustering
import torch
import os
import numpy as np
import multiprocessing

torch.set_num_threads(1)

# Obtain the arguments
args = get_args()

# Obtain the data
trainset, validationset = get_data()

# Create the folders
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
    convex_IB = ConvexIB(n_x = 784, n_y = 10, K = args.K, beta = beta, logvar_t = args.logvar_t, 
        logvar_kde = args.logvar_kde, train_logvar_t = args.train_logvar_t, u_func_name = args.u_func_name, hyperparameter = args.hyperparameter)
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
plot_behavior(args.logs_dir,args.figs_dir,args.K,betas,args.train_logvar_t,np.log2(10),args.u_func_name,args.hyperparameter)
plot_clustering(args.logs_dir,args.figs_dir,betas,args.K,args.train_logvar_t,example=args.example_clusters)