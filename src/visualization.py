import matplotlib.pyplot as plt
import numpy as np
from utils import get_data
from sklearn.cluster import DBSCAN

def init_results_visualization(K):
    '''
    Initializes the results plot
    '''

    if K == 2:
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize = (20+1,4), gridspec_kw={'width_ratios': [1,1,1,1.25,1]})
    else:
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize = (16+1,4), gridspec_kw={'width_ratios': [1,1,1,1.25,1]})
    return fig, ax

def plot_results(IXT_train, IXT_validation, ITY_train, ITY_validation,
    loss_train, loss_validation, t, y, epochs, HY, K, fig, ax):
    '''
    Plots the results in figure fig
    '''

    HY = HY / np.log(2) # in bits

    # Print the Loss
    ax[0].clear()
    ax[0].plot(epochs, loss_train, '-', color = 'red', label = 'train')
    ax[0].plot(epochs, loss_validation, '-', color = 'blue', label = 'validation')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel(r'$-\mathcal{L}_{IB}(T)$')
    ax[0].set_xlim(left=0)
    ax[0].legend()

    # Print the IXT
    ax[1].clear()
    ax[1].plot(epochs, IXT_train, '-', color = 'red', label = 'train')
    ax[1].plot(epochs, IXT_validation, '-', color = 'blue', label = 'validation')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel(r'$I(X;T)$')
    ax[1].set_ylim(bottom=0)
    ax[1].set_xlim(left=0)
    ax[1].legend()

    # Print the ITY
    lim = np.linspace(0,np.max(epochs),1000)
    ax[2].clear()
    ax[2].plot(lim, np.ones(1000)*HY, color = 'black', linestyle = '--')
    ax[2].plot(epochs, ITY_train, '-', color = 'red', label = 'train')
    ax[2].plot(epochs, ITY_validation, '-', color = 'blue', label = 'validation')
    ax[2].set_xlabel('epochs')
    ax[2].set_ylabel(r'$I(T;Y)$')
    ax[2].set_ylim(bottom=0, top=HY*(1.1))
    ax[2].set_xlim(left=0)
    ax[2].legend()

    # Print the information plane
    maxval = max(HY*1.1,max(np.max(IXT_train), np.max(IXT_validation)))
    diag = np.linspace(0,maxval,1000)
    ax[3].clear()
    ax[3].plot(diag, diag, color = 'darkorange', linestyle = '--')
    ax[3].plot(diag, np.ones(1000)*HY, color = 'darkorange', linestyle = '--')
    ax[3].fill_between(diag, 0, np.where(diag>HY, HY, diag), alpha = 0.5, color='darkorange')
    ax[3].plot(diag, np.where(diag>HY, HY, diag), alpha = 0.5, color='blue', linewidth=4)
    ax[3].plot(IXT_train[:-1], ITY_train[:-1], 'X', color='red', markersize=9, markeredgecolor='black', label = 'train')
    ax[3].plot(IXT_validation[:-1], ITY_validation[:-1], '.', color='blue', markersize=9, markeredgecolor='black', label = 'validation')
    ax[3].plot(IXT_train[-1], ITY_train[-1], '*', color='red', markersize=9, markeredgecolor='black')
    ax[3].plot(IXT_validation[-1], ITY_validation[-1], '*', color='blue', markersize=9, markeredgecolor='black')
    ax[3].plot(IXT_train, ITY_train, linestyle=':', color='red')
    ax[3].plot(IXT_validation, ITY_validation, linestyle=':', color='blue')
    ax[3].set_xlabel(r'$I(X;T)$')
    ax[3].set_ylabel(r'$I(T;Y)$')
    ax[3].annotate(r' $I(X;Y) = H(Y)$', xy=(maxval*0.75,HY*(1.05)), color='darkorange')
    ax[3].annotate(r' $I(X;T) \geq I(T;Y)$', xy=(HY*(0.5),HY*(1.05)), color='darkorange')
    ax[3].set_xlim(left=0,right=maxval)
    ax[3].set_ylim(bottom=0, top=HY*(1.1))
    ax[3].legend()

    if K == 2:
        # Print the representations
        npoints = 10000
        ax[4].clear()
        ax[4].scatter(t[0:npoints,0], t[0:npoints,1], c=y[0:npoints], cmap='tab10', marker='.', alpha=0.1)
        ax[4].set_xticks([])
        ax[4].set_yticks([])
        ax[4].set_xlabel('Bottleneck variable space')

    plt.tight_layout()
    plt.pause(0.01)

def plot_behavior(logs_dir,figs_dir,K,betas,train_logvar_t,HY,hfun='exp',param=1,compression=1,problem_type='classification',deterministic=True):

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize = (16,4))

    # Prepare the empty vectors
    
    IXT_train = np.empty(len(betas))
    IXT_validation = np.empty(len(betas))
    ITY_train = np.empty(len(betas))
    ITY_validation = np.empty(len(betas))

    # For all betas
    for i,beta in enumerate(betas):

        # Load the information plane point they obtained
        name_base = "K-" + str(K) + "-B-" + str(round(beta,3)).replace('.', '-') \
            + "-Tr-" + str(bool(train_logvar_t)) + '-'
        IXT_train[i] = np.load(logs_dir + name_base + 'train_IXT.npy')[-1]
        IXT_validation[i] = np.load(logs_dir + name_base + 'validation_IXT.npy')[-1] 
        ITY_train[i] = np.load(logs_dir + name_base + 'train_ITY.npy')[-1] 
        ITY_validation[i] = np.load(logs_dir + name_base + 'validation_ITY.npy')[-1] 
        
    # Create expected behavior
    if deterministic:
        if hfun == 'exp':
            bmin = (1 / (param * np.exp(param * HY)))
            betas_th = [bmin] + list(betas[betas >= bmin])
            IXT_expected = np.empty(len(betas_th))
            IXT_expected = np.array([-np.log(param*b + 1e-10)/param for b in betas_th])
        elif hfun == 'pow':
            bmin = (1/((1+param)*HY**param))
            betas_th = [bmin] + list(betas[betas >= bmin])
            IXT_expected = np.empty(len(betas_th))
            IXT_expected = np.array([(1/((1+param)*b + 1e-10)**(1/param)) for b in betas_th])
        elif hfun == 'shifted-exp':
            betas_th = betas 
            IXT_expected = -1.0*np.log(param*betas+1e-10)/param + compression
        ITY_expected = IXT_expected

    # Truncate too large IXT for uncompressed representations so we can see them in the plot
    if betas[0] == 0 and IXT_train[0] > HY * 9:
        IXT_train[0] = HY * 9
        IXT_validation[0] = HY * 9

    # Print the information plane
    maxval = max(HY*2,max(np.max(IXT_train), np.max(IXT_validation)))
    diag = np.linspace(0,maxval,1000)
    ax[0].plot(diag, diag, color = 'darkorange', linestyle = '--')
    ax[0].plot(diag, np.ones(1000)*HY, color = 'darkorange', linestyle = '--')
    ax[0].fill_between(diag, 0, np.where(diag>HY, HY, diag), alpha = 0.5, color='darkorange')
    ax[0].plot(diag, np.where(diag>HY, HY, diag), alpha = 0.5, color='blue', linewidth=4)
    if deterministic:
        ax[0].plot(IXT_expected, ITY_expected, '*:', color='green', markeredgecolor='black', markersize=9, label = 'theoretical')
    ax[0].plot(IXT_train, ITY_train, 'X:', color='red', markersize=9, markeredgecolor='black', label = 'train')
    ax[0].plot(IXT_validation, ITY_validation, '.:', color='blue', markersize=9, markeredgecolor='black', label = 'validation')
    ax[0].set_xlabel(r'$I(X;T)$')
    ax[0].set_ylabel(r'$I(T;Y)$')
    if problem_type == 'classification':
        if deterministic:
            ax[0].annotate(r' $I(X;Y) = H(Y)$', xy=(maxval*0.75,HY*(1.05)), color='darkorange')
        else:
            ax[0].annotate(r' $H(Y)$', xy=(maxval*0.75,HY*(1.05)), color='darkorange')
    else:
        ax[0].annotate(r' $I(X;Y)$', xy=(maxval*0.75,HY*(1.05)), color='darkorange')
    ax[0].annotate(r' $I(X;T) \geq I(T;Y)$', xy=(HY*(1.05),HY*(1.05)), color='darkorange')
    ax[0].set_xlim(left=0,right=max(maxval,HY*2.5))
    ax[0].set_ylim(bottom=0, top=HY*(1.1))
    ax[0].legend()

    # Print the evolution of I(T;Y)
    diag = np.linspace(np.min(betas),np.max(betas),1000)
    ax[1].plot(diag, np.ones(1000)*HY, color = 'darkorange', linestyle = '--')
    if deterministic:
        ax[1].plot(betas_th, ITY_expected, '*:', color = 'green', markeredgecolor='black', markersize=9, label='theoretical')
    ax[1].plot(betas, ITY_train, 'X:', color='red', markersize=9, markeredgecolor='black', label = 'train')
    ax[1].plot(betas, ITY_validation, '.:', color = 'blue', markersize=9, markeredgecolor='black', label='validation')
    ax[1].set_xlabel(r'$\beta$')
    ax[1].set_ylabel(r'$I(T;Y)$')
    if problem_type == 'classification':
        if deterministic:
            ax[1].annotate(r' $I(X;Y) = H(Y)$', xy=(betas[-1]*0.75,HY*(1.05)), color='darkorange')
        else:
            ax[1].annotate(r' $H(Y)$', xy=(betas[-1]*0.75,HY*(1.05)), color='darkorange')
    else:
        ax[1].annotate(r' $I(X;Y)$', xy=(betas[-1]*0.75,HY*(1.05)), color='darkorange')
    ax[1].set_xlim(left=np.min(betas),right=np.max(betas))
    ax[1].set_ylim(bottom=0,top=HY*(1.1))
    ax[1].legend()

    # Print the evolution of I(X;T)
    diag = np.linspace(np.min(betas),np.max(betas),1000)
    ax[2].plot(diag, np.ones(1000)*HY, color = 'darkorange', linestyle = '--')
    if deterministic:
        ax[2].plot(betas_th, IXT_expected, '*:', color = 'green', markersize=9, markeredgecolor='black', label='theoretical')
    ax[2].plot(betas, IXT_train, 'X:', color='red', markersize=9, markeredgecolor='black', label = 'train')
    ax[2].plot(betas, IXT_validation, '.:', color = 'blue', markersize=9,  markeredgecolor='black', label='validation')
    ax[2].set_xlabel(r'$\beta$')
    ax[2].set_ylabel(r'$I(X;T)$')
    if problem_type == 'classification':
        if deterministic:
            ax[2].annotate(r' $I(X;Y) = H(Y)$', xy=(betas[-1]*0.75,HY*(1.05)), color='darkorange')
        else:
            ax[2].annotate(r' $H(Y)$', xy=(betas[-1]*0.75,HY*(1.05)), color='darkorange')
    else:
        ax[2].annotate(r' $I(X;Y)$', xy=(betas[-1]*0.75,HY*(1.05)), color='darkorange')
    ax[2].set_xlim(left=np.min(betas),right=np.max(betas))
    ax[2].set_ylim(bottom=0,top=maxval)
    ax[2].legend()

    plt.tight_layout()
    plt.plot()
 
    name_base = "K-" + str(K) + "-NB-" + str(len(betas)).replace('.', '-') \
        + "-Tr-" + str(bool(train_logvar_t)) + '-'
    plt.savefig(figs_dir + name_base + 'behavior.pdf', format = 'pdf')
    plt.savefig(figs_dir + name_base + 'behavior.png', format = 'png')


def plot_clustering(logs_dir,figs_dir,betas,K=2,train_logvar_t=False,example='False'):

    # Get name for the images
    name_base = "K-" + str(K) + "-NB-" + str(len(betas)).replace('.', '-') \
        + "-Tr-" + str(bool(train_logvar_t)) + '-'

    # Load hidden variables 
    name_base = "K-" + str(K) + "-B-" + str(round(betas[0],3)).replace('.', '-') \
            + "-Tr-" + str(bool(train_logvar_t)) + '-'
    npoints = min(10000,len(np.load(logs_dir + name_base + 'hidden_variables.npy')))
    hidden_variables = np.empty((len(betas),npoints,2))
    for i,beta in enumerate(betas):
        name_base = "K-" + str(K) + "-B-" + str(round(beta,3)).replace('.', '-') \
            + "-Tr-" + str(bool(train_logvar_t)) + '-'
        hidden_variables[i] = np.load(logs_dir + name_base + 'hidden_variables.npy')

    # Cluster to show in example of clusters 
    if example:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,6))
        fig.subplots_adjust(wspace=0,hspace=0)
        dict_selected_clusters = {10:[0,0],8:[0,1],6:[1,0],2:[1,1]} # example of clusters to show

    # Compute clusterization 
    n_clusters = np.empty(len(betas))
    for i,beta in enumerate(betas):
        if i == 0:
            n_clusters[i] = 10 # we know beforehand
            continue
        db = DBSCAN(eps=0.3, min_samples=50).fit(hidden_variables[i])
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters[i] = len(set(labels)) - (1 if -1 in labels else 0)
        if example and n_clusters[i] in dict_selected_clusters:
            pos_x, pos_y = dict_selected_clusters[n_clusters[i]]
            ax[pos_x,pos_y].clear()

            unique_labels = set(labels)
            colors = [plt.cm.tab10(each) for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = [0, 0, 0, 1]

                class_member_mask = (labels == k)
                xy = hidden_variables[i,class_member_mask & core_samples_mask]
                ax[pos_x, pos_y].plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(col), alpha = 0.2)
                xy = hidden_variables[i,class_member_mask & ~core_samples_mask]
                ax[pos_x, pos_y].plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(col),  markeredgecolor='gray', markersize=5, alpha = 0.03)

            ax[pos_x,pos_y].annotate(r'$\beta$='+str(round(beta,3)), xy=(45, 160), xycoords='axes points', size=10, ha='right', va='top')
            ax[pos_x,pos_y].set_xticks([])
            ax[pos_x,pos_y].set_yticks([])
    
    # If we want to show the example of clusters, show it
    if example:
        plt.plot()
        plt.savefig(figs_dir + name_base + 'clusters_examples.pdf', format = 'pdf')
        plt.savefig(figs_dir + name_base + 'clusters_examples.png', format = 'png')

    
    # Plot the clusterization vs betas
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (6,4))
    ax.plot(betas, n_clusters, '*:', color='orange', markersize=9, markeredgecolor='black')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('# clusters')
    plt.plot()
    plt.savefig(figs_dir + name_base + 'clusterization.pdf', format = 'pdf')
    plt.savefig(figs_dir + name_base + 'clusterization.png', format = 'png')
