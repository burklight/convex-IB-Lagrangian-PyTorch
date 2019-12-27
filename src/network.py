import torch

class Deterministic_encoder(torch.nn.Module):
    '''
    Probabilistic encoder of the network.
    - We use the one in Kolchinsky et al. 2017 "Nonlinear Information Bottleneck"
    - Parameters:
        · K (int) : dimensionality of the bottleneck variable
        · n_x (int) : dimensionality of the input variable
    '''

    def __init__(self,K,n_x,network_type):
        super(Deterministic_encoder,self).__init__()

        self.K = K
        self.n_x = n_x
        self.network_type = network_type

        if self.network_type == 'mlp_mnist':
            layers = []
            layers.append(torch.nn.Linear(self.n_x,800))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(800,800))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(800,self.K))
            self.f_theta = torch.nn.Sequential(*layers)

        elif self.network_type == 'conv_net_fashion_mnist':
            layers = []
            layers.append(torch.nn.ReflectionPad2d(1))
            layers.append(torch.nn.Conv2d(1,5,4,2))
            layers.append(torch.nn.Conv2d(5,50,5,2))
            self.f_theta_conv = torch.nn.Sequential(*layers)
            
            layers = []
            layers.append(torch.nn.Linear(1250,800))
            layers.append(torch.nn.ReLU6())
            layers.append(torch.nn.Linear(800,self.K))
            self.f_theta_lin = torch.nn.Sequential(*layers)

        elif self.network_type == 'mlp_california_housing':
            layers = []
            layers.append(torch.nn.Linear(self.n_x,128))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(128,128))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(128,self.K))
            self.f_theta = torch.nn.Sequential(*layers)
        
        elif self.network_type == 'conv_net_trec':
            '''
            Network type largely inspired on 'https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/3%20-%20Faster%20Sentiment%20Analysis.ipynb'
            '''
            self.embedding = torch.nn.Embedding(n_x,100)
            self.convolutions = torch.nn.ModuleList([
                torch.nn.Conv2d(1,100,(fs,100)) for fs in [2,3,4]
            ])
            self.f_theta_lin = torch.nn.Linear(3*100,self.K)

    def forward(self,x):

        if self.network_type == 'mlp_mnist' or self.network_type == 'mlp_california_housing':
            x = x.view(-1,self.n_x)
            mean_t = self.f_theta(x)
        elif self.network_type == 'conv_net_fashion_mnist':
            mean_t_conv = self.f_theta_conv(x) 
            mean_t_conv = mean_t_conv.view(-1,1250)
            mean_t = self.f_theta_lin(mean_t_conv)
        elif self.network_type == 'conv_net_trec':
            x = x.permute(1,0)
            z = self.embedding(x)
            z = z.unsqueeze(1)
            z_convs = [torch.nn.functional.relu6(convolution(z)).squeeze(3) for convolution in self.convolutions]
            z_pooled = [torch.nn.functional.max_pool1d(z_conv, z_conv.shape[2]).squeeze(2) for z_conv in z_convs]
            z_cat = torch.cat(z_pooled,dim=1)
            mean_t = self.f_theta_lin(z_cat)

        return mean_t

class Deterministic_decoder(torch.nn.Module):
    '''
    Deterministic decoder of the network.
    - We use the one in Kolchinsky et al. 2017 "Nonlinear Information Bottleneck"
    - Parameters:
        · K (int) : dimensionality of the bottleneck variable
        · n_y (int) : dimensionality of the output variable (number of classes)
    '''

    def __init__(self,K,n_y,network_type):
        super(Deterministic_decoder,self).__init__()

        self.K = K
        self.network_type = network_type

        if network_type == 'mlp_mnist' or network_type == 'conv_net_fashion_mnist':
            layers = []
            layers.append(torch.nn.Linear(self.K,800))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(800,n_y))
            self.g_theta = torch.nn.Sequential(*layers)
        elif network_type == 'mlp_california_housing':
            layers = []
            layers.append(torch.nn.Linear(self.K,128))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(128,n_y))
            self.g_theta = torch.nn.Sequential(*layers)
        elif network_type == 'conv_net_trec':
            layers = []
            layers.append(torch.nn.Linear(self.K,128))
            layers.append(torch.nn.ReLU6())
            layers.append(torch.nn.Linear(128,n_y))
            self.g_theta = torch.nn.Sequential(*layers)


    def forward(self,t):

        logits_y =  self.g_theta(t)
        return logits_y

class nlIB_network(torch.nn.Module):
    '''
    Nonlinear Information Bottleneck network.
    - We use the one in Kolchinsky et al. 2017 "Nonlinear Information Bottleneck"
    - Parameters:
        · K (int) : dimensionality of the bottleneck variable
        · n_x (int) : dimensionality of the input variable
        · n_y (int) : dimensionality of the output variable (number of classes)
        · train_logvar_t (bool) : if true, logvar_t is trained
    '''

    def __init__(self,K,n_x,n_y,logvar_t=-1.0,train_logvar_t=False,network_type='mlp_mnist',TEXT=None):
        super(nlIB_network,self).__init__()

        self.network_type = network_type
        self.encoder = Deterministic_encoder(K,n_x,self.network_type)
        self.decoder = Deterministic_decoder(K,n_y,self.network_type)
        if train_logvar_t:
            self.logvar_t = torch.nn.Parameter(torch.Tensor([logvar_t]))
        else:
            self.logvar_t = torch.Tensor([logvar_t])
        if self.network_type == 'conv_net_trec':
            pretrained_embedding = TEXT.vocab.vectors
            self.encoder.embedding.weight.data.copy_(pretrained_embedding)
            UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
            PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
            self.encoder.embedding.weight.data[UNK_IDX] = torch.zeros(100)
            self.encoder.embedding.weight.data[PAD_IDX] = torch.zeros(100)

    def encode(self,x,random=True):

        mean_t = self.encoder(x)
        if random:
            t = mean_t + torch.exp(0.5*self.logvar_t) * torch.randn_like(mean_t)
        else:
            t = mean_t
        return t

    def decode(self,t):

        logits_y = self.decoder(t)
        return logits_y

    def forward(self,x):

        t = self.encode(x)
        logits_y = self.decode(t)
        return logits_y
