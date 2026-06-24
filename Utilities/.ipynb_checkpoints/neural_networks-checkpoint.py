import torch

class feedforward(torch.nn.Module) :
    
    def __init__(self, layers_dim, activations, adaptive_activations = False, device = 'cuda', seed = 1234) :
        
        super().__init__()
        
        if len(layers_dim) < 3 :
            raise Exception('len(layers_dim) < 3')
        else :
            self.layers_dim = layers_dim
            
        if isinstance(activations, list) and len(activations) == len(layers_dim) - 2 :
            self.activations = activations
        elif isinstance(activations, str) :
            self.activations = (len(self.layers_dim) - 2)* [activations]
        else :
            raise Exception('problem in activations')
        
        self.layers = self.create_net()
        self.init_net(seed)
        
        if device == 'cpu' :
            self.device = torch.device('cpu')
        else :
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
        self.activations_parameters = self.get_activations_parameters(adaptive_activations)
        
        #print(self)
        #print('device :', self.device)
        
        
    def create_net(self) :
        layers = torch.nn.ModuleList()
        dim = self.layers_dim[0]
        for hdim in self.layers_dim[1 :] :
            layers.append(torch.nn.Linear(dim, hdim))
            dim = hdim
        return layers
    
    def init_net(self, seed) :
        torch.manual_seed(seed)
        for p in self.parameters() : 
            try :
                torch.nn.init.xavier_uniform_(p)
            except :
                torch.nn.init.constant_(p, 0)
    
    def get_activation(self, name) :
        if name == 'tanh' :
            return torch.tanh
        elif name == 'gelu' :
            return torch.nn.functional.gelu
        elif name == 'relu' :
            return torch.nn.functional.relu
        elif name == 'softplus' :
            return torch.nn.functional.softplus
        else :
            raise Exception('unsupported activation function')
            
    def get_activations_parameters(self, adaptive_activations, alpha=0.1, n=10) :
        
        if adaptive_activations == True :
            alpha = torch.tensor(alpha).to(self.device)
            return torch.nn.ParameterList(n* torch.nn.Parameter(alpha, requires_grad = True) 
                                       for _ in range(len(self.activations)))
        else :
            return list(1. for _ in range(len(self.activations)))
        
    def forward(self, x) :
        for i, layer in enumerate(self.layers[ : -1]) :
            a = self.activations_parameters[i]
            f = self.get_activation(self.activations[i])
            x = f(a* layer(x))
        return self.layers[-1](x)
