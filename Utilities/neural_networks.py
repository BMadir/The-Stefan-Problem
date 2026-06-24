import torch

# Feed Forward
class feedforward(torch.nn.Module):
    
    def __init__(
            self,
            layers_dim,
            activations,
            adapt_act=False,
            device='cuda',
            seed=1234
    ):
        super().__init__()
        if len(layers_dim) < 3:
            raise Exception('len(layers_dim) < 3')
        else:
            self.layers_dim = layers_dim
            
        if isinstance(activations, list) and len(activations) == len(layers_dim) - 2:
            self.activations = activations
        elif isinstance(activations, str):
            self.activations = (len(self.layers_dim) - 2)* [activations]
        else:
            raise Exception('problem in activations')

        if device == 'cpu':
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        
        self.layers = self._layers(layers_dim)
        self._act_fns = [self._act_fn(n) for n in self.activations]
        self._act_params = self.__act_params(adapt_act)
        self._initialize(seed)

        self.to(self.device)

    def weights(self):
        for n, p in self.named_parameters():
            if "weight" in n:
                yield p

    def bias(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                yield p

    def _numel(self):
        self.numel = sum(
            2 * p.numel() if torch.is_complex(p) else p.numel() for p in self.parameters()
        )
        return self.numel

    def _layers(self, layers_dim):
        layers = torch.nn.ModuleList()
        dim = layers_dim[0]
        for hdim in layers_dim[1 :]:
            layers.append(torch.nn.Linear(dim, hdim))
            dim = hdim
        return layers

    def _initialize(self, seed):
        torch.manual_seed(seed)
        for l in self.layers:
            torch.nn.init.xavier_uniform_(l.weight)
            torch.nn.init.constant_(l.bias, 0)

    def _act_fn(self, name):

        if name == "sf":
            return  lambda x: torch.sin(2* torch.pi* x)
        if hasattr(torch.nn, name):
            return getattr(torch.nn, name)()
        if hasattr(torch, name):
            return getattr(torch, name)
        elif hasattr(torch.nn.functional, name):
            return getattr(torch.nn.functional, name)
        else:
            raise Exception('problem in activations')
    
    def __act_params(self, adapt_act, a=0.1, n=10):
        len_act = len(self.activations)
        if adapt_act:
            a = torch.tensor(a, device=self.device)
            return torch.nn.ParameterList(n* torch.nn.Parameter(a, requires_grad=True) for _ in range(len_act))
        else:
            return list(1. for _ in range(len_act))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[: -1]):
            a = self._act_params[i]
            f = self._act_fns[i]
            x = f(a* layer(x))
        return self.layers[-1](x)

    def save(self, name='model', file=''):
        save_dict = {
            'layers_dim': self.layers_dim,
            'activations': self.activations,
            'state_dict': self.state_dict()
        }
        torch.save(save_dict, file + name + '.pth')

# Sinusoidal features
class sf(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, trainable=True, seed=1234):
        super().__init__()
        linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.seed = seed
        if trainable:
            pass
        else:
            linear.weight.requires_grad_(False)
            if bias:
                linear.bias.requires_grad_(False)
        self._initialize(linear)
        self.linear = linear
        self.weight = linear.weight
        self.bias = linear.bias

    @staticmethod
    def gamma(x):
        x = 2 * torch.pi * x
        return torch.sin(x)

    def _initialize(self, layer, mean=0., std=1., constant=0.):
        torch.nn.init.normal_(layer.weight, mean=mean, std=std, generator=torch.Generator().manual_seed(self.seed))
        torch.nn.init.constant_(layer.bias, constant)

    def forward(self, x):
        y = self.linear(x)
        y = self.gamma(y)
        return y

# Fourier features
class ff(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, trainable=True, seed=1234):
        super().__init__()
        self.seed = seed
        out_features = out_features if (out_features % 2 == 0) else out_features + 1
        linear = torch.nn.Linear(in_features, out_features // 2, bias=bias)
        if trainable:
            pass
        else:
            linear.weight.requires_grad_(False)
            if bias:
                linear.bias.requires_grad_(False)
        self._initialize(linear)
        self.linear = linear
        self.weight = linear.weight
        self.bias = linear.bias

    @staticmethod
    def gamma(x):
        x = 2 * torch.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], 1)

    def _initialize(self, layer, mean=0., std=1., constant=0.):
        torch.nn.init.normal_(layer.weight, mean=mean, std=std, generator=torch.Generator().manual_seed(self.seed))
        torch.nn.init.constant_(layer.bias, constant)

    def forward(self, x):
        y = self.linear(x)
        y = self.gamma(y)
        return y


class ff_net(feedforward):
    def __init__(self, layers_dim, activations, device='cuda', seed=123):
        super().__init__(layers_dim=layers_dim, activations=activations, device=device, seed=seed)
        assert layers_dim[1] % 2 == 0, "layers_dim[1] % 2 != 0"
        _ff = ff(layers_dim[0], layers_dim[1], seed=seed).to(device)
        self.layers[0] = _ff
        self._act_fns[0] = lambda x: x
        self.activations[0] = "identity"


class sf_net(feedforward):
    def __init__(self, layers_dim, activations, device='cuda', seed=1234):
        super().__init__(layers_dim=layers_dim, activations=activations, device=device, seed=seed)
        _sf = sf(layers_dim[0], layers_dim[1], seed=seed).to(device)
        self.layers[0] = _sf
        self._act_fns[0] = lambda x: x
        self.activations[0] = "identity"

# Residual network
class ResNetBlock(torch.nn.Module):
    
    def __init__(self, input_size, output_size, nonlinearity='tanh', id_parameter=True, device='cpu'):
        super().__init__()
        
        self.nonlinearity = getattr(torch.nn.functional, nonlinearity)
        self.linear_1 = torch.nn.Linear(input_size, output_size)
        self.linear_2 = torch.nn.Linear(output_size, output_size)
        
        if id_parameter:
            self.id_parameter = torch.nn.Parameter(torch.randn((1, output_size), device=torch.device(device)), requires_grad=True)
        else:
            self.id_parameter = torch.ones((1, output_size), device=torch.device(device))
            
    def pad(self, x, y):
        "pad x with zeros (unpad is also possible but not recommended), to match y size"
        
        n = y.shape[1] - x.shape[1]
        pad = (int(n/2), n - int(n/2))
        
        return torch.nn.functional.pad(x, pad)

    def forward(self, x):
        
        y = self.linear_2(self.nonlinearity(self.linear_1(x)))
        
        return y + self.id_parameter *self.pad(x, y)
    

class ResNet(torch.nn.Module):
    def __init__(self, layers_dim, activations, device='cuda', seed=1234):
        super().__init__()
        
        if device == 'cpu':
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            
        if len(layers_dim) < 3:
            raise Exception('len(layers_dim) < 3')
        else:
            self.layers_dim = layers_dim
            
        if isinstance(activations, list) and len(activations) == len(layers_dim) - 2:
            self.activations = activations
        elif isinstance(activations, str):
            self.activations = (len(self.layers_dim) - 2)* [activations]
        else:
            raise Exception('problem in activations')
        
        self.layers = self.create_layers()
        self.init_net(seed)
        
        self.to(self.device)
        
    def create_layers(self):
        layers = torch.nn.ModuleList()
        dim = self.layers_dim[0]
        for hdim in self.layers_dim[1 :]:
            layers.append(ResNetBlock(dim, hdim, device=self.device))
            dim = hdim
        return layers
    
    def init_net(self, seed):
        torch.manual_seed(seed)
        for p in self.parameters():
            try:
                torch.nn.init.xavier_uniform_(p)
            except:
                torch.nn.init.constant_(p, 0)

    def weights(self):
        for n, p in self.named_parameters():
            if "weight" in n:
                yield p

    def bias(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                yield p

    def _numel(self):
        self.numel = sum(
            2 * p.numel() if torch.is_complex(p) else p.numel() for p in self.parameters()
        )
        return self.numel
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[: -1]):
            f = getattr(torch.nn.functional, self.activations[i])
            x = f(layer(x))
        return self.layers[-1](x)
