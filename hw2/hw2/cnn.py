import torch
import itertools as it
import torch.nn as nn


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int, channels: list,
                 pool_every: int, hidden_dims: list):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)
        
        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs should exist at the end, without a MaxPool after them.
        # ====== YOUR CODE: ======
        # Appending the first conv layer and after that relu
        layers.append(nn.Conv2d(in_channels,self.channels[0],kernel_size = 3,padding =1))
        layers.append(nn.ReLU())
        N = len(self.channels)
        N_divisable = N - N % self.pool_every
        remaining = N % self.pool_every
        # Running on the channel dims and appendin convolution layers and relu accordingly
        # After p convolution apply max pool  
        for idx in range(1,N_divisable):
            layers.append(nn.Conv2d(self.channels[idx-1],self.channels[idx],kernel_size = 3,padding =1,bias=True))
            layers.append(nn.ReLU())
            # Adding maxpool after every p convolutions 
            if (idx-1)%self.pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size = 2,stride=2))
        # Last maxpool layer
        #layers.append(nn.MaxPool2d(kernel_size = 2))
        # Append the remaining convolutions and relu
        for idx in range(remaining):
            layers.append(nn.Conv2d(self.channels[idx + N_divisable-1],self.channels[idx + N_divisable],kernel_size = 3,padding =1,bias=True))
            layers.append(nn.ReLU())
            
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        N = len(self.channels)
        # overall pulling operators
        N = N//self.pool_every 
        M = len(self.hidden_dims)
        in_channels, in_h, in_w, = tuple(self.in_size)
        
        # we had N//pool_every max pooling operations, number of output channels is
        # channels[-1] and we started with an image size of in_h*in_w 
        
        if len(self.channels) == 1:
            MLP_in_dim = (self.channels[-1])*in_h*in_w
        else:
            MLP_in_dim = (self.channels[-1])*(in_h//(2**N))*(in_w//(2**N))
        
        layers = []
        # TODO: Create the classifier part of the model:
        #  (Linear -> ReLU)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        layers.append(nn.Linear(MLP_in_dim,self.hidden_dims[0]))
        layers.append(nn.ReLU())
        for idx in range(M-1):
            layers.append(nn.Linear(self.hidden_dims[idx],self.hidden_dims[idx+1],bias=True))
            layers.append(nn.ReLU())
                
        layers.append(nn.Linear(self.hidden_dims[-1],self.out_classes,bias=True))
        
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        features = features.view(features.shape[0],-1)
        out = self.classifier(features)
        
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 batchnorm=False, dropout=0.):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
        convolution in the block. The length determines the number of
        convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
        be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
        convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
        Zero means don't apply dropout.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  the main_path, which should contain the convolution, dropout,
        #  batchnorm, relu sequences, and the shortcut_path which should
        #  represent the skip-connection.
        #  Use convolutions which preserve the spatial extent of the input.
        #  For simplicity of implementation, we'll assume kernel sizes are odd.
        # ====== YOUR CODE: ======
        main_layers = []
        
        # constructing the input layer 
        # we assume kernel sizes are odd so to preserve spacial dimentions we 
        # padd with the kernel size divided by 2
        padding = kernel_sizes[0]//2
        main_layers.append(nn.Conv2d(in_channels,channels[0],kernel_sizes[0],padding = padding))
        main_layers.append(nn.Dropout2d(dropout))
        if batchnorm ==True:    
            main_layers.append(nn.BatchNorm2d(channels[0]))
        main_layers.append(nn.ReLU())
        
        for idx in range(len(channels)-1):
            padding = kernel_sizes[idx+1]//2
            main_layers.append(nn.Conv2d(channels[idx],channels[idx +1],kernel_sizes[idx+1],padding = padding))

            if idx < len(channels)-2:    
                main_layers.append(nn.ReLU())
                main_layers.append(nn.Dropout2d(dropout))
                if batchnorm ==True:    
                    main_layers.append(nn.BatchNorm2d(channels[idx + 1]))
            
        if channels[-1] != in_channels:
            self.shortcut_path = nn.Sequential(nn.Conv2d(in_channels,channels[-1],kernel_size = 1,bias=False)
            )            
        else:    
            self.shortcut_path = nn.Sequential()
            
        self.main_path = nn.Sequential(*main_layers)
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ReLU)*P -> MaxPool]*(N/P)
        #   \------- SKIP ------/
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs (with a skip over them) should exist at the end,
        #  without a MaxPool after them.
        # ====== YOUR CODE: ======
        
        # Appending the first conv layer and after that relu        

        layers.append(ResidualBlock(in_channels,self.channels[0:self.pool_every],[3]*self.pool_every))
        layers.append(nn.MaxPool2d(kernel_size = 2,stride=2))

        N = len(self.channels)
        N_divisable = N - N % self.pool_every
        remaining = N % self.pool_every

        # Running on the channel dims and appending convolution layers and relu accordingly
        # After p convolution apply max pool  
        for idx_int,idx in enumerate(range(0,N_divisable,self.pool_every)):
            if idx == 0:
                continue
            layers.append(ResidualBlock(self.channels[idx-1],self.channels[idx:idx+self.pool_every],[3]*self.pool_every))
            layers.append(nn.MaxPool2d(kernel_size = 2,stride=2))
            
        
        if remaining > 0 :

            layers.append(ResidualBlock(self.channels[N_divisable-1],self.channels[N_divisable:N_divisable+remaining],[3]*remaining))
        # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================
