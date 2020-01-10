from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class DiscriminatorConv(nn.Module):
    """
    a convolution block for the encoder
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 stride_list:list,padding_list:list ,batchnorm=True, dropout=0.2,bias_flag = False):
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
        assert len(channels) == len(stride_list) 
        assert len(channels) == len(padding_list) 

        #assert all(map(lambda x: x % 2 == 1, kernel_sizes))
        self.out_channels = channels[-1]
        self.main_path= None

        # DONE: Implement a generic residual block.
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

        main_layers.append(nn.Conv2d(in_channels,channels[0],kernel_sizes[0],padding = padding_list[0]
                                     ,stride = stride_list[0],bias = bias_flag))
        main_layers.append(nn.Dropout2d(dropout))
        if batchnorm ==True:    
            main_layers.append(nn.BatchNorm2d(channels[0]))
        main_layers.append(nn.ELU(alpha = 0.5)) 
        
        
        for idx in range(len(channels)-1):
            
            main_layers.append(nn.Conv2d(channels[idx],channels[idx +1],kernel_sizes[idx+1],padding =padding_list[idx+1],
                                         stride = stride_list[idx+1],bias = bias_flag))

            if idx < len(channels)-2:    
                #main_layers.append(nn.ReLU())
                main_layers.append(nn.ELU(alpha = 0.5))
                main_layers.append(nn.Dropout2d(dropout))
                if batchnorm ==True:    
                    main_layers.append(nn.BatchNorm2d(channels[idx + 1]))
        
        main_layers.append(nn.Sigmoid())              
        self.main_path = nn.Sequential(*main_layers)
        self.layers_list = main_layers
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out = torch.relu(out)
        return out



class GeneratorConv(nn.Module):
    """
    a convolution block for the decoder
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 stride_list:list,padding_list:list ,batchnorm=True, dropout=0.2,bias_flag = False):
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
        assert len(channels) == len(stride_list) 
        assert len(channels) == len(padding_list) 

        #assert all(map(lambda x: x % 2 == 1, kernel_sizes))
        self.out_channels = channels[-1]
        self.main_path= None

        # DONE: Implement a generic residual block.
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

        main_layers.append(nn.ConvTranspose2d(in_channels,channels[0],kernel_sizes[0],padding = padding_list[0]
                                     ,stride = stride_list[0],bias = bias_flag))
        main_layers.append(nn.Dropout2d(dropout))
        if batchnorm ==True:    
            main_layers.append(nn.BatchNorm2d(channels[0]))
        main_layers.append(nn.ELU(alpha = 0.5)) 
        
        
        for idx in range(len(channels)-1):
            
            main_layers.append(nn.ConvTranspose2d(channels[idx],channels[idx +1],kernel_sizes[idx+1],padding =padding_list[idx+1],
                                         stride = stride_list[idx+1],bias = bias_flag))

            if idx < len(channels)-2:    
                #main_layers.append(nn.ReLU())
                main_layers.append(nn.ELU(alpha = 0.5))
                main_layers.append(nn.Dropout2d(dropout))
                if batchnorm ==True:    
                    main_layers.append(nn.BatchNorm2d(channels[idx + 1]))
                        
        self.main_path = nn.Sequential(*main_layers)
        self.layers_list = main_layers
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out = torch.relu(out)
        return out



class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        self.in_size = in_size
        self.in_channels = self.in_size[0]
        self.out_channels = 256
        self.channels_list = [128,256,512,self.out_channels]
        self.kernels_list = [4]*len(self.channels_list)
        self.stride_list = [2]*(len(self.channels_list)-1)+[1]
        self.padding_list = [1]*(len(self.channels_list)-1) + [0]
        
        self.model_conv = DiscriminatorConv(self.in_channels, self.channels_list,self.kernels_list,
                             self.stride_list,self.padding_list)
        l_size = [1]+[self.in_size[0]] +[self.in_size[1]] +[self.in_size[2]]
        test = torch.randn(l_size)
        featurs = self.model_conv(test)
        featurs = featurs.view(1,-1)
        
        self.fc_in_dim = featurs.shape[1]
        self.fc_hidden = [64,1]
        #Creating a fc NN for the classification part
        layers = []
        M = len(self.fc_hidden)
        mlp_in_dim = self.fc_in_dim
        for idx in range(M):
            layers.append(nn.Linear(mlp_in_dim,self.fc_hidden[idx]))
            layers.append(nn.ReLU())
            mlp_in_dim = self.fc_hidden[idx]
            layers.append(nn.Dropout(p = 0.1))
            
        layers.append(nn.Linear(mlp_in_dim,1))
        
        self.model_fc = nn.Sequential(*layers)
        
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        
        batch_size = x.shape[0]
        featurs = F.tanh(self.model_conv(x))
        featurs = featurs.view(batch_size,-1)
        
        print(self.fc_in_dim)
        print(featurs.shape)
        
        y = self.model_fc(featurs)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        self.featuremap_size = featuremap_size
        self.out_channels = out_channels
        
        self.channels_list = [512,256,128,64,self.out_channels]
        self.kernels_list = [self.featuremap_size]*len(self.channels_list)
        #self.stride_list = [1]+[2]*(len(self.channels_list)-1)
        self.stride_list = [2]*len(self.channels_list)
        
        self.padding_list = [0]+[1]*(len(self.channels_list)-1)        
        
        
        
        self.net_G = GeneratorConv(self.z_dim,self.channels_list,self.kernels_list,
                             self.stride_list,self.padding_list)
        
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # DONE: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        z = torch.randn(n,self.z_dim,device = device)
        #,requires_grad = with_grad
        samples = self.forward(z)
        
        if with_grad==False:
            samples = samples.detach()
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        z = z.view(z.shape[0],-1,1,1)
        x  = self.net_G(z)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    
    # Range of the uniform sampling
    u_range = 0.5*label_noise

    # Creating the noisy discriminator teargets 
    target_data = data_label*torch.ones_like(y_data) + label_noise*torch.rand_like(y_data) - u_range
    target_generated = (1-data_label)*torch.ones_like(y_data) + label_noise*torch.rand_like(y_data) - u_range    
    
    # our criterion 
    criterion = nn.BCEWithLogitsLoss()
    
    # Target and labels should be on same device
    target_data.to(y_data.device)
    target_generated.to(y_generated.device)
    
    loss_data = criterion(y_data,target_data)
    loss_generated = criterion(y_generated,target_generated)
    

    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    target = torch.ones_like(y_generated)*data_label
    target.to(y_generated.device)
    
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(y_generated,target)
    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    #Train Discriminator with real and fake images 
    
    dsc_optimizer.zero_grad()
    
    data = x_data
    
    y_data = dsc_model(data)
    y_gen = dsc_model(gen_model.sample(y_data.shape[0],with_grad = False))
    
    dsc_loss = dsc_loss_fn(y_data,y_gen)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    
    #Train generator 
    gen_optimizer.zero_grad()
    data_gen = dsc_model(gen_model.sample(y_data.shape[0],with_grad = True))
    
    gen_loss = gen_loss_fn(data_gen)
    
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f'{checkpoint_file}.pt'

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    pass
    # ========================

    return saved
