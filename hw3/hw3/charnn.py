import re

import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor

def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # DONE:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    all_chars = set(text)
    idx_to_char = dict(enumerate(all_chars))
    char_to_idx = {char:ind for ind,char in idx_to_char.items()}
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # DONE: Implement according to the docstring.
    # ====== YOUR CODE: ======
    count = 0
    text_clean = text
    for char_to_remove in chars_to_remove:
        text_clean = text_clean.replace(char_to_remove,'')
    
    n_removed = len(text) - len(text_clean)        
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tesnsor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # DONE: Implement the embedding.
    # ====== YOUR CODE: ======
    N = len(text)
    D = len(char_to_idx)
    result = torch.zeros([N,D],dtype = torch.int8)
    for ind,char in enumerate(text):
        result[ind,char_to_idx[char]] = 1
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # DONE: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    N = len(embedded_text)
    D = len(idx_to_char)
    s = str()
    for vec in embedded_text:
        ind = torch.where(vec == 1)
        ind = ind[0].item()
        s = s + idx_to_char[ind] 
    result = s
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int,
                              device='cpu'):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    # create an embeding for the text 
    embeded_text = []
    for s in text:
        embeded_text.append(char_to_idx[s])
        
    #droping the end of the text so it will fit in size
    shift_embed_text = embeded_text[1:]
    
    N = len(text)//seq_len
    shift_embed_text = shift_embed_text[:N*seq_len] 
    
    embeded_text = embeded_text[0:N*seq_len]
    emb_dim = len(char_to_idx)
     
    #creating sequences
    samples = chars_to_onehot(text[:N*seq_len],char_to_idx)
    samples = samples.view(N,seq_len,emb_dim)
    
    #creating labels
    labels = torch.tensor(shift_embed_text[:N*seq_len])
    labels = labels.view(N,-1)
    
    samples.to(device)
    labels.to(device)
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    result = y/temperature
    result = torch.softmax(result,dim = dim)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    
    with torch.no_grad():
        h_t = None
        x = torch.unsqueeze(chars_to_onehot(start_sequence, char_to_idx), 0)
        
        for i in range(n_chars - len(start_sequence)):
            x = x.type(torch.float)
            
            x = x.to(model.device)
            
            y, h_t = model(x, hidden_state=h_t)
            
            prob = hot_softmax(y[0, -1, :], temperature=T)
            
            x_samp = torch.multinomial(prob, 1)
            
            out_text += idx_to_char[x_samp.item()]
            
            x = torch.unsqueeze(chars_to_onehot(out_text[-1], char_to_idx), 0)
    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents  one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of indices is takes, samples in the same index of
        #  adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        N = int(self.batch_size*(len(self.dataset)//self.batch_size))
        idx = [i for i in range(0,N) ]
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """
    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        
        #self.sigmoid = nn.Sigmoid()
        #self.tanh = nn.Tanh()
        
        # TODO -   Dropout layer 
        #self.drop_layer = nn.Dropout(dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for layer in range(self.n_layers):
            input_dim = self.h_dim
            if layer == 0:
                input_dim = self.in_dim
           
            # Define parameters for update gate (z)
            W_xz = nn.Linear(input_dim,self.h_dim,bias = False)
            W_hz = nn.Linear(self.h_dim,self.h_dim)
            
            # Define parameters for reset gate (r)
            W_xr = nn.Linear(input_dim,self.h_dim,bias = False)
            W_hr = nn.Linear(self.h_dim,self.h_dim)
            
            # Define parameters for candidate (g)
            W_xg = nn.Linear(input_dim,self.h_dim,bias = False)
            W_hg = nn.Linear(self.h_dim,self.h_dim)
            
            # Dropout
            dropout_layer = nn.Dropout(dropout)
            
           
            W_xz = W_xz.to(self.device)
            W_hz = W_hz.to(self.device)
            
            W_xr = W_xr.to(self.device)
            W_hr = W_hr.to(self.device)

            W_xg = W_xg.to(self.device)
            W_hg = W_hg.to(self.device)
            dropout_layer = dropout_layer.to(self.device)
            
            
            
            # Appending the model parameters  
            self.layer_params.append((W_xz , W_hz , W_xr , W_hr , W_xg , W_hg ,dropout_layer))
            
            # Adding modules for update gate (z)
            self.add_module(name= 'W_xz_layer_{}'.format(layer), module=W_xz)
            self.add_module(name= 'W_hz_layer_{}'.format(layer), module=W_hz)
            
            # Adding modules for reset gate (r)
            self.add_module(name= 'W_xr_layer_{}'.format(layer), module=W_xr)
            self.add_module(name= 'W_hr_layer_{}'.format(layer), module=W_hr)
            
            # Adding modules for candidate (g)
            self.add_module(name= 'W_xg_layer_{}'.format(layer), module=W_xg)
            self.add_module(name= 'W_hg_layer_{}'.format(layer), module=W_hg)
            
            # Add dropout
            self.add_module(name= 'dropout_{}'.format(layer), module=dropout_layer)
            
        # Output layer
        self.W_y = nn.Linear(self.h_dim,self.out_dim)
        self.W_y = self.W_y.to(self.device)
 
        
        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape
        input = input.to(self.device)
        
        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device))
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        self.to(device=layer_input.device)
        
        out_list = []
        for seq in range(seq_len):
            X_t = layer_input[:,seq,:]
            
            X_t =  X_t.to(self.device)
            
            #print('X {}'.format(X_t.device))
            for idx in range(self.n_layers):
                model_params = self.layer_params[idx]
                hidden_i = layer_states[idx]
                
                #print('h {}'.format(hidden_i.device))
                # Extracting parameters
                W_xz = model_params[0]
                W_hz = model_params[1]
                W_xr = model_params[2]
                W_hr = model_params[3]
                W_xg = model_params[4]
                W_hg = model_params[5]
                drop_l = model_params[6]
                
                # Computing the gates and next state
                Z_t = torch.sigmoid( W_xz(X_t) + W_hz(hidden_i))
                r_t = torch.sigmoid( W_xr(X_t) + W_hr(hidden_i))
                g_t = torch.tanh(W_xg(X_t) + W_hg((r_t*hidden_i)))
                
                new_hidden = Z_t*hidden_i + (1-Z_t)*g_t
                
                # New hidden state for next layer 
                layer_states[idx] = drop_l(new_hidden)
                
                X_t = new_hidden
                
            out_list.append(self.W_y(X_t))
                
        layer_output = torch.stack(out_list,dim=1)
        hidden_state = torch.stack(layer_states,dim=1)
        # ========================
        return layer_output, hidden_state
