from torchvision import models
import os
import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn_1 = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.rnn_2 = nn.LSTM(hidden_dim, 
                          hidden_dim, 
                          num_layers=n_layers, 
                          bidirectional=bidirectional, 
                          dropout=dropout)
        self.rnn_3 = nn.LSTM(hidden_dim, 
                          hidden_dim, 
                          num_layers=n_layers, 
                          bidirectional=bidirectional, 
                          dropout=dropout)
                        
        
        # self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        #pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), enforce_sorted=False)
        
        packed_output, _ = self.rnn_1(packed_embedded)
        packed_output, _ = self.rnn_2(packed_output)
        packed_output, (hidden, cell) = self.rnn_3(packed_output)
        
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        # hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim * num directions]
            
        return self.fc(hidden)

    # returns the word embedding given text
    # this function is only needed for evaluation of attribution method
    def forward_embedding_only(self, text):
        # text = [sent len, batch size]

        embedded = self.embedding(text)

        return embedded

    # returns logit given word embedding
    # this function is only needed for evaluation of attribution method
    def forward_no_embedding(self, embedding, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(embedding)

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), enforce_sorted=False)

        packed_output, _ = self.rnn_1(packed_embedded)
        packed_output, _ = self.rnn_2(packed_output)
        packed_output, (hidden, cell) = self.rnn_3(packed_output)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        # hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)


def build_classifiers(cfg):
    #TODO remove print
    print("Load multi-layer LSTM")
    INPUT_DIM = 25002
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 1
    BIDIRECTIONAL = False
    DROPOUT = 0.5
    PAD_IDX = 1

    model = RNN(INPUT_DIM, 
                EMBEDDING_DIM, 
                HIDDEN_DIM, 
                OUTPUT_DIM, 
                N_LAYERS, 
                BIDIRECTIONAL, 
                DROPOUT, 
                PAD_IDX)

    # select a model to analyse
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), '../rnn-model.pt')))
    return model


def get_module(model, module):
    r"""Returns a specific layer in a model based.
    Shameless copy from `TorchRay
    <https://github.com/facebookresearch/TorchRay/blob/master/torchray/attribution/common.py>`_

    :attr:`module` is either the name of a module (as given by the
    :func:`named_modules` function for :class:`torch.nn.Module` objects) or
    a :class:`torch.nn.Module` object. If :attr:`module` is a
    :class:`torch.nn.Module` object, then :attr:`module` is returned unchanged.
    If :attr:`module` is a str, the function searches for a module with the
    name :attr:`module` and returns a :class:`torch.nn.Module` if found;
    otherwise, ``None`` is returned.
    Args:
        model (:class:`torch.nn.Module`): model in which to search for layer.
        module (str or :class:`torch.nn.Module`): name of layer (str) or the
            layer itself (:class:`torch.nn.Module`).
    Returns:
        :class:`torch.nn.Module`: specific PyTorch layer (``None`` if the layer
            isn't found).
    """
    if isinstance(module, torch.nn.Module):
        return module

    assert isinstance(module, str)
    if module == '':
        return model

    for name, curr_module in model.named_modules():
        if name == module:
            return curr_module

    return None
