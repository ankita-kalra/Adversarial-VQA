import torch
import torch.nn as nn
from utils import to_variable


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, dropout=0.65):
        if not self.training:
            return inp
        tensor_mask = inp.data.new(1, inp.size(1), inp.size(2)).bernoulli_(1 - dropout)
        var_mask = torch.autograd.Variable(tensor_mask, requires_grad=False) / (1 - dropout)
        var_mask = var_mask.expand_as(inp)
        return var_mask * inp


class LM_LSTM(nn.Module):
    def __init__(self, hidden_size, embedding_dim, output_size, n_layers=1, is_gru=0):
        super(LM_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.is_gru = is_gru
        self.drop_out = nn.Dropout(0.4)
        self.locked_dropout = LockedDropout()
        self.encoder = nn.Embedding(output_size, embedding_dim)
        if self.is_gru == 2:
            self.rnns = nn.ModuleList([
                nn.RNN(embedding_dim, hidden_size=self.hidden_size),
                nn.RNN(hidden_size, hidden_size=self.hidden_size),
                nn.RNN(hidden_size, hidden_size=self.embedding_dim)
            ])
        else:
            self.rnns = nn.ModuleList([
                nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_size) if is_gru == 0 else nn.GRU(input_size=embedding_dim, hidden_size=self.hidden_size),
                nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size) if is_gru == 0 else nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size),
                nn.LSTM(input_size=self.hidden_size, hidden_size=embedding_dim) if is_gru == 0 else nn.GRU(input_size=self.hidden_size, hidden_size=embedding_dim)
                ])
        self.decoder = nn.Linear(hidden_size, output_size)
        # tying weights
        self.decoder.weight = self.encoder.weight
        self.init_weights()

    def init_hidden(self, batch_size):
        return [(to_variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                 to_variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))) if self.is_gru == 0 else to_variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                (to_variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                 to_variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))) if self.is_gru == 0 else to_variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                (to_variable(torch.zeros(self.n_layers, batch_size, self.embedding_dim)),
                 to_variable(torch.zeros(self.n_layers, batch_size, self.embedding_dim))) if self.is_gru == 0 else to_variable(torch.zeros(self.n_layers, batch_size, self.embedding_dim))]

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, input, hidden):
        encoded = self.encoder(input)
        # Applying locked dropout as mentioned in paper
        encoded = self.locked_dropout(encoded, 0.65)
        h = encoded
        new_hidden = []
        for i, rnn in enumerate(self.rnns):
            h, state = rnn(h, hidden[i])
            new_hidden.append(state)
            if i != 2:
                h = self.locked_dropout(h, 0.3)

        output = self.locked_dropout(h, 0.4)
        output = self.decoder(output.view(-1, self.embedding_dim))
        return output, new_hidden
