import torch.nn as nn
import torch

class RNN_Model(nn.Module):
    def __init__(self, type, embedding_dim, num_of_classes, num_of_layers, hidden_size, dropout=0, bidirectional=False):
        super(RNN_Model, self).__init__()

        self.type = type

        if type == 'lstm':
            self.RNN = nn.LSTM(embedding_dim, hidden_size, num_of_layers, dropout=dropout, bidirectional=bidirectional)
        else:
            self.RNN = nn.GRU(embedding_dim, hidden_size, num_of_layers, dropout=dropout, bidirectional=bidirectional)


        if bidirectional:
            self.classifier = nn.Sequential(nn.Linear(2*hidden_size, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, num_of_classes))
        else:
            self.classifier = nn.Sequential(nn.Linear(hidden_size, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, num_of_classes))


    def forward(self, x):
        if self.type == 'lstm':
            output, (_, __) = self.RNN(x)
        else: 
            output, _ = self.RNN(x)
        return self.classifier(output[-1, :, :])