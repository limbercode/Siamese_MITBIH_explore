from torch import relu
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv1d(125, 32, 131), nn.PReLU(),
                                     nn.MaxPool1d(5,5),
                                     nn.Conv1d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool1d(2,2))

        self.fc = nn.Sequential(nn.Linear(64 * 10, 50),
                                nn.PReLU()
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(-1,64*10)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
