from torch import relu
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv1d(128, 32, 131), nn.PReLU(),
                                     nn.MaxPool1d(5,5),
                                     nn.Conv1d(32,128 , 5), nn.PReLU(),
                                     nn.MaxPool1d(2,2))

        self.fc = nn.Sequential(nn.Linear(10, 50),
                                nn.PReLU()
                                )

    def forward(self, x):
        output = self.convnet(x)

        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)

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



class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.conv2= nn.Conv1d(128, 128, 32)
        self.fc2 = nn.Linear(50, n_classes)
        self.fc1 = nn.Linear(19,50)
    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        output = self.conv2(output)
        output = self.nonlinear(output)
        output = self.fc1(output)
        output = self.fc2(output)
        scores = F.log_softmax(output, dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))