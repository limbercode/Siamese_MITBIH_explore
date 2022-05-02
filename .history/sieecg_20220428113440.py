import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import h5py as hp
from trainer import fit
import numpy as np
from torch.utils import data 
cuda = torch.cuda.is_available()

ECG_class=['N','V','L','R']

def load_mat(path_data,name_data,dtype='float32'):
    data=hp.File(path_data)
    arrays_d={}
    for k,v in data.items():
        arrays_d[k]=np.array(v)
    dataArr=np.array(arrays_d[name_data],dtype=dtype)
    return dataArr

Path='./' #自定义路径要正确
DataFile='Data_CNN.mat'
LabelFile='Label_OneHot.mat'
#print("Loading data and labels...")
Data=load_mat(Path+DataFile,'Data')
Label=load_mat(Path+LabelFile,'Label')
Data=Data.T
Indices=np.arange(Data.shape[0]) #随机打乱索引并切分训练集与测试集
np.random.shuffle(Indices)

#print("Divide training and testing set...")
train_x=Data[Indices[:500]]
train_y=Label[Indices[:500]]
test_x=Data[Indices[15000:]]
test_y=Label[Indices[15000:]]

class my_data(data.Dataset):
    def __init__(self, x, y,train, transform=None):
        self.xsignals =torch.from_numpy(x)
        self.ylabels = torch.from_numpy(y).max(dim=1)[1]
        self.train=train
        if transform is not None:   
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):   #必须加载的方法
        signal = self.xsignals[index]
        Label = self.ylabels[index]
        return signal, Label   #返回处理完的图片数据和标签

    def __len__(self):     #必须加载的方法,实际上好像没什么用
        return len(self.ylabels)

train_dataset=my_data(train_x,train_y,1)
test_dataset=my_data(test_x,test_y,0)
class SiameseECG(data.Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, ecg_dataset):
        self.ecg_dataset = ecg_dataset

        self.train = self.ecg_dataset.train
        self.transform = self.ecg_dataset.transform

        if self.train:
            self.train_labels = self.ecg_dataset.ylabels
            self.train_data = self.ecg_dataset.xsignals
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.ecg_dataset.ylabels
            self.test_data = self.ecg_dataset.xsignals
            self.labels_set = list(set(self.test_labels))
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}
            print(self.test_labels)
            print(self.label_to_indices)
            print(self.label_to_indices[self.test_labels])
            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i]]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]


        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.ecg_dataset)

siamese_train_dataset = SiameseECG(train_dataset) # Returns pairs of images and target same/different
#siamese_test_dataset = SiameseECG(test_dataset)
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
#siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet, SiameseNet
from loss import ContrastiveLoss

margin = 1.
embedding_net = EmbeddingNet()
model = SiameseNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
epochs = 1
log_interval = 100

for (x, y),step in enumerate(siamese_train_loader):
    print(step,(x,y))