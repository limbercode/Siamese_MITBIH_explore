import torch
import sys
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import h5py as hp
import numpy as np
from torch.utils import data
import time
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
cuda = torch.cuda.is_available()


class IntermediateLayerGetter(nn.ModuleDict):
    """ get the output of certain layers """

    def __init__(self, model, return_layers):
        # 判断传入的return_layers是否存在于model中
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}  # 构造dict
        layers = OrderedDict()
        # 将要从model中获取信息的最后一层之前的模块全部复制下来
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)  # 将所需的网络层通过继承的方式保存下来
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 将所需的值以k,v的形式保存到out中
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


def load_mat(path_data,name_data,dtype='float32'):
    data=hp.File(path_data)
    arrays_d={}
    for k,v in data.items():
        arrays_d[k]=np.array(v)
    dataArr=np.array(arrays_d[name_data],dtype=dtype)
    return dataArr



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


if __name__ == '__main__':
    ECG_class = ['N', 'V', 'L', 'R']
    n_classes=4
    Path = './'  # 自定义路径要正确
    DataFile = 'Data_CNN.mat'
    LabelFile = 'Label_OneHot.mat'
    print("Loading data and labels...")
    Data = load_mat(Path + DataFile, 'Data')
    Label = load_mat(Path + LabelFile, 'Label')
    Data = Data.T
    Indices = np.arange(Data.shape[0])  # 随机打乱索引并切分训练集与测试集
    np.random.shuffle(Indices)

    # print("Divide training and testing set...")
    train_x = Data[Indices[:512]]
    train_y = Label[Indices[:512]]
    test_x = Data[Indices[15000:]]
    test_y = Label[Indices[15000:]]
    train_dataset = my_data(train_x, train_y, 1)
    test_dataset = my_data(test_x, test_y, 0)
    batch_size = 128
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
    classfi_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    classfi_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    # Set up the network and training parameters
    from networks import EmbeddingNet, ClassificationNet
    embedding_net = EmbeddingNet()
    model = ClassificationNet(embedding_net,n_classes)
    if cuda:
        model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    epochs = 30
    log_interval = 100
    total_train_step=0;
    start_time=time.time()

    for i in range(epochs):
        for batch_index, (data, label) in enumerate(classfi_train_loader):
            label=label.cuda()
            data=data.cuda()
            output =model(data)
            loss=loss_fn.forward(output,label).cuda()
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            total_train_step+=1
            # if total_train_step % 40 == 0:
            #     end_time = time.time()  # 训练结束时间
            #     print("训练时间: {}".format(end_time - start_time))
            #     print("训练次数: {}, train_Loss: {}".format(total_train_step, loss))

    return_layer={'conv2':'feature'}
    backbone=IntermediateLayerGetter(model,return_layer)
    backbone.eval();
    output=[];
    for batch_index, (data, label) in enumerate(classfi_train_loader):
        data=data.cuda()
        out=backbone(data)
        for i in range(batch_size):
            output.append(out['feature'][i])
    print(output)


