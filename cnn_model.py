import torchvision
import torch.nn as nn
import torch
import numpy as np

class CNN_Model(nn.Module):
    def __init__(self, model_name, embedding_dim, isFullNet = False, num_of_class = 3, pretrained = True, train_weights =  1, dropout = 0):
        super(CNN_Model, self).__init__()

        self.isFullNet = isFullNet
        self.num_of_class = num_of_class
        self.dropout = dropout

        if model_name == 'densenet':
            cnn_model = torchvision.models.densenet121(
                pretrained=pretrained, progress=True, drop_rate=0.3, memory_efficient=False)
            num_features = cnn_model.classifier.in_features
        
        elif model_name == 'resnet':
            cnn_model = torchvision.models.resnet18(pretrained=pretrained, progress=True)
            num_features = cnn_model.fc.in_features

        elif model_name == 'resnext':
            cnn_model = torchvision.models.resnext50_32x4d(pretrained=pretrained, progress=True)
            num_features = cnn_model.fc.in_features

        elif model_name == 'mnasnet':
            cnn_model = torchvision.models.mnasnet1_0(pretrained=pretrained, progress=True)
            num_features = 1280

        elif model_name == 'squeezenet':
            cnn_model = torchvision.models.squeezenet1_1(
                pretrained=pretrained, progress=True)
            num_features = 512

        elif model_name == 'alexnet':
            cnn_model = torchvision.models.alexnet(pretrained=pretrained)
            num_features = 256

        elif model_name == 'vgg':
            cnn_model = torchvision.models.vgg11(pretrained=pretrained)
            num_features = 512

        elif model_name == 'mobilenet':
            cnn_model = torchvision.models.mobilenet_v2(pretrained=pretrained)
            num_features = 1280

        elif model_name == 'shufflenet':
            cnn_model = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained)
            num_features = 1024

        elif model_name == 'basic' :
            cnn_model = nn.Sequential( nn.Conv2d(3, 6, 3),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(6),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(6, 16, 3),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(16),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(16, 32, 3),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(32, 64, 3),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64),
                                    nn.MaxPool2d(2, 2))

            num_features = 64

        self.embedding_dim = embedding_dim

        self.model = nn.Sequential(*list(cnn_model.children())[:-1])

        num_of_layers = len(list(self.model.children()))
        for layer_num in range(0, num_of_layers, -1):
            if num_of_layers - layer_num < num_of_layers * (1-train_weights):
                for param in list(self.model.children())[layer_num].parameters():
                    param.requires_grad = False

        self.fc1 = nn.Sequential(nn.Linear(int(num_features), int(num_features/2)),
                                 nn.Dropout(dropout),
                                 nn.ReLU(),
                                 nn.Linear(int(num_features/2), int(num_features/4)),
                                 nn.Dropout(dropout),
                                 nn.ReLU())

        if self.isFullNet:
            self.fc2 = nn.Linear(
                int(num_features/4), self.num_of_class)
        else:
            self.fc2 = nn.Linear(
                int(num_features/4), embedding_dim)

        

        self.pooling = nn.AdaptiveMaxPool2d((1,1))

    def forward(self, x):
        
        
        if self.isFullNet:
            output = torch.zeros(  [x.size(0), self.num_of_class])
            x_temp = self.pooling(self.model(x))
            x_temp = x_temp.view(x_temp.size(0), -1)
            output = self.fc2(self.fc1(x_temp))
        else:
            output = torch.zeros( [x.size(1), x.size(0), self.embedding_dim])
            for i in range(x.size(1)):
                x_temp = self.pooling(self.model(x[:, i, :, :, :]))
                x_temp = x_temp.view(x_temp.size(0), -1)
                output[i] = self.fc2(self.fc1(x_temp))

        return output