import torch
import torch.nn as nn

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(64*6*6, 600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(600, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x


class FashionCNN_Modular(nn.Module):
    def __init__(self, 
                 num_filters_layer1, 
                 num_filters_layer2, 
                 kernel_size_layer1, 
                 kernel_size_layer2,
                 fc1_units, 
                 dropout_rate, 
                 activation_function):
        super(FashionCNN_Modular, self).__init__()
        
        activation_fn = {'ReLU': nn.ReLU(), 'LeakyReLU': nn.LeakyReLU(), 'ELU': nn.ELU()}[activation_function]

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, num_filters_layer1, kernel_size=kernel_size_layer1, padding=1),
            nn.BatchNorm2d(num_filters_layer1),
            activation_fn,
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_filters_layer1, num_filters_layer2, kernel_size=kernel_size_layer2, padding=1),
            nn.BatchNorm2d(num_filters_layer2),
            activation_fn,
            nn.MaxPool2d(2)
        )
        
        # Calculate the output size of the second convolutional layer
        self.feature_size = self._get_conv_output((1, 28, 28))
        
        self.fc1 = nn.Linear(self.feature_size, fc1_units)
        self.drop = nn.Dropout2d(dropout_rate)
        self.fc2 = nn.Linear(fc1_units, 10)

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
