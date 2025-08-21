import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, conv_layers_specs, fc_layers_specs):
        super(Net, self).__init__()
        self.conv_layers = nn.ParameterList()
        self.fc_layers = nn.ParameterList()
        for c_specs in conv_layers_specs:
            self.conv_layers.append(nn.Conv2d(c_specs[0], c_specs[1], c_specs[2]))
        for fc_specs in fc_layers_specs:
            self.fc_layers.append(nn.Linear(fc_specs[0], fc_specs[1]))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = self.pool(F.relu(conv_layer(x)))
        x = torch.flatten(x, 1)
        for fc_layer in self.fc_layers[:-1]:
            x = F.relu(fc_layer(x))
        x = self.fc_layers[-1](x)
        return x
