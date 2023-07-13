import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(self._make_layer(in_channels, growth_rate))
            in_channels += growth_rate

    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        return layer

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = torch.cat([out, layer(out)], dim=1)
        return out

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

class DenseNet(nn.Module):
    def __init__(self, growth_rate, block_config, num_classes=1000):
        super(DenseNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dense1 = self._make_dense_block(growth_rate, block_config[0])
        self.trans1 = self._make_transition_block(growth_rate * block_config[0], growth_rate * block_config[0] // 2)

        self.dense2 = self._make_dense_block(growth_rate, block_config[1])
        self.trans2 = self._make_transition_block(growth_rate * block_config[1], growth_rate * block_config[1] // 2)

        self.dense3 = self._make_dense_block(growth_rate, block_config[2])
        self.trans3 = self._make_transition_block(growth_rate * block_config[2], growth_rate * block_config[2] // 2)

        self.dense4 = self._make_dense_block(growth_rate, block_config[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(growth_rate * block_config[3], num_classes)

    def _make_dense_block(self, growth_rate, num_layers):
        dense_block = DenseBlock(self.in_channels, growth_rate, num_layers)
        self.in_channels += growth_rate * num_layers
        return dense_block

    def _make_transition_block(self, in_channels, out_channels):
        transition_block = TransitionBlock(in_channels, out_channels)
        self.in_channels = out_channels
        return transition_block

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dense1(x)
        x = self.trans1(x)

        x = self.dense2(x)
        x = self.trans2(x)

        x = self.dense3(x)
        x = self.trans3(x)

        x = self.dense4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def DenseNet24(num_classes=1000):
    growth_rate = 32
    block_config = [6, 12, 24, 16] 

    def _make_dense_block(in_channels, growth_rate, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
            in_channels += growth_rate
        return nn.Sequential(*layers)

    def _make_transition_block(in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    num_channels = 64
    for i, num_layers in enumerate(block_config):
        dense_block = _make_dense_block(num_channels, growth_rate, num_layers)
        model.add_module(f"DenseBlock{i+1}", dense_block)
        num_channels += num_layers * growth_rate
        if i != len(block_config) - 1:
            transition_block = _make_transition_block(num_channels, num_channels // 2)
            model.add_module(f"TransitionBlock{i+1}", transition_block)
            num_channels = num_channels // 2

    # Classification Layer
    model.add_module("AdaptiveAvgPool2d", nn.AdaptiveAvgPool2d((1, 1)))
    model.add_module("Flatten", nn.Flatten())
    model.add_module("Linear", nn.Linear(num_channels, num_classes))

    return model

