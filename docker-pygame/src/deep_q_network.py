import paddle.nn as nn

class DeepQNetwork(nn.Layer):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x
