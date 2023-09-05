
from basic_net.dcn import DeformableConv2d
from basic_net.nls import NlsConv2d

def init_conv(name):
    if name == "deform":
        return DeformableConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
    elif name == "conv":
        return nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
    elif name == "stnls":
        return NlsConv2d(32,32,ps=3,stride=1)

class MNISTClassifier(nn.Module):
    def __init__(self,method_name):

        super(MNISTClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = init_conv(method_name)
        self.conv5 = init_conv(method_name)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x) # [14, 14]
        x = torch.relu(self.conv2(x))
        x = self.pool(x) # [7, 7]
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.gap(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x
