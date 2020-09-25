from torch import nn

# SE模块首先对卷积得到的特征图进行Squeeze操作，得到channel级的全局特征，然后对全局特征进行Excitation操作，学习各个channel间的关系，
# 也得到不同channel的权重，最后乘以原来的特征图得到最终特征。本质上，SE模块是在channel维度上做attention或者gating操作，
# 这种注意力机制让模型可以更加关注信息量最大的channel特征，而抑制那些不重要的channel特征。
# 另外一点是SE模块是通用的，这意味着其可以嵌入到现有的网络架构中。
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, int(channel/reduction), bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(int(channel/reduction), channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)