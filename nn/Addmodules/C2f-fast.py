import torch
import torch.nn as nn

__all__ = ['C2f-fast']


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    
class conv_block(nn.Module):  
    def __init__(self,  
                 in_features,  
                 out_features,  
                 kernel_size=(3, 3),  
                 stride=(1, 1),  
                 padding=(1, 1),  
                 dilation=(1, 1),  
                 norm_type='bn',  
                 activation=True,  
                 use_bias=True,  
                 groups=1):  
        super().__init__()  
        self.conv = nn.Conv2d(in_channels=in_features,  
                              out_channels=out_features,  
                              kernel_size=kernel_size,  
                              stride=stride,  
                              padding=padding,  
                              dilation=dilation,  
                              bias=use_bias,  
                              groups=groups)  
  
        self.norm_type = norm_type  
        self.act = activation  
  
        if self.norm_type == 'gn':  
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)  
        elif self.norm_type == 'bn':  
            self.norm = nn.BatchNorm2d(out_features)  
        if self.act:  
            self.relu = nn.ReLU(inplace=False)  
  
    def forward(self, x):  
        x = self.conv(x)  
        if self.norm_type is not None:  
            x = self.norm(x)  
        if self.act:  
            x = self.relu(x)  
        return x 

class MDC(nn.Module):
    def __init__(self, in_features, out_features, norm_type='bn', activation=True, rate=[1, 2]):
        super().__init__()

        self.block1 = conv_block(
            in_features=in_features//2,
            out_features=out_features//2,
            padding=rate[0],
            dilation=rate[0],
            norm_type=norm_type,
            activation=activation,
            groups=in_features // 2
            )
        self.block2 = conv_block(
            in_features=in_features//2,
            out_features=out_features//2,
            padding=rate[1],
            dilation=rate[1],
            norm_type=norm_type,
            activation=activation,
            groups=in_features // 2
            )
        self.out_s = conv_block(
            in_features=2,
            out_features=2,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
        )
        self.out = conv_block(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
            )

    def forward(self, x):
        split_tensors = []
        x = torch.chunk(x, 2, dim=1)
        x1 = self.block1(x[0])
        x2 = self.block2(x[1])
        for channel in range(x1.size(1)):
            channel_tensors = [tensor[:, channel:channel + 1, :, :] for tensor in [x1, x2]]
            concatenated_channel = self.out_s(torch.cat(channel_tensors, dim=1))  # 拼接在 batch_size 维度上
            split_tensors.append(concatenated_channel)
        x = torch.cat(split_tensors, dim=1)
        x = self.out(x)
        return x
    
class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div                   #进行卷积操作的部分
        self.dim_untouched = dim - self.dim_conv3         #保持不变的部分
        self.partial_conv3 = MDC(in_features=self.dim_conv3, out_features=self.dim_conv3)       
        # self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class C2f-fast_Bottleneck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.DualPConv = nn.Sequential(Partial_conv3(dim, n_div=4, forward='split_cat'),
                                       Partial_conv3(dim, n_div=4, forward='split_cat'))

    def forward(self, x):
        return self.DualPConv(x)

class C2f-fast(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(CSPPC_Bottleneck(self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c),1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 224, 224)
    image = torch.rand(*image_size)

    # Model
    model = C2f-fast(64, 128)
    print(model)
    out = model(image)
    print(out.size())