import torch
from . import basic_model, register_model
from .. import modules
import torch.nn as nn
if True:
    try:
        from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
    except:
        from torch.nn import LayerNorm
else:
    from torch.nn import LayerNorm
from torch.utils.model_zoo import load_url as load_state_dict_from_url

class SConv1d(nn.Module):
    def __init__(self, 
                 template_shape,
                 in_dim, 
                 out_dim, 
                 kernel_size, 
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 initializer = nn.init.xavier_uniform_,
                ):
        super().__init__()
        assert kernel_size % 2 == 1, 'Only support odd kernel size.'
        self.k_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.conv = nn.Conv2d(template_shape[1], self.k_size, (template_shape[2] - in_dim + 1, template_shape[3] - out_dim + 1))
        initializer(self.conv.weight)
        self.bias = nn.Parameter(initializer(torch.FloatTensor(1, out_dim)).squeeze()) if bias else None
    def forward(self, vec, template):
        weight = self.conv(template).squeeze(0).permute(2, 1, 0)
        vec = vec.transpose(1, 2)
        vec = torch.nn.functional.conv1d(vec, weight, self.bias, stride = self.stride, padding = self.padding)
        vec = vec.transpose(1, 2)
        return vec

class Trans(nn.Module):
    def forward(self, vec, template = None):
        return vec.transpose(1, 2)

class TContainer(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.net = nn.ModuleList(models)
    def forward(self, vec, template):
        for l in self.net:
            vec = l(vec, template)
        return vec

class TLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = LayerNorm(dim)
    def forward(self, vec, template = None):
        return self.net(vec)

class TBatchNorm1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.BatchNorm1d(dim)
    def forward(self, vec, template = None):
        vec = vec.transpose(1, 2)
        vec = self.net(vec)
        vec = vec.transpose(1, 2)
        return vec

class TReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.ReLU(inplace = True)
    def forward(self, vec, template = None):
        return self.net(vec)

class TMaxPool1d(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.net = nn.MaxPool1d(kernel_size, stride, padding)
    def forward(self, vec, template):
        vec = vec.transpose(1, 2)
        vec = self.net(vec)
        vec = vec.transpose(1, 2)
        return vec

class TAdaptiveAvgPool1d(nn.Module):
    def __init__(self, par):
        super().__init__()
        self.net = nn.AdaptiveAvgPool1d(par)
    def forward(self, vec, template):
        vec = vec.transpose(1, 2)
        vec = self.net(vec)
        vec = vec.transpose(1, 2)
        return vec

@register_model('sresnet18')
class mresnet18(nn.Module):
    @classmethod
    def setup_model(cls):
        return resnet18

def conv3x3(template_shape, in_planes, out_planes, stride=1, groups=1, dilation=1):
    return SConv1d(template_shape, in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(template_shape, in_planes, out_planes, stride=1):
    return SConv1d(template_shape, in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, template_shape, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = LayerNorm
        if groups != 1 or base_width != 64:
            raise ValueError(f'BasicBlock only supports groups=1 and base_width=64, got groups = {groups} and base_width = {base_width}')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(template_shape, inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = TReLU()
        self.conv2 = conv3x3(template_shape, planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, template):
        identity = x # -
        out = self.conv1(x, template) # -
        out = self.bn1(out, template)
        out = self.relu(out, template)
        # +
        out = self.conv2(out, template)
        out = self.bn2(out, template)
        if self.downsample is not None:
            identity = self.downsample(x, template)

        out += identity
        out = self.relu(out, template)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, template_shape, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = LayerNorm# nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(template_shape, inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(template_shape, width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(template_shape, width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, template):
        identity = x

        out = self.conv1(x, template)
        out = self.bn1(out, template)
        out = self.relu(out, template)

        out = self.conv2(out, template)
        out = self.bn2(out, template)
        out = self.relu(out, template)

        out = self.conv3(out, template)
        out = self.bn3(out, template)

        if self.downsample is not None:
            identity = self.downsample(x, template)

        out += identity
        out = self.relu(out, template)

        return out

class ResNet(nn.Module):
    def __init__(self,
                 template_shape,
                 block, # Bottleneck
                 layers, # [2, 2, 2, 2]
                 num_classes = 1000, 
                 zero_init_residual = False,
                 groups = 1, 
                 width_per_group = 64, 
                 replace_stride_with_dilation = None, # None
                 norm_layer = None
                ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = TLayerNorm #BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups # 1
        self.base_width = width_per_group # 64
        self.conv1 = SConv1d(template_shape, 300, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = TReLU()
        self.maxpool = TMaxPool1d(kernel_size=3, stride=2, padding=1)

        # self.layer1 = self._make_layer(template_shape, block, 300, layers[0]) # [2, 2, 2, 2]
        # self.layer2 = self._make_layer(template_shape, block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(template_shape, block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(template_shape, block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = TAdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64 * block.expansion, 300)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual: # false
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, template_shape, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation # 1
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion: # self.inplanes = 64, block.expansion = 1
            downsample = TContainer(
                [conv1x1(template_shape, self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
                ]
            )

        layers = []
        layers.append(block(template_shape, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(template_shape, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return TContainer(layers)

    def forward(self, batch, embedding, extra_input, template, writer):
        source, wizard, target = batch
        if isinstance(embedding, nn.ModuleDict):
            x = embedding['emb_src'](source.long())
            x = embedding['emb_tgt'](wizard.long())
            x = torch.cat([wiz, vec], dim = 1)
        else:
            x = embedding(source.long())
        
        x = self.conv1(x, template) # channel 3-> 64
        # x = self.bn1(x, template)
        # x = self.relu(x, template)
        # x = self.maxpool(x, template)
        
        # x = self.layer1(x, template) # 64 -> 64
        # x = self.layer2(x, template) # 64 -> 128
        # x = self.layer3(x, template) # 128 -> 256
        # x = self.layer4(x, template) # 256 -> 512
        x = self.avgpool(x, template)
        # x = xreshape(x.size(0), -1)
        x = self.fc(x).squeeze()

        return x, None

def _resnet(arch, inplanes, planes, model_config, template):
    model = ResNet(template.shape, inplanes, planes)
    return model

def resnet18(model_config, template):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], model_config, template)

def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], model_config, template)

def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], model_config, template)

def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], model_config, template)

def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], model_config, template)

def resnext50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], model_config, template)


def resnext101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], model_config, template)












