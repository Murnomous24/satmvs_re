import torch
import torch.nn as nn
import torch.nn.functional as F

# build depth regression from probability cost volume
def depth_regression(prob, depth_values):
    # prob: [B, Ndepth, H, W]
    # depth_values; [B, Ndepth] or [B, Ndepth, H, W] if sub-pixel

    if depth_values.dim() <= 2: # [B, Ndepth], all pixel same depth range
        depth_values = depth_values.view(*depth_values.shape, 1, 1) # [B, Ndepth, 1, 1]
    else: # [B, Ndepth, H, W], each pixel has different depth range
        depth_values = F.interpolate(
            depth_values,
            [prob.shape[2], prob.shape[3]],
            mode = 'bilinear',
            align_corners = False
        ) # upscale the depth_values, to match the shape of prob
    
    # soft argmin depth estimation 
    depth = torch.sum(prob * depth_values, 1) 
    return depth

# TODO: initlize the network weight
def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)

# TODO: initlize the batch normal weight
def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return 

class Conv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride = 1,
            relu = True,
            bn = True,
            bn_momentum = 0.1, # what
            init_method = "xavier",
            **kwargs
        ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride = stride,
            bias = (not bn),
            **kwargs
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum = bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace = True)
        return x

    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride = 1,
            relu = True,
            bn = True,
            bn_momentum = 0.1, # what
            init_method = "xavier",
            **kwargs
        ):
        super().__init__()
        
        self.out_channels = out_channels
        assert stride in [1, 2] # TODO: why this assert
        self.stride = stride
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            bias = (not bn),
            **kwargs
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum = bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace = True)
        return x

    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class DeConv2dFuse(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            relu = True,
            bn = True,
            bn_momentum = 0.1
    ):
        super().__init__()

        self.deconv = Deconv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride = 2,
            padding = 1,
            out_padding = 1,
            bn = True,
            relu = relu,
            bn_momentum = bn_momentum
        )
        self.conv = Conv2d(
            2 * out_channels,
            out_channels,
            kernel_size,
            stride = 1,
            padding = 1,
            out_padding = 1,
            bn = True,
            relu = relu,
            bn_momentum = bn_momentum
        )
    
    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim = 1)
        x = self.conv(x)
        return x

# extraction multi stage features
class FeatureNet(nn.Module):
    def __init__(
            self,
            base_channels,
            num_stage = 3,
            stride = 4,
            arch_mode = "unet"
    ):
        super().__init__()
        
        assert arch_mode in ["unet", "fpn"], f"mode must be 'unet' or 'fpn', but get {arch_mode}"
        
        self.arch_mode = arch_mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage

        # network
        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding = 1),
            Conv2d(base_channels, base_channels, 3, 1, padding = 1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride = 2, padding = 2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding = 1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding = 1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride = 2, padding = 2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding = 1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding = 1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias = False)
        self.out_channels = [4 * base_channels]

        if self.arch_mode == 'unet':
            if num_stage == 3:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
                self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias = False)
                self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias = False)
                self.out_channels.append(2 * base_channels)
                self.out_channels.append(base_channels)
            
            elif num_stage == 2:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias = False)
                self.out_channels.append(2 * base_channels)
        elif self.arch_mode == 'fpn':
            final_chs = base_channels * 4
            
            if num_stage == 3:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias = True)
                self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias = True)

                self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding = 1, bias = False)
                self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding = 1, bias = False)

                self.out_channels.append(base_channels * 2)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias = True)

                self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding = 1, bias = False)
                self.out_channels.append(base_channels)

    def forward(self, x):
        # print("featurenet forward 1: start forward")
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        # print("featurenet forward 2: conv ok")
        intra_feat = conv2
        outputs = {}

        out = self.out1(intra_feat)
        outputs["stage1"] = out
        # print("featurenet forward 2: out1 ok")

        if self.arch_mode == 'unet':
            if self.num_stage == 3:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = self.deconv2(conv0, intra_feat)
                out = self.out3(intra_feat)
                outputs["stage3"] = out
            elif self.num_stage == 2:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out
        elif self.arch_mode == 'fpn':
            if self.num_stage == 3:
                # print("featurenet forward 3: fpn num_stgae 3 start")
                # print(f"featurenet forward 3: conv0: {conv0.shape}, conv1: {conv1.shape}, conv2: {conv2.shape}")
                intra_feat = F.interpolate(intra_feat, scale_factor = 2, mode = "nearest") + self.inner1(conv1)
                # print("featurenet forward 3: fpn num_stgae 3 inner1 ok")
                out = self.out2(intra_feat)
                outputs["stage2"] = out
                # print("featurenet forward 3: fpn num_stgae 3 out2 ok")

                intra_feat = F.interpolate(intra_feat, scale_factor = 2, mode = "nearest") + self.inner2(conv0)
                # print("featurenet forward 3: fpn num_stgae 3 inner2 ok")
                out = self.out3(intra_feat)
                outputs["stage3"] = out
                # print("featurenet forward 3: fpn num_stgae 3 out3 ok")
            elif self.num_stage == 2:
                # print("featurenet forward 3: fpn num_stgae 2 start")
                intra_feat = F.interpolate(intra_feat, scale_factor = 2, mode = "nearest") + self.inner1(conv1)
                # print("featurenet forward 3: fpn num_stgae 2 inner1 ok")
                out = self.out2(intra_feat)
                outputs["stage2"] = out
                # print("featurenet forward 3: fpn num_stgae 3 out2 ok")
        
        return outputs

class Conv3d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size = 3,
            stride = 1,
            relu = True,
            bn = True,
            bn_momentum = 0.1,
            init_method = "xavier",
            **kwargs
    ):
        super().__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2] # TODO: why this assert
        self.stride = stride

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride = stride,
            bias = (not bn),
            **kwargs
        )
        self.bn = nn.BatchNorm3d(
            out_channels,
            momentum = bn_momentum
        ) if bn else None
        self.relu = relu
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace = True)
        return x
    
    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv3d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size = 3,
            stride = 1,
            relu = True,
            bn = True,
            bn_momentum = 0.1,
            init_method = "xavier",
            **kwargs
    ):
        super().__init__()

        self.out_channels = out_channels
        assert stride in [1, 2] # TODO: why this assert
        self.stride = stride

        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride = stride,
            bias = (not bn),
            **kwargs
        )
        self.bn = nn.BatchNorm3d(
            out_channels,
            momentum = bn_momentum
        ) if bn else None
        self.relu = relu
    
    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace = True)
        return x
    
    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

# cost volume regularization
class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()

        self.conv0 = Conv3d(in_channels, base_channels, padding = 1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride = 2, padding = 1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding = 1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride = 2, padding = 1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding = 1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride = 2, padding = 1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding = 1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride = 2, padding = 1, output_padding = 1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride = 2, padding = 1, output_padding = 1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels, stride = 2, padding = 1, output_padding = 1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride = 1, padding = 1, bias = False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))

        up7 = self.conv7(x)
        # print(f"DEBUG: conv4 shape: {conv4.shape}, up7 shape: {up7.shape}")
        x = conv4 + self.conv7(x)
        # print(f"debug 1 success")
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        return x
    
class ConvBnReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size = 3,
            stride = 1,
            pad = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = pad, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace = True)

# depth map refine network
class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)
    
    def forward(self, image, depth_init):
        # print(f"refinenet forward: image: {image.shape}, depth_init: {depth_init.shape}")
        concat = torch.cat((image, depth_init), dim = 1) # TODO: F.cat or torch.cat?
        # print(f"refinenet forward: concat: {concat.shape}")
        depth_res = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_res

        return depth_refined