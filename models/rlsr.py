import torch, math
import torch.nn as nn
from models import rln, rrdbnet, swinir, srcnn, unet, ddn, srgan

class Upsampler_UC(nn.Module):
    '''
    Upsampler. up-comv.

    Args:
    - scale_factor (int): scale_factor. Default: 4.
    - n_features (int): number of features. Default: 64.
    - bias (bool): bias in the convolutional layers. Default: bool.
    - pm (str): padding mode. Default: 'zero'.
    '''
    def __init__(self, scale_factor=4, n_features=64, bias=True, pm='zeros'):
        super().__init__()

        self.scale_factor = scale_factor

        self.conv1 = nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3,\
            stride=1, padding=1, bias=bias, padding_mode=pm)
        self.conv2 = nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3,\
            stride=1, padding=1, bias=bias, padding_mode=pm)
        self.conv3 = nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3,\
            stride=1, padding=1, bias=bias, padding_mode=pm)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale_factor == 4:
            fea = self.act(self.conv1(nn.functional.interpolate(input=x,   scale_factor=2, mode='nearest')))
            fea = self.act(self.conv2(nn.functional.interpolate(input=fea, scale_factor=2, mode='nearest')))

        if self.scale_factor == 3:
            fea = self.act(self.conv1(nn.functional.interpolate(input=x, scale_factor=3, mode='nearest')))
            fea = self.act(self.conv2(fea))

        if self.scale_factor == 2:
            fea = self.act(self.conv1(nn.functional.interpolate(input=x, scale_factor=2, mode='nearest')))
            fea = self.act(self.conv2(fea))

        out = self.act(self.conv3(fea))
        return out

class Upsampler_PS(nn.Module):
    '''
    Based on pixel shuffle.

    Args:
    - scale (int): scale factor Default: 4.
    - n_features (int): number of features. Defualt: 8.
    - kernel_size (int): kernel size. Default: 3.
    - bn (Bool): whether to use batch normalization. Defualt: False.
    - act (str): activation function name. Default: False.
    - bias (bool): bias in the convolutional layer. Default: True.
    '''
    def __init__(self, scale=4, n_features=8, kernel_size=3, bn=False, act=False, bias=True):
        super().__init__()
        modules = []
        if (scale & (scale - 1)) == 0: # 2, 4, 8... 2^n
            for _ in range(int(math.log(scale, 2))):
                modules.append(nn.Conv2d(in_channels=n_features, out_channels=4 * n_features,\
                    kernel_size=kernel_size, padding=(kernel_size // 2), bias=bias))
                modules.append(nn.PixelShuffle(2))
                if bn == True:      modules.append(nn.BatchNorm2d(num_features=n_features))
                if act == 'relu' :  modules.append(nn.ReLU(inplace=True))
                if act == 'prelu':  modules.append(nn.PReLU(num_parameters=n_features))
        elif scale == 3:
            modules.append(nn.Conv2d(in_channels=n_features, out_channels=9 * n_features,\
                kernel_size=kernel_size, padding=(kernel_size // 2), bias=bias))
            modules.append(nn.PixelShuffle(3))
            if bn == True:      modules.append(nn.BatchNorm2d(num_features=n_features))
            if act == 'relu' :  modules.append(nn.ReLU(inplace=True))
            if act == 'prelu':  modules.append(nn.PReLU(num_parameters=n_features))
        else:
            raise NotImplementedError
        
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return self.body(x)

class Tail(nn.Module):
    '''
    '''
    def __init__(self, in_channels=3 ,n_features=8, kernel_size=3, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_features, kernel_size=kernel_size,\
            stride=1, padding=(kernel_size // 2), bias=bias)
        self.conv2 = nn.Conv2d(in_channels=n_features + in_channels, out_channels=n_features,\
            kernel_size=kernel_size, stride=1, padding=(kernel_size // 2), bias=bias) 
        self.conv3 = nn.Conv2d(in_channels=n_features + n_features, out_channels=n_features,\
            kernel_size=kernel_size, stride=1, padding=(kernel_size // 2), bias=bias) 
        self.conv4 = nn.Conv2d(in_channels=n_features, out_channels=in_channels,\
            kernel_size=kernel_size, stride=1, padding=(kernel_size // 2), bias=bias) 

        self.act3 = nn.Softplus()
        self.act1 = nn.Softplus()
        self.act2 = nn.Softplus()
        self.act4 = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.act1(self.conv1(x))
        conv2 = self.act2(self.conv2(torch.cat(tensors=[conv1, x], dim=1)))
        conv3 = self.act3(self.conv3(torch.cat(tensors=[conv1, conv2], dim=1)))
        out   = self.act4(self.conv4(conv3))
        return out

class image_style_transfer(nn.Module):
    '''
    Args:
    - in_channels (int): input image channels.
    '''
    def __init__(self, in_channels=3, kernel_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        )

    def forward(self, x):
        residual = self.net(x)
        return x + residual

class Initializer(nn.Module):
    '''
    Use for initilization.
    - in_channels: input image channels.
    '''
    def __init__(self, in_channels, scale, kernel_size=3):
        super().__init__()
        self.scale = scale
        self.transfer = image_style_transfer(in_channels=in_channels, kernel_size=kernel_size)
    
    def forward(self, x):
        xsr = nn.functional.interpolate(input=x, scale_factor=self.scale, mode='bilinear')
        xsr = self.transfer(xsr) # style-transfer
        return xsr
        
class ForwardProject(nn.Module):
    '''
    Args:
    - backbone (nn.Module): Module used to extract features.
    - in_channels (int): Number of input channels. Default: 3.
    - n_features (int): Number fo features. Default: 64.
    - scale (int): Downsampling scale factor. Default: 4.
    - bias (bool): Bias used in the convolution layer. Default: True.
    - pm (str, optional): padding mode, 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'.
    - kernel_size (int): Kernel size of the last convolutional layer. Default: 3.
    - only_net (bool): Use only the network part without pooling layer. 
    '''
    def __init__(self, backbone, in_channels=3, fpm=1, n_features=64, scale_factor=4, bias=True, bn=False,\
        kernel_size=3, only_net=False, pixel_binning_mode='ave'):
        super().__init__()
        self.bn = bn
        self.pixel_binning_mode = pixel_binning_mode
        self.only_net  = only_net
        self.extracter = backbone # feature extraction

        if self.bn == True: self.norm = nn.BatchNorm2d(num_features=n_features)
        self.conv = nn.Conv2d(in_channels=n_features, out_channels=in_channels * fpm, kernel_size=kernel_size,\
            stride=1, padding=kernel_size // 2, bias=bias)

        if self.only_net == False: 
            self.pool = nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor)
    
    def forward(self, x):
        fea = self.extracter(x)
        if self.bn == True: fea = nn.ReLU(inplace=True)(self.norm(fea))
        out = nn.Sigmoid()(self.conv(fea))

        if self.only_net == False: out = self.pool(out)
        return out

class BackwardProject(nn.Module):
    '''
    Args:
    - backbone (nn.Module): Module used to extract features.
    - upsample (nn.Module): Up-sampler.
    - only_net (bool): Without plus one part. Default: False.
    '''
    def __init__(self, backbone, upsample, only_net=False, scale_factor=4, RL_version='RL'):
        super().__init__()
        self.extracter = backbone
        self.upsample  = upsample

        self.eps = 0.0001
        self.only_net  = only_net
        self.scale_factor = scale_factor
        self.RL_version = RL_version

    def forward(self, x):
        fea = self.extracter(x)     # feature extraction
        map = self.upsample(fea)    # up-sample

        if self.RL_version == 'RL':
            if self.only_net == False: 
                map = map + torch.ones_like(map)
                out = nn.ReLU(inplace=True)(map) + self.eps
            if self.only_net == True:
                out = nn.ReLU(inplace=True)(map)

        if self.RL_version == 'ISRA':
            out = nn.Softplus()(map)
        return out

class Prior(nn.Module):
    '''
    Image prior.
    Args:
    - backbone (nn.Module): Module used to extract features.
    - n_features (int): number fo features out from backbone.
    - out_channels (int): Number of channels of the output. Default: 3.
    '''
    def __init__(self, backbone, n_features, out_channels=3, kernel_size=3):
        super().__init__()
        self.extractor = backbone
        self.conv = nn.Conv2d(in_channels=n_features, out_channels=out_channels, kernel_size=kernel_size,\
            stride=1, padding=kernel_size // 2, bias=True)
    
    def forward(self, x):
        fea = self.extractor(x)
        out = self.conv(fea)
        return out

class TV_grad(nn.Module):
    '''
    Gradient of TV.
    @ Wang, C. et al. Sparse deconvolution for background noise suppression with
    total variation regularization in light field microscopy. Opt Lett 48, 1894 (2023).
    '''
    def __init__(self, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x):
        batch = []
        for i in range(x.shape[0]):
            sample = []
            for j in range(x.shape[1]):
                x_sc = x[i, j] # single chennel image
                u = torch.gradient(x_sc, edge_order=1)            # gradient
                u0 = u[0] / (torch.sqrt(self.epsilon + u[0]**2 + u[1]**2))
                u1 = u[1] / (torch.sqrt(self.epsilon + u[0]**2 + u[1]**2))
                d = torch.gradient(u0)[0] + torch.gradient(u1)[1] # divergence
                sample.append(d)
            sample = torch.stack(sample, dim=0)
            batch.append(sample)
        batch = torch.stack(batch, dim=0)
        return batch

class RLSR(nn.Module):
    '''
    Args:
    - image_size (list): Input image size. Default: (128,128).
    - scale (int): Upsampling scale. Default: 4.
    - in_channels (int): Number of channels of input.
    - n_features (int): Number of features or embedding dimension. Default: 64.
    - n_blocks (tuple [int]): Number of bisic blocks. Default: (1, 1).
    - n_iter (int): Number of iterations. Default: 1.
    - init_mode (str): Algotithms used for initialization: `'nearest'` | `'bilinear'`. Default:`'bicubic'`.
    - backbone_type (str): The network used as backbone, 'rrdb', 'rln', or 'swinir'. Default: 'rrdb'.
    - window_size (int): The window size used in Residual Swin Transformer Block (RSTB). Default: 8.
    - mlp_ratio (int): ?. Default: 2.
    - kernel_size (int): Kernel size of the conv after upsampler. Default: 3.
    - only_net (bool): Only use network part, without model-based structures. Default: False.
    - output_mode (str): `last-one`, `each-iter-train`, `each-iter-test`, `inters`.
    - initializer (nn.Module): pretrained network for initilization. Defualt: None.
    - RL_version (str): Version of RL algorithm. 'RL', 'ISRA'.
    '''
    def __init__(self, img_size=(128, 128), scale=4, in_channels=3, n_features=(64,64), n_blocks=(1, 1), n_iter=1,\
                bias=True, init_mode='ave_bicubic', backbone_type='rrdb', window_size=8, mlp_ratio=2,\
                kernel_size=3, only_net=False, output_mode='each-iter-train', initializer=None, bn=True,\
                upsample_mode='conv_up', bp_ks=3,\
                use_prior=False, prior_type='learned', lambda_prior = 1.0, train_only_prior = False, prior_inchannels=1,\
                RL_version='RL', pixel_binning_mode='ave',\
                constraint_01=True, forw_proj=None, cat_x=False,\
                fpm=1, bpm=1, prior_bn=False):
        super().__init__()
        self.eps = 0.0001
        self.scale = scale
        self.n_iter = n_iter # number of iteration
        self.init_mode = init_mode
        self.only_net  = only_net
        self.RL_version = RL_version
        self.pixel_binning_mode = pixel_binning_mode
        self.n_features = n_features
        self.constraint_01 = constraint_01
        self.bn = bn
        self.cat_x = cat_x

        self.fpm = fpm
        self.bpm = bpm

        self.upsample_mode = upsample_mode

        # Prior parameter
        self.use_prior = use_prior
        self.train_only_prior = train_only_prior
        self.prior_type = prior_type
        self.lambda_prior = lambda_prior
        self.prior_inchannels = prior_inchannels

        # output parameter
        self.output_mode = output_mode

        print('Output mode: {}'.format(self.output_mode))

        # -------------------------------------------------------------------------------------------
        # network-based style-transform
        # -------------------------------------------------------------------------------------------
        if self.only_net == True:
            if self.init_mode in ['net', 'pre-trained']:
                self.transfer = image_style_transfer(in_channels=in_channels)

        if self.only_net == False:
            if self.init_mode == 'net':
                self.transfer = image_style_transfer(in_channels=in_channels)
            if self.init_mode == 'pre-trained':
                self.initializer = initializer
                for param in self.initializer.parameters(): param.requires_grad = False

        # -------------------------------------------------------------------------------------------
        # Forward projection
        # -------------------------------------------------------------------------------------------
        if forw_proj == None: # no pre-trained forward projection network
            if backbone_type == 'rrdb':
                backbone_fp = rrdbnet.backbone(in_channels=in_channels, n_features=self.n_features[0],\
                    n_blocks=n_blocks[0], growth_channels=self.n_features[0] // 2, bias=bias)
            if backbone_type == 'srcnn':
                backbone_fp = srcnn.backbone(num_channels=in_channels, n_features=self.n_features[0])

            self.FP = ForwardProject(backbone=backbone_fp, in_channels=in_channels, n_features=self.n_features[0],\
                scale_factor=self.scale, bias=bias, kernel_size=kernel_size, only_net=self.only_net,\
                pixel_binning_mode=pixel_binning_mode, fpm=self.fpm)

        elif isinstance(forw_proj, nn.Module): # pre-trained forward projection network
            self.FP = forw_proj 
            for param in self.FP.parameters(): param.requires_grad = False
        else: # known forward projector
            self.FP = forw_proj

        # -------------------------------------------------------------------------------------------
        # Upsample block
        # -------------------------------------------------------------------------------------------
        upsample = []
        if upsample_mode == 'conv_up':
            upsample.append(Upsampler_PS(scale=self.scale, n_features=self.n_features[1], bn=False, act=False, bias=bias))
            # upsample.append(Upsampler_UC(scale_factor=self.scale, n_features=self.n_features[1], bias=bias, pm=pm))
        if bn == True:
            upsample.append(nn.BatchNorm2d(num_features=self.n_features[1]))
            upsample.append(nn.ReLU(inplace=True))
        upsample.append(nn.Conv2d(in_channels=self.n_features[1], out_channels=in_channels * self.bpm,\
            kernel_size=bp_ks, stride=1, padding=bp_ks // 2, bias=bias))
        upsample = nn.Sequential(*upsample)
        
        # -------------------------------------------------------------------------------------------
        # Backward projection
        # -------------------------------------------------------------------------------------------
        in_chan = in_channels * self.fpm
        if self.cat_x == True: in_chan = in_chan + in_channels
        if backbone_type == 'rrdb':
            backbone_bp = rrdbnet.backbone(in_channels=in_chan, n_features=self.n_features[1], n_blocks=n_blocks[1],\
                growth_channels=self.n_features[1] // 2, bias=bias, use_bn=False)
        if backbone_type == 'srcnn':
            backbone_bp = srcnn.backbone(num_channels=in_chan, n_features=self.n_features[1])

        self.BP = BackwardProject(backbone=backbone_bp, upsample=upsample, only_net=self.only_net, RL_version=RL_version)
        
        # -------------------------------------------------------------------------------------------
        # Prior
        # -------------------------------------------------------------------------------------------
        if self.use_prior == True:
            if self.prior_type == 'learned':
                in_chan = in_channels * (self.prior_inchannels -1) + in_channels * self.bpm
                backbone_pri = unet.backbone(in_channels=in_chan, bilinear=False, use_bn=prior_bn)
                self.Prior = Prior(backbone=backbone_pri, n_features=8, out_channels=in_channels, kernel_size=3)
            if self.prior_type == 'TV':
                self.Prior = TV_grad(epsilon=1.0)

        # train only the prior network, free FP and BP part.
        if self.train_only_prior == True:
            print('# Frezee FP and BP.')
            for param in self.FP.parameters(): param.requires_grad = False
            for param in self.BP.parameters(): param.requires_grad = False

        # weight initialization
        # self.BP.apply(ddn.init_weights)
        self.Edge = TV_grad(epsilon=1.0)

    def forward(self, x):
        # ------------------------------------------------------------------------------------------
        # Only use network part, without any model-based structures (include interpolation, pooling,
        # division, multiplication).
        if self.only_net == True: 
            if self.init_mode in ['net', 'pre-trained']: x = self.transfer(x)
            x = self.FP(x)
            x = self.BP(x)
            if self.use_prior == True: x = self.Prior(x)
            return x

        # ------------------------------------------------------------------------------------------
        if self.only_net == False:
            out = []
            # --------------------------------------------------------------------------------------
            # initialization 
            if self.init_mode == 'net':
                xsr = nn.functional.interpolate(input=x, scale_factor=self.scale, mode='bilinear')
                xsr = self.transfer(xsr) # style-transfer

            if self.init_mode == 'pre-trained':
                xsr = self.initializer(x)

            if self.init_mode in ['bicubic', 'bilinear', 'nearest']:
                xsr = nn.functional.interpolate(input=x, scale_factor=self.scale, mode=self.init_mode)

            if self.init_mode == 'constant':
                xsr = nn.functional.interpolate(input=x, scale_factor=self.scale, mode='nearest')
                xsr = torch.ones_like(xsr) * 0.5
            
            if self.init_mode == 'ave_bicubic':
                x_ave = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=3 // 2)(x) # suppress noise
                xsr = nn.functional.interpolate(input=x_ave, scale_factor=self.scale, mode='bicubic')
                
            # --------------------------------------------------------------------------------------
            xsr = self.constraint_01_pos(xsr)
            if self.output_mode in ['each-iter-train', 'each-iter-train-prior']:
                if self.init_mode == 'net': out.append(xsr)
            if self.output_mode in ['each-iter-test', 'inters']: out.append(xsr)

            # --------------------------------------------------------------------------------------
            for i_iter in range(self.n_iter):
                # Forward Projection
                fp = self.FP(xsr)
                # ----------------------------------------------------------------------------------
                # Division
                if (self.RL_version == 'RL') or (self.output_mode == 'inters'):
                    dv = torch.div(x.repeat(1, self.fpm, 1, 1), fp + self.eps)
                # ----------------------------------------------------------------------------------
                # Richardson Lucy iteration
                if self.RL_version == 'RL':     
                    # Backward-Projection
                    if self.cat_x == False: bp = self.BP(dv)
                    if self.cat_x == True:
                        if self.upsample_mode == 'conv_up': 
                            bp = self.BP(torch.cat((self.Edge(x), dv), dim=1))
                        if self.upsample_mode == 'up_conv':
                            dv_up = nn.functional.interpolate(input=dv, scale_factor=self.scale, mode='bicubic')
                            bp = self.BP(torch.cat((self.Edge(xsr), dv_up), dim=1))

                    # Update
                    xsr0 = torch.mul(xsr, bp)
                    xsr0 = self.constraint_01_pos(xsr0)

                    # Prior
                    if self.use_prior:
                        if self.prior_inchannels == 1:
                            # prior_input = self.Edge(xsr0)
                            prior_input = xsr0
                        if self.prior_inchannels == 2:
                            prior_input = torch.cat((xsr0, self.Edge(xsr0)), dim=1)
                        if self.prior_inchannels == 3:
                            x_inter = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
                            prior_input = torch.cat((xsr0, x_inter, self.Edge(xsr0)), dim=1)

                        one_pg = 1.0 + self.lambda_prior * self.Prior(prior_input)
                        one_pg = nn.ReLU(inplace=True)(one_pg) + self.eps
                        xsr = torch.div(xsr0, one_pg)
                        xsr = self.constraint_01_pos(xsr)
                    else: 
                        xsr = xsr0
                # ----------------------------------------------------------------------------------
                # Image Space Restoration Algorithm
                if self.RL_version == 'ISRA':
                    # Backward-Projection
                    if self.cat_x == False:
                        bp_b, bp_fp  = self.BP(x), self.BP(fp)
                    if self.cat_x == True:
                        bp_b  = self.BP(torch.cat((x, self.Edge(x)), dim=1))
                        bp_fp = self.BP(torch.cat((fp, self.Edge(fp)), dim=1))

                    bp = torch.div(bp_b, bp_fp + self.eps)

                    # Prior
                    if self.use_prior:
                        if self.bpm == 1:
                            xsr0 = torch.mul(xsr, bp)
                            xsr0 = self.constraint_01_pos(xsr0)

                        # damping operation
                        # 1-P(bp)
                        one_bp = torch.sub(1.0, bp)
                        if self.prior_inchannels == 1:
                            prior_input = one_bp
                        if self.prior_inchannels == 2:
                            prior_input = torch.cat((one_bp, xsr), dim=1)
                            # prior_input = torch.cat((bp_b, bp_fp), dim=1)
                        if self.prior_inchannels == 3:
                            x_inter = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
                            prior_input = torch.cat((one_bp, xsr, x_inter), dim=1)
                        one_pg = 1.0 - self.Prior(prior_input)
                        # one_pg = 1.0 - (self.Prior(prior_input) + one_bp)
                        one_pg = nn.ReLU(inplace=True)(one_pg) + self.eps
                    else:
                        one_pg = bp

                    # Update
                    xsr = torch.mul(xsr, one_pg)
                    xsr = self.constraint_01_pos(xsr)
                # ----------------------------------------------------------------------------------
                # Collect outputs
                if self.output_mode == 'inters':
                    if self.use_prior == True:  out.extend([fp, dv, bp, xsr, xsr0, one_pg])
                    if self.use_prior == False: out.extend([fp, dv, bp, xsr])

                if self.output_mode == 'each-iter-train': out.append(xsr)
                if self.output_mode == 'each-iter-train-prior': out.extend([xsr0, xsr])

                if self.output_mode == 'each-iter-test':
                    if self.use_prior == True:  out.extend([xsr0, xsr])
                    if self.use_prior == False: out.append(xsr)

            # ---------------------------------------------------------------------------------------
            if self.output_mode in ['each-iter-train', 'each-iter-test', 'each-iter-train-prior']: 
                return torch.stack(out, dim=0)
            if self.output_mode == 'inters':   return out
            if self.output_mode == 'last-one': return xsr

    def forward_project(self, x):
        return self.FP(x)

    def back_project(self, x):
        return self.BP(x)
        
    def constraint_01_pos(self, x):
        if self.constraint_01 == True:
            x = torch.clamp(x, min=0.0, max=1.0)
            x = torch.where(x == 0.0, self.eps, x)
        else:
            x = x
        return x
            
        