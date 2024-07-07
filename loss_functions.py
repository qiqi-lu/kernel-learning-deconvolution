import torch
import torchvision
from pytorch_msssim import ms_ssim

def SSIM(x, y, c1=1e-4, c2=9e-4):
    '''
    https://github.com/eguomin/regDeconProject/blob/master/DeepLearning/Dual_Input_DL.py#L426
    '''
    if len(y.shape) == 4: dims = (-2, -1)     # 2D data
    if len(y.shape) == 5: dims = (-3, -2, -1) # 3D data
    
    mean_x = torch.mean(x, dim=dims, keepdim=True)
    mean_y = torch.mean(y, dim=dims, keepdim=True)
    sigma_x = torch.mean(torch.square(torch.sub(x, mean_x)), dim=dims, keepdim=True)
    sigma_y = torch.mean(torch.square(torch.sub(y, mean_y)), dim=dims, keepdim=True)
    sigma_cross = torch.mean(torch.mul(torch.sub(x, mean_x), torch.sub(y, mean_y)), dim=dims, keepdim=True)

    ssim_1 = 2 * torch.mul(mean_x, mean_y) + c1
    ssim_2 = 2 * sigma_cross + c2
    ssim_3 = torch.square(mean_x) + torch.square(mean_y) + c1
    ssim_4 = sigma_x + sigma_y + c2
    ssim   = torch.div(torch.mul(ssim_1, ssim_2), torch.mul(ssim_3, ssim_4))
    return ssim

def SSIM_neg_ln(x, y):
    ssim = SSIM(x, y)
    ssim = torch.mean(ssim, dim=(-2, -1))
    ssim_neg_ln = -1.0 * torch.log((1.0 + ssim) / 2.0)
    ssim_neg_ln = torch.sum(torch.mean(ssim_neg_ln, dim=(-2, -1)))
    return ssim_neg_ln

def SSIM_one_sub(x, y):
    return torch.mean(1.0 - SSIM(x, y))

def mse_center_slice(x, y, start=1):
    x_c = x[..., start:-start, :, :]
    y_c = y[..., start:-start, :, :]
    mse = torch.mean(torch.square(x_c - y_c))
    return mse

def mae_center_slice(x, y, start=1):
    x_c = x[..., start:-start, :, :]
    y_c = y[..., start:-start, :, :]
    mae = torch.mean(torch.abs(x_c - y_c))
    return mae

def MSSSIM3D(x, y):
    data_range = y.max()-y.min()
    y = torch.flatten(y, start_dim=1, end_dim=2)
    x = torch.flatten(x, start_dim=1, end_dim=2)
    msssim = ms_ssim(x, y, data_range=data_range, size_average=True, win_size=7)
    return 1.0 - msssim

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, weights_path=None, device=None):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        if weights_path == None:
            model = torchvision.models.vgg16(pretrained=True)
        else:
            print('Load weigth for [VGG16] from', weights_path)
            model = torchvision.models.vgg16()
            model.load_state_dict(torch.load(weights_path))

        model.to(device)
        blocks.append(model.features[:4].eval())
        blocks.append(model.features[4:9].eval())
        blocks.append(model.features[9:16].eval())
        blocks.append(model.features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
    
    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if len(input.shape) == 5:
            loss = 0
            for i in range(input.shape[0]):
                loss += self.forward_single(input=input[i], target=target, feature_layers=feature_layers, style_layers=style_layers)
        if len(input.shape) == 4:
            loss = self.forward_single(input=input, target=target, feature_layers=feature_layers, style_layers=style_layers)
        return loss

    def forward_single(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class TV_grad(torch.nn.Module):
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

def mae_edge(x, y, edge_func):
    loss = 0.
    if len(x.shape) == 5:
        for n in range(x.shape[0]):
            x_edge = edge_func(x[n])
            y_edge = edge_func(y)
            lo = torch.mean(torch.abs(x_edge - y_edge))
            loss = loss + lo
    if len(x.shape) == 4:
        x_edge = edge_func(x)
        y_edge = edge_func(y)
        loss = torch.mean(torch.abs(x_edge - y_edge))
    return loss


if __name__ == '__main__':
    x = torch.ones(size=(2, 3, 128, 128, 128))
    y = torch.ones(size=(2, 3, 128, 128, 128)) + 1
    ssim = SSIM(x, y)
    ssim_o = SSIM_one_sub(x,y)
    out = SSIM_neg_ln(x, y)
    print(out)