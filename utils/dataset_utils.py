import torch, os, pydicom
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms

def interp(x, ps_xy=1, ps_z=1):
    x = np.array(x, dtype=np.float32)
    num_dim = len(x.shape)

    if num_dim == 3:
        z_scale = ps_z/ps_xy
        x = torch.tensor(x)[None, None]
        x = torch.nn.functional.interpolate(x, scale_factor=(z_scale, 1, 1),\
            mode='nearest')
        x = x.numpy()[0,0]
        
    if num_dim == 2:
        z_scale = ps_z/ps_xy
        x = torch.tensor(x)[None, None]
        x = torch.nn.functional.interpolate(x, scale_factor=(z_scale, 1),\
            mode='nearest')
        x = x.numpy()[0, 0]
    return x

def gauss_kernel_1d(shape=3, std=1.0):
    x = torch.linspace(start=0, end=shape-1, steps = shape)
    x_center = (shape-1)/2

    g = torch.exp(-((x-x_center)**2 / (2.0 * std**2)))
    g = g / torch.sum(g) # shape = 3
    return g

def gauss_kernel_2d(shape=[3, 3], std=[1.0, 1.0], pixel_size=[1.0, 1.0]):
    x_data, y_data = np.mgrid[0:shape[0], 0:shape[1]]
    x_center, y_center = (shape[0]-1)/2, (shape[1]-1)/2

    x = torch.tensor(x_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.float32)
    
    g = torch.exp(-(((x-x_center)*pixel_size[0])**2 / (2.0 * std[0]**2)\
                  + ((y-y_center)*pixel_size[1])**2 / (2.0 * std[1]**2)))
    g = g / torch.sum(g) # shape = [3, 3]
    return g

def gauss_kernel_3d(shape=[3, 3, 3], std=[1.0, 1.0, 1.0], pixel_size=[1.0, 1.0, 1.0]):
    x_data, y_data, z_data = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    x_center, y_center, z_center = (shape[0]-1)/2, (shape[1]-1)/2, (shape[2]-1)/2
    x = torch.tensor(x_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.float32)
    z = torch.tensor(z_data, dtype=torch.float32)

    g = torch.exp(-(((x-x_center)*pixel_size[0])**2 / (2.0 * std[0]**2)\
                  + ((y-y_center)*pixel_size[1])**2 / (2.0 * std[1]**2)\
                  + ((z-z_center)*pixel_size[2])**2 / (2.0 * std[2]**2)))
    g = g / torch.sum(g) # shape = [3, 3, 3]
    return g


def padding_kernel(x, y):
    dim = len(y.shape)
    if dim == 3:
        i_x, j_x, k_x = x.shape
        i_y, j_y, k_y = y.shape
        if (j_x <= j_y) & (i_x <= i_y):
            x = np.pad(x, pad_width=\
                (((i_y-i_x)//2,)*2,((j_y-j_x)//2,)*2, ((k_y-k_x)//2,)*2))
    if dim == 2:
        j_x, k_x = x.shape
        j_y, k_y = y.shape
        if j_x <= j_y:
            x = np.pad(x, pad_width=(((j_y-j_x)//2,)*2, ((k_y-k_x)//2,)*2))
    return x

def ave_pooling(x, scale_factor=1):
    '''Average pooling for 2D/3D image.
    '''
    x = torch.tensor(x, dtype=torch.float32)
    if len(x.shape) == 2: x = torch.nn.functional.avg_pool2d(x[None,None],\
                                kernel_size=scale_factor)
    if len(x.shape) == 3: x = torch.nn.functional.avg_pool3d(x[None,None],\
                                kernel_size=scale_factor)
    return x.numpy()[0,0]

def add_mix_noise(x, poisson=0, sigma_gauss=0, scale_factor=1):
    '''Add Poisson and Gaussian noise.
    '''
    x = np.maximum(x, 0.0)
    # add poisson noise
    if poisson == 1:
        x_poi = np.random.poisson(lam=x)
    else:
        x_poi = x

    # downsampling
    if scale_factor > 1:
        x_poi = ave_pooling(x_poi, scale_factor=scale_factor)
        
    # add gaussian noise
    if sigma_gauss > 0:
        max_signal = np.max(x_poi)
        x_poi_norm = x_poi / max_signal
        x_poi_gaus = x_poi_norm + np.random.normal(loc=0,\
            scale=sigma_gauss / max_signal, size=x_poi_norm.shape)
        x_n = x_poi_gaus * max_signal
    else:
        x_n = x_poi

    return x_n.astype(np.float32)

def fft_n(kernel, s=None):
    kernel_fft = np.abs(np.fft.fftshift(np.fft.fftn(kernel, s=s)))
    return kernel_fft

def center_crop(x, size):
    '''Crop the center region of the 3D image x.
    '''
    dim = len(x.shape)
    if dim == 3:
        Nz, Ny, Nx = x.shape
        out = x[Nz//2 - size[0]//2: Nz//2 + size[0]//2 + 1,\
                Ny//2 - size[1]//2: Ny//2 + size[1]//2 + 1,\
                Nx//2 - size[2]//2: Nx//2 + size[2]//2 + 1]
    if dim == 2:
        Ny, Nx = x.shape
        out = x[Ny//2 - size[1]//2: Ny//2 + size[1]//2 + 1,\
                Nx//2 - size[2]//2: Nx//2 + size[2]//2 + 1]
    return out

def even2odd(x):
    '''Convert the image x to a odd-shape image.
    '''
    dim = len(x.shape)
    assert dim in [2, 3], 'Only 2D or 3D image are supported.'
    if dim == 3:
        i, j, k = x.shape
        if i % 2 == 0: i = i - 1
        if j % 2 == 0: j = j - 1
        if k % 2 == 0: k = k - 1
        x = torch.tensor(x)
        x_inter = torch.nn.functional.interpolate(x[None, None],\
            size=(i, j ,k), mode='trilinear')
    if dim == 2:
        i, j = x.shape
        if i % 2 == 0: i = i - 1
        if j % 2 == 0: j = j - 1
        x = torch.tensor(x)
        x_inter = torch.nn.functional.interpolate(x[None, None],\
            size=(i, j), mode='bilinear')
    x_inter = x_inter / x_inter.sum()
    return x_inter.numpy()[0,0]

def percentile_norm(x, p_low=0, p_high=100):
    '''percentile-based normalization.
    '''
    xmax, xmin = np.percentile(x, p_high), np.percentile(x, p_low)
    x = (x - xmin) / (xmax - xmin)
    x = np.clip(x, a_min=0.0, a_max=1.0)
    return x

def linear_transform(x, y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    x_linear = b_1*x + b_0
    return x_linear

def read_image(img_path, normalization=False, data_range=None):
    '''
    Read image.
    Args:
    - img_path (str): Image path.
    - normalization (bool): Normalize data into (0,1).
    - data_range (tuple): (min, max) value of data.
    '''
    # check file type, get extension of file
    _, ext = os.path.splitext(img_path)

    # DICOM data
    if ext == '.dcm':
        img_dcm = pydicom.dcmread(img_path)
        img = img_dcm.pixel_array
        img = img.astype(np.float32)

    # TIFF data
    if ext == '.tif': img = io.imread(img_path)

    if len(img.shape) == 2: 
        img = np.expand_dims(img, axis=0)
    elif len(img.shape) == 3: 
        img = np.expand_dims(img, axis=0)

    # Image normalization
    if normalization == True:
        if data_range == None:
            img_max, img_min = img.max(), img.min()
            img = (img - img_min) / (img_max - img_min)
        if type(data_range) == tuple:
            img = (img - data_range[0]) / (data_range[1] - data_range[0])

    return img.astype(np.float32)

class SRDataset(Dataset):
    '''
    Super-resolution dataset used to get low-resolution and hig-resolution data.
    Args:
    - hr_root_path (str): root path for high-resolution data.
    - lr_root_path (str): root path for  low-resolution data.
    - hr_txt_file_path (str): path of file saving path of high-resolution data.
    - lr_txt_file_path (str): path of file saving path of low-resolution data.
    - id_range (tuple): extract part of the data. 
                        Default: None, all the data in dataset.
    - transform (bool): data transformation. Default: None.
    - normalization (tuple[bool]): whether to normalize the data 
                    when read image (lr, hr). Default: (False, False).
    '''
    def __init__(self, hr_root_path, lr_root_path, hr_txt_file_path,\
        lr_txt_file_path, id_range=None, transform=None,\
        normalization=(False, False)):
        super().__init__()
        self.hr_root_path  = hr_root_path
        self.lr_root_path  = lr_root_path
        self.transform     = transform
        self.normalization = normalization

        with open(lr_txt_file_path) as f:
            self.file_names_lr = f.read().splitlines()
        with open(hr_txt_file_path) as f:
            self.file_names_hr = f.read().splitlines()

        if id_range != None:
            data_size = len(self.file_names_lr)
            self.file_names_lr = self.file_names_lr[id_range[0]: id_range[1]]
            self.file_names_hr = self.file_names_hr[id_range[0]: id_range[1]]

            print('DATASET: Use only part of datasets. ({}|{})'\
                .format(len(self.file_names_lr), data_size))

    def __len__(self):
        return len(self.file_names_lr)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        
        img_path_lr = os.path.join(self.lr_root_path, self.file_names_lr[idx])
        img_path_hr = os.path.join(self.hr_root_path, self.file_names_hr[idx])

        image_lr = read_image(img_path_lr, normalization=self.normalization[0])
        image_hr = read_image(img_path_hr, normalization=self.normalization[1])

        if self.transform is not None:
            image_lr = self.transform(image_lr)
            image_hr = self.transform(image_hr)

        # scale = np.percentile(image_hr, 95)
        # return {'lr': torch.tensor(image_lr/scale), 'hr': torch.tensor(image_hr/scale)}
        return {'lr': torch.tensor(image_lr),\
                'hr': torch.tensor(image_hr)}

class CytoDataset(Dataset):
    '''
    Super-Resolution dataset.
    - A total of 239100 tile LR and HR registered image pairs from 28 different whole slide image.
    - The registered image pairs were divided into training and testing according to an 8:2 ration.
    - There are 191280 pairs of tile images in the training set and 47820 pairs of tile images in
    the testing set.
    '''
    def __init__(self,txt_file,root_dir,id_range=None,transform=None):
        super().__init__()
        txt_file_lr = os.path.join(txt_file,'lr.txt')
        txt_file_hr = os.path.join(txt_file,'hr.txt')

        with open(txt_file_lr) as f: self.file_names_lr = f.read().splitlines()
        with open(txt_file_hr) as f: self.file_names_hr = f.read().splitlines()

        if id_range != None:
            data_size = len(self.file_names_lr)
            self.file_names_lr = self.file_names_lr[id_range[0]:id_range[1]]
            self.file_names_hr = self.file_names_hr[id_range[0]:id_range[1]]
            print('Use only part of datasets. ({}|{})'.format(len(self.file_names_lr),data_size))

        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.file_names_lr)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_dir_lr = os.path.join(self.root_dir,self.file_names_lr[idx])
        img_dir_hr = os.path.join(self.root_dir,self.file_names_hr[idx])

        image_lr = io.imread(img_dir_lr)
        image_hr = io.imread(img_dir_hr)

        if self.transform:
            image_lr = self.transform(image_lr)
            image_hr = self.transform(image_hr)

        sample = {'lr': image_lr, 'hr': image_hr}

        return sample

class CytoDataset_synth(Dataset):
    def __init__(self,txt_file,dir_hr,dir_synth,id_range=None,transform=None):
        super().__init__()
        print('Training on synthetic datasets: ',dir_synth)
        txt_file_hr = os.path.join(txt_file,'hr.txt')

        with open(txt_file_hr) as f: 
            self.file_names_hr = f.read().splitlines()

        if id_range != None:
            data_size = len(self.file_names_hr)
            self.file_names_hr = self.file_names_hr[id_range[0]:id_range[1]]
            print('Use only part of datasets. ({}|{})'.format(len(self.file_names_hr),data_size))
        else:
            print('Use all datasets, total {}.'.format(len(self.file_names_hr)))

        self.dir_hr  = dir_hr
        self.dir_synth = dir_synth
        self.transform = transform
 
    def __len__(self):
        return len(self.file_names_hr)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        p = os.path.split(self.file_names_hr[idx])
        subject = os.path.split(p[0])[1]

        img_dir_lr = os.path.join(self.dir_synth,subject,p[1])
        img_dir_hr = os.path.join(self.dir_hr,self.file_names_hr[idx])

        image_lr = io.imread(img_dir_lr)
        image_hr = io.imread(img_dir_hr)

        if self.transform:
            image_lr = self.transform(image_lr)
            image_hr = self.transform(image_hr)

        sample = {'lr': image_lr, 'hr': image_hr}

        return sample

class Rescale(object):
    '''
    Rescale the image in a sample to a given size.
    '''
    def __init__(self,output_size):
        assert isinstance(output_size,(int, tuple))
        self.output_size = output_size

    def __call__(self,sample):
        image_lr, image_hr = sample['lr'], sample['hr']

        h, w = image_lr.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size
        
        new_h, new_w = int(new_h), int(new_w)
        image_lr_new = transform.resize(image_lr, (new_h,new_w))

        return {'lr': image_lr_new, 'hr': image_hr}

class ToNumpy(object):
    '''
    Convert pytorch tensor into numpy array, and shift the channel axis to the last axis.
    Args:
    - tensor (torch tensor): input tensor.
    '''
    def __call__(self, tensor):
        img = tensor.cpu().detach().numpy()
        # move the chennel axis to the last dimension.
        if len(img.shape) == 4: img = np.transpose(img, axes=(0, 2, 3, 1))
        if len(img.shape) == 5: img = np.transpose(img, axes=(0, 2, 3, 4, 1))
        return img

def tensor2rgb(x):
    x = torch.clamp(x, min=0.0, max=1.0)
    x = (x * 255.0).to(torch.uint8)
    x = x.cpu().detach().numpy()
    if len(x.shape) == 4: x = np.transpose(x, axes=(0, 2, 3, 1))
    if len(x.shape) == 5: x = np.transpose(x, axes=(0, 1, 3, 4, 2))
    return x

def tensor2gray(x):
    x = x.cpu().detach().numpy()
    if len(x.shape) == 4: x = np.transpose(x, axes=(0, 2, 3, 1))
    if len(x.shape) == 5: x = np.transpose(x, axes=(0, 1, 3, 4, 2))
    return x

if __name__ == '__main__':
    # data_set_name = 'tinymicro_synth'
    # data_set_name = 'tinymicro_real'
    # data_set_name = 'biosr_real'
    data_set_name = 'lung3_synth'
    # data_set_name = 'msi_synth'

    # -------------------------------------------------------------------------------------
    if data_set_name == 'tinymicro_synth':
        # TinyMicro (synth)
        hr_root_path = os.path.join('data', 'raw', 'cyto_potable_microscope', 'data1')
        lr_root_path = os.path.join('data', 'raw', 'cyto_potable_microscope', 'data_synth', 'train', 'sf_4_k_2.0_gaussian_mix_ave') # TinyMicro (synth)

        hr_txt_file_path = os.path.join('data', 'raw', 'cyto_potable_microscope', 'train_txt', 'hr.txt') 
        lr_txt_file_path = os.path.join('data', 'raw', 'cyto_potable_microscope', 'train_txt', 'lr.txt')
        normalization = (False, False)

    if data_set_name == 'tinymicro_real':
        # TinyMicro (real)
        hr_root_path = os.path.join('data', 'raw', 'cyto_potable_microscope', 'data1')
        lr_root_path = os.path.join('data', 'raw', 'cyto_potable_microscope', 'data1')

        hr_txt_file_path = os.path.join('data', 'raw', 'cyto_potable_microscope', 'train_txt', 'hr.txt') 
        lr_txt_file_path = os.path.join('data', 'raw', 'cyto_potable_microscope', 'train_txt', 'lr.txt')
        normalization = (False, False)

    if data_set_name == 'biosr_real':
        pass
    if data_set_name == 'lung3_synth':
        # Lung3 (synth) 
        # F:\Datasets\Lung3\manifest-41uMmeOh151290643884877939
        # F:\Datasets\Lung3\manifest-41uMmeOh151290643884877939\data_synth\train\sf_4_k_2.0_gaussian_mix_ave
        hr_root_path = os.path.join('F:', os.sep, 'Datasets', 'Lung3', 'manifest-41uMmeOh151290643884877939')
        lr_root_path = os.path.join('F:', os.sep, 'Datasets', 'Lung3', 'manifest-41uMmeOh151290643884877939', 'data_synth', 'train', 'sf_4_k_2.0_gaussian_mix_ave')

        hr_txt_file_path = os.path.join('F:', os.sep, 'Datasets', 'Lung3', 'manifest-41uMmeOh151290643884877939', 'train_txt', 'hr.txt') 
        lr_txt_file_path = os.path.join('F:', os.sep, 'Datasets', 'Lung3', 'manifest-41uMmeOh151290643884877939', 'train_txt', 'lr.txt') 
        normalization = (False, True)
    if data_set_name == 'msi_synth':
        pass
    
    fig_dir  = os.path.join('outputs','figures')

    # -------------------------------------------------------------------------------------
    trans = transforms.Compose([
        transforms.ToTensor(),
        ])

    # -------------------------------------------------------------------------------------
    paired_dataset = SRDataset(hr_root_path=hr_root_path, lr_root_path=lr_root_path,\
        hr_txt_file_path=hr_txt_file_path, lr_txt_file_path=lr_txt_file_path,\
        transform=trans, id_range=[0, 1000], normalization=normalization)

    print('Datasize: ', paired_dataset.__len__())

    dataloader = DataLoader(dataset=paired_dataset, batch_size=5, shuffle=False, num_workers=0)

    # -------------------------------------------------------------------------------------
    i_batch_show = 0
    for i_batch, sample in enumerate(dataloader):
        print(i_batch, sample['lr'].size(), sample['hr'].size(), 'max: ', torch.max(sample['hr']).item(),\
            'min: ',torch.min(sample['hr']).item())
        if i_batch == i_batch_show:
            fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 5), dpi=600, constrained_layout=True)
            [ax.set_axis_off() for ax in axes.ravel()]

            images_lr_batch, images_hr_batch = sample['lr'], sample['hr']
            if images_hr_batch.shape[1] == 1:
                cm = 'gray'
            else:
                cm = None
            for i in range(5):
                axes[0, i].imshow(images_lr_batch[i].transpose(0, -1).transpose(0, 1), cmap=cm, vmin=0.0, vmax=1.0)
                axes[1, i].imshow(images_hr_batch[i].transpose(0, -1).transpose(0, 1), cmap=cm, vmin=0.0, vmax=1.0)

            save_to = os.path.join(fig_dir, data_set_name)
            if os.path.exists(save_to) == False:
                os.makedirs(save_to, exist_ok=True)
            plt.savefig(os.path.join(fig_dir, data_set_name, 'sample_batch_{}'.format(i_batch_show)))
            break
    print('end')