import torch

def get_device(use_cpu=False,gpu_id=0):
    '''
    Example:
    device = device_utils.get_device(gpu_id = 5)
    '''
    if use_cpu == True:
        device = torch.device("cpu")
        print('Running on the CPU.')

    elif torch.cuda.is_available():
        num_gpu = torch.cuda.device_count() # count the number of gpu
        device = torch.device("cuda:{}".format(gpu_id))
        print('Running on the GPU, GPU ID: {}, Total {}.'.format(gpu_id,num_gpu))
    else:
        device = torch.device("cpu")
        print('Running on the CPU.')
        
    return device
