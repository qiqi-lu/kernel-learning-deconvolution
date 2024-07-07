import argparse

parser = argparse.ArgumentParser(description='EDSR')

# Hardware specifications
parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu only')
parser.add_argument('--gpu_id', type=int, default=0, help='id of GPU used for processing')
parser.add_argument('--seed',type=int,default=1,help='random seed')
parser.add_argument('--act',type=str,default='relu',help='activition function')
parser.add_argument('--n_resblocks')
# Log specifications
parser.add_argument('--save_models',action='store_true',default=True,help='save all intermediate models')

args = parser.parse_args()

print(args)
