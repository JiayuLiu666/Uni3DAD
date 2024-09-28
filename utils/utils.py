import numpy as np
import random
import torch
from torchvision import transforms
from PIL import ImageFilter

def set_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

class KNNGaussianBlur(torch.nn.Module):
    def __init__(self, radius=4):
        super().__init__()
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=self.radius)

    def __call__(self, img):
        map_max = img.max()
        final_map = self.load(self.unload(img[0] / map_max).filter(self.blur_kernel)) * map_max
        return final_map
    
def get_relative_rgb_f_indices(target_pc_idices, img_size=224, f_size=28):
    scale = int(img_size / f_size)
    row = torch.div(target_pc_idices,img_size,rounding_mode='floor')
    col = target_pc_idices % img_size
    rgb_f_row = torch.div(row,scale,rounding_mode='floor')
    rgb_f_col = torch.div(col,scale,rounding_mode='floor')
    rgb_f_indices = rgb_f_row * f_size + rgb_f_col
    rgb_f_indices = torch.unique(rgb_f_indices)

    # More Background feature #
    B = 2
    rgb_f_indices = torch.cat([rgb_f_indices+B,rgb_f_indices-B,rgb_f_indices+28*B,rgb_f_indices-28*B],dim=0)
    rgb_f_indices[rgb_f_indices<0] = torch.max(rgb_f_indices)
    rgb_f_indices[rgb_f_indices>783] = torch.min(rgb_f_indices)
    rgb_f_indices = torch.unique(rgb_f_indices)

    return rgb_f_indices

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None

def get_logp(C, z, logdet_J):
    logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J
    return logp

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

def normalize(x):
    return (x - x.mean()) / x.std()

def pc_normalization(point_cloud):
    pointcloud_s = point_cloud.reshape(-1,3)
    pointcloud_s_t = pointcloud_s - (np.array([np.mean(pointcloud_s[:,0]),np.mean(pointcloud_s[:,1]),np.mean(pointcloud_s[:,2])]))
    pointcloud_s_t = pointcloud_s_t / (np.array([np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0]), np.max(pointcloud_s[:,1]) - np.min(pointcloud_s[:,1]), np.max(pointcloud_s[:,2]) - np.min(pointcloud_s[:,2])]))
    pointcloud_s = pointcloud_s_t
    return pointcloud_s