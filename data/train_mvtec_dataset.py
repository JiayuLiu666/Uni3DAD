import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
import torch
from Common import point_operation
from Common import data_utils as d_utils
# from loss import farthest_point_sample as fps


# DATASETS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets', 'mvtec3d'))

DATASETS_PATH = '/home/liuj35/datasets/mvtec3d/'

def fps(xyz, npoint):
    
    """
    code borrowed from: http://www.programmersought.com/article/8737853003/#14_query_ball_point_93
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
    	# Update the i-th farthest point
        centroids[:, i] = farthest
        # Take the xyz coordinate of the farthest point
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        # Calculate the Euclidean distance from all points in the point set to this farthest point
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # Update distances to record the minimum distance of each point in the sample from all existing sample points
        mask = dist < distance
        distance[mask] = dist[mask]
        # Find the farthest point from the updated distances matrix, and use it as the farthest point for the next iteration
        farthest = torch.max(distance, -1)[1]
    return centroids


def get_neighbor_mean(img, p):
    n_neighbors = np.sum(img[p[0] - 1: p[0] + 2, p[1] - 1: p[1] + 2] > 0)
    if n_neighbors == 0:
        return None
    nb_mean = np.sum(img[p[0] - 1: p[0] + 2, p[1] - 1: p[1] + 2], axis=(0, 1)) / n_neighbors
    return nb_mean


def fill_gaps(img):
    new_img = np.copy(img)
    zero_pixels = np.where(img[:,:,2] == 0)
    for x, y in zip(*zero_pixels):
        if img[x, y, 2] == 0:
            nb_mean = get_neighbor_mean(img[:,:,2], [x, y])
            if nb_mean is not None:
                new_img[x, y] = nb_mean
    return new_img


def get_corner_points(img):
    upper_left = np.sum(img[:2, :2]) / np.sum(img[:2, :2] > 0)
    upper_right = np.sum(img[-2:, :2]) / np.sum(img[-2:, :2] > 0)
    lower_left = np.sum(img[:2, -2:]) / np.sum(img[:2, -2:] > 0)
    lower_right = np.sum(img[-2:, -2:]) / np.sum(img[-2:, -2:] > 0)
    return upper_left, upper_right, lower_left, lower_right


def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]


class MVTec3D(Dataset):

    def __init__(self, split, class_name, img_size):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(DATASETS_PATH, self.cls, split)
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])

class MVTec3DTrain(MVTec3D):
    def __init__(self, class_name, img_size):
        super().__init__(split="train", class_name=class_name, img_size=img_size)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        # rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        # rgb_paths.sort()
        tiff_paths.sort()
        # sample_paths = list(zip(rgb_paths, tiff_paths))
        sample_paths = list(zip(tiff_paths))

        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        # rgb_path = img_path[0]
        # tiff_path = img_path[1]
        tiff_path = img_path[0]
        # img = Image.open(rgb_path).convert('RGB')
        # img = self.rgb_transform(img)
        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc)
        organized_pc_np = resized_organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = torch.from_numpy(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0)
        idx = fps(unorganized_pc_no_zeros, 4096)
        sparse_pcd = unorganized_pc_no_zeros[0,idx]
        sparse_pcd_np = sparse_pcd.numpy()
        sparse_pcd_np_normalize = point_operation.normalize_point_cloud(sparse_pcd_np)
        sparse_pcd_np_normalize = sparse_pcd_np_normalize[0]
          
        # return (0, sparse_pcd_np_normalize, resized_organized_pc, resized_depth_map_3channel, nonzero_indices), label
        return (0, sparse_pcd_np_normalize), label


class MVTec3DTest(MVTec3D):
    def __init__(self, class_name, img_size):
        super().__init__(split="test", class_name=class_name, img_size=img_size)
        self.gt_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                rgb_paths.sort()
                tiff_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))

                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img_original = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img_original)

        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc)
        organized_pc_np = resized_organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros_np = unorganized_pc[nonzero_indices, :]
        
        #Resample
        unorganized_pc_no_zeros = torch.from_numpy(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0)
        idx = fps(unorganized_pc_no_zeros, 4096)
        # sparse_pcd = unorganized_pc_no_zeros[0,idx].squeeze(dim=0)
        sparse_pcd = unorganized_pc_no_zeros[0,idx]
        sparse_pcd_np = sparse_pcd.numpy()
        sparse_pcd_np_normalize = point_operation.normalize_point_cloud(sparse_pcd_np)
        sparse_pcd_np_normalize = sparse_pcd_np_normalize[0]
        #-----
        unorganized_pc_no_zeros_np = np.expand_dims(unorganized_pc_no_zeros_np, axis=0) 
        pcd_np_normalize = point_operation.normalize_point_cloud(unorganized_pc_no_zeros_np)[0]
            

        if gt == 0:
            gt = torch.zeros(
                [1, resized_depth_map_3channel.size()[-2], resized_depth_map_3channel.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        return (img, sparse_pcd_np_normalize, resized_organized_pc, resized_depth_map_3channel, nonzero_indices, unorganized_pc_no_zeros_np), gt[:1], label, rgb_path, pcd_np_normalize
        # return (img, resized_organized_pc, resized_depth_map_3channel), gt[:1], label

def get_data_loader(split, class_name, img_size, batch_size=10):
    if split in ['train']:
        dataset = MVTec3DTrain(class_name=class_name, img_size=img_size)
    elif split in ['test']:
        dataset = MVTec3DTest(class_name=class_name, img_size=img_size)
    if split in ['train']:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True,
                             pin_memory=True)
    elif split in ['test']:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=False,
                             pin_memory=True)
    return dataset, data_loader
