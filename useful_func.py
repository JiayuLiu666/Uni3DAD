import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import proj3d
from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from ChamferDistancePytorch import fscore
from extensions.chamfer_dist import ChamferDistanceL1
from extensions.pointops.functions import pointops
import torch.nn as nn
from ChamferDistancePytorch.chamfer_python import distChamfer, distChamfer_raw
from scipy.spatial import KDTree
import torch.nn.functional as F
import open3d as o3d
from utils.mvtec3d_util import organized_pc_to_unorganized_pc

def plot_point_cloud(data, dim='2D',save_fig=False, plot_name=None):
    if dim == '2D':
        x = data[:, 0]
        y = data[:, 1]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        ax.scatter(x, y, s=5)
        if save_fig:
            plt.savefig(plot_name+'2D'+'.png')
        else:
            plt.show()

    elif dim == '3D':
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d'+'.png')

        ax.scatter(x, y, z)
        if save_fig:
            plt.savefig(plot_name+'2D'+'.png')
        else:
            plt.show()

        
    
    else:
        print('Error')
    
class DirectedHausdorff(object):
    """
    Hausdorf distance
    """
    def __init__(self, reduce_mean=True):
        # super(DirectedHausdorff,self).__init__()
        self.reduce_mean = reduce_mean
    
    def __call__(self, point_cloud1, point_cloud2):
        """
        :param point_cloud1: (B, 3, N)  partial
        :param point_cloud2: (B, 3, M) output
        :return: directed hausdorff distance, A -> B
        """
        n_pts1 = point_cloud1.shape[2]
        n_pts2 = point_cloud2.shape[2]

        pc1 = point_cloud1.unsqueeze(3)
        pc1 = pc1.repeat((1, 1, 1, n_pts2)) # (B, 3, N, M)
        pc2 = point_cloud2.unsqueeze(2)
        pc2 = pc2.repeat((1, 1, n_pts1, 1)) # (B, 3, N, M)

        l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1)) # (B, N, M)

        shortest_dist, _ = torch.min(l2_dist, dim=2)

        hausdorff_dist, _ = torch.max(shortest_dist, dim=1) # (B, )

        if self.reduce_mean:
            hausdorff_dist = torch.mean(hausdorff_dist)

        return hausdorff_dist
    
def calc_cd(output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False):
    # cham_loss = dist_chamfer_3D.chamfer_3DDist()
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))

    if separate:
        res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
               torch.cat([dist1.mean(1).unsqueeze(0),dist2.mean(1).unsqueeze(0)])]
    else:
        res = [cd_p, cd_t]
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2, 0.0001)
        res.append(f1)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res
    
def calc_dcd_full(x, gt, T=1000, n_p=1, return_raw=False, separate=False, return_freq=False, non_reg=False):
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True, separate=separate)
    exp_dist1, exp_dist2 = torch.exp(-dist1 * T), torch.exp(-dist2 * T)

    loss1 = []
    loss2 = []
    gt_counted = []
    x_counted = []

    for b in range(batch_size):
        count1 = torch.bincount(idx1[b])
        weight1 = count1[idx1[b].long()].float().detach() ** n_p
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1.append((- exp_dist1[b] * weight1 + 1.).mean())

        count2 = torch.bincount(idx2[b])
        weight2 = count2[idx2[b].long()].float().detach() ** n_p
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2.append((- exp_dist2[b] * weight2 + 1.).mean())

        if return_freq:
            expand_count1 = torch.zeros_like(idx2[b])  # n_x
            expand_count1[:count1.shape[0]] = count1
            x_counted.append(expand_count1)
            expand_count2 = torch.zeros_like(idx1[b])  # n_gt
            expand_count2[:count2.shape[0]] = count2
            gt_counted.append(expand_count2)

    loss1 = torch.stack(loss1)
    loss2 = torch.stack(loss2)
    loss = (loss1 + loss2) / 2

    if separate:
        res = [torch.cat([loss1.unsqueeze(0), loss2.unsqueeze(0)]), cd_p, cd_t]
    else:
        res = [loss, cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    if return_freq:
        x_counted = torch.stack(x_counted)
        gt_counted = torch.stack(gt_counted)
        res.extend([x_counted, gt_counted])
    return res
    

class LRScheduler(object):

    def __init__(self, optimizer, warm_up=0):
        super(LRScheduler, self).__init__()
        self.optimizer = optimizer
        self.warm_up = warm_up

    def update(self, iteration, learning_rate, num_group=1000, ratio=1):
        if iteration < self.warm_up:
            learning_rate *= iteration / self.warm_up
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = learning_rate * ratio**i
            
def binary_search(arr, num, epsilon=1e-5):
    low, high = 0, len(arr) - 1
    
    while low <= high:
        mid = (low + high) // 2
        
        if abs(arr[mid] - num) < epsilon:
            return mid
        elif arr[mid] < num:
            low = mid + 1
        else:
            high = mid - 1
            
    return low

def create_half_sphere_points(radius, total_points):
    # Estimate division for theta and phi
    phi_points = int(np.sqrt(total_points / 2))  # Roughly half the points for phi since it's a half-sphere
    theta_points = total_points // phi_points  # Adjust theta_points to maintain total

    # Define the range for angles, ensuring even distribution
    theta = np.linspace(0, 2 * np.pi, theta_points, endpoint=False)
    phi = np.linspace(0, np.pi / 2, phi_points, endpoint=True)

    # Meshgrid is not used here as we're directly computing points
    points = []
    for phi_val in phi:
        for theta_val in theta:
            x = radius * np.sin(phi_val) * np.cos(theta_val)
            y = radius * np.sin(phi_val) * np.sin(theta_val)
            z = radius * np.cos(phi_val)
            points.append((x, y, z))

    return np.array(points, dtype=np.float64)

def save_point_cloud_to_xyz(points, file_path):
    # Open the file at the specified path in write mode
    with open(file_path, 'w') as file:
        # Iterate over each point in the point cloud
        for point in points:
            # Write the x, y, and z values to the file, separated by spaces
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        center = pointops.fps(xyz, self.num_group)
        idx = pointops.knn(center, xyz, self.group_size)[0]
        neighborhood = pointops.index_points(xyz, idx)
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center
    
class DiscriminatorLoss(object):
    """
    feature distance from discriminator
    """
    def __init__(self, data_parallel=False):
        self.l2 = nn.MSELoss()
        self.data_parallel = data_parallel

    def __call__(self, D, fake_pcd, real_pcd):
        if self.data_parallel:
            with torch.no_grad():
                d, real_feature = nn.parallel.data_parallel(
                    D, real_pcd.detach())
            d, fake_feature = nn.parallel.data_parallel(D, fake_pcd)
        else:
            with torch.no_grad():
                real_feature = D(real_pcd.detach())
            fake_feature = D(fake_pcd)

        D_penalty = F.l1_loss(fake_feature, real_feature)
        return D_penalty

def farthest_point_sample(xyz, npoint):
    
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

def reorder_point_cloud(A, B, msk, tensor_out=True):
    # Build a KDTree for efficient nearest neighbor searches in N
    if len(A.shape) == 3 and len(B.shape) == 3:
        if not torch.is_tensor(B):
            pass
        else:
            B = B.clone().detach().cpu().numpy()
        if not torch.is_tensor(A):
            pass
        else:
            A = A.clone().detach().cpu().numpy()

        tree_N = KDTree(A[0][:,:2])
        
        # Find nearest neighbor in N for each point in A
        distances, indexes = tree_N.query(B[0][:,:2])
        
        # Reorder A based on the sorted order of the nearest neighbor indexes in N
        # This step assumes the goal is to sort A such that points with closest neighbors in N are ordered together
        # sorted_order = np.argsort(indexes)
        reordered_A = A[0][indexes]
        
        reordered_A_msk = msk[0][indexes]
        
        if tensor_out:
            reordered_A_tensor = torch.from_numpy(reordered_A).unsqueeze(dim=0).cuda()
            reordered_A_msk = reordered_A_msk.unsqueeze(dim=0).cuda()
        else:
            return reordered_A_tensor, torch.from_numpy(reordered_A_msk.unsqueeze(dim=0))
    else:
        if not torch.is_tensor(B):
            pass
        else:
            B = B.clone().detach().cpu().numpy()
        if not torch.is_tensor(A):
            pass
        else:
            A = A.clone().detach().cpu().numpy()
        tree_N = KDTree(A[:,:2])
        distances, indexes = tree_N.query(B[:,:2])
        reordered_A = A[indexes]
        reordered_A_msk = msk[0][indexes]
        if tensor_out:
            reordered_A_tensor = torch.from_numpy(reordered_A).cuda()
            reordered_A_msk = reordered_A_msk.unsqueeze(dim=0).cuda()
        else:
            return reordered_A, torch.from_numpy(reordered_A_msk.unsqueeze(dim=0))
    return reordered_A_tensor, reordered_A_msk

def get_fpfh_features(organized_pc, voxel_size=0.05):
    organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))

    radius_normal = voxel_size * 2
    o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
    (radius=radius_feature, max_nn=100))
    fpfh = pcd_fpfh.data.T
    full_fpfh = np.zeros((unorganized_pc.shape[0], fpfh.shape[1]), dtype=fpfh.dtype)
    full_fpfh[nonzero_indices, :] = fpfh
    full_fpfh_reshaped = full_fpfh.reshape((organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[1]))
    full_fpfh_tensor = torch.tensor(full_fpfh_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
    return full_fpfh_tensor

class kNNLoss(nn.Module):
    """
    Proposed PatchVariance component
    """
    def __init__(self, k=10, n_seeds=20):
        super(kNNLoss,self).__init__()
        self.k = k
        self.n_seeds = n_seeds
    def forward(self, pcs):
        n_seeds = self.n_seeds
        k = self.k
        seeds = farthest_point_sample(pcs,n_seeds) # which gives index
        seeds_value = torch.stack([pc[seed] for pc, seed in zip(pcs,seeds)]) 
        pcs_new = pcs.unsqueeze(2).repeat(1,1,n_seeds,1)
        seeds_new = seeds_value.unsqueeze(1).repeat(1,4096,1,1)
        dist = pcs_new.add(-seeds_new)
        dist_value = torch.norm(dist,dim=3)
        dist_new = dist_value.transpose(1,2)
        top_dist, idx = torch.topk(dist_new, k+1, dim=2, largest=False)
        overall_mean = top_dist[:,:,1:].mean()
        top_dist = top_dist/overall_mean
        var = torch.var(top_dist.mean(dim=2)).mean()
        return var

# class regional_aware_cd(object):
#     def __init__(self, opts):
#         super().__init__()
#         self.group_divider = Group(num_group = opts.num_group, group_size=opts.group_size)
#         self.nbr_ratio = opts.nbr_ratio
#         self.shape_criterion = ChamferDistanceL1()
        
#     def _group_points(self, nbrs, center, B, G):
#         nbr_groups = []
#         center_groups = []
#         perm = torch.randperm(G)
#         acc = 0
#         for i in range(1):
#             mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
#             mask[:, perm[acc:acc+self.mask_ratio[i]]] = True
#             nbr_groups.append(nbrs[mask].view(B, self.mask_ratio[i], self.group_size, -1))
#             center_groups.append(center[mask].view(B, self.mask_ratio[i], -1))
#             acc += self.mask_ratio[i]
#         return nbr_groups, center_groups
        
#     def get_loss(self, pts, pred):
#         # target point cloud
#         nbrs , center = self.group_divider(pts)  # neighborhood, center
#         B, G, _ = center.shape
#         nbr_groups, center_groups = self._group_points(nbrs, center, B, G)
#         rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        
#         # shape reconstruction loss
#         rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
#         idx = pointops.knn(center_groups[0], pred,  int(self.nbr_ratio * self.group_size))[0]
#         nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
#         shape_recon_loss = self.shape_recon_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()

#         return shape_recon_loss

        
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc        