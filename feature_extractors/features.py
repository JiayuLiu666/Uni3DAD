"""
PatchCore logic based on https://github.com/rvorias/ind_knn_ad
"""

from sklearn import random_projection
from utils.utils import KNNGaussianBlur
from utils.utils import set_seeds
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm
from utils.au_pro_util import calculate_au_pro
from models.models import Model
from models.pointnet2_utils import interpolating_points
from ptflops import get_model_complexity_info
from sklearn.decomposition import sparse_encode

class Features(torch.nn.Module):
    def __init__(self, opts, image_size=224, f_coreset=0.1, coreset_eps=0.9):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.deep_feature_extractor = Model(
                                device=self.device, 
                                xyz_backbone_name=opts.xyz_backbone_name, 
                                group_size = opts.group_size, 
                                num_group=opts.num_group
                                )
        self.deep_feature_extractor.to(self.device)
        self.opts = opts
        self.image_size = image_size
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.blur = KNNGaussianBlur(4)
        self.n_reweight = 3
        set_seeds(0)
        self.patch_lib = []
        self.s_map_lib = []
        self.resize = torch.nn.AdaptiveAvgPool2d((28, 28)) #change
        
        # anomaly score related
        # self.pixel_preds = list()
        # self.image_preds = list()
        # self.image_labels = list()
        # self.pixel_labels = list()
        # self.GAN_predictions = list()
        # self.gts = list()
        # self.predictions = list()
        # self.pro_score = list()
        # self.image_rocauc = 0
        # self.pixel_rocauc = 0
        # self.au_pro = 0

    def __call__(self, xyz):
        # Extract the desired feature maps using the backbone model.
        xyz = xyz.to(self.device)
        with torch.no_grad():
            xyz_feature_maps, center, ori_idx, center_idx = self.deep_feature_extractor(xyz)

        interpolate = True    
        if interpolate:
            interpolated_feature_maps = interpolating_points(xyz, center.permute(0,2,1), xyz_feature_maps).to("cpu")

        xyz_feature_maps = [fmap.to("cpu") for fmap in [xyz_feature_maps]]

        if interpolate:
            return xyz_feature_maps, center, ori_idx, center_idx, interpolated_feature_maps
        else:
            return xyz_feature_maps, center, ori_idx, center_idx

    def add_sample_to_mem_bank(self, sample):
        raise NotImplementedError

    def predict(self, sample, mask, label):
        raise NotImplementedError

    def compute_s_s_map(self, patch, feature_map_dims):
        dist = torch.cdist(patch, self.patch_lib)
        min_val, min_idx = torch.min(dist, dim=1)
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        s_map = self.blur(s_map)
        
        return s_map

    # def calculate_metrics(self):
    #     self.image_preds = np.stack(self.image_preds)
    #     self.image_labels = np.stack(self.image_labels)
    #     self.pixel_preds = np.array(self.pixel_preds)

    #     self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
    #     self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
    #     self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)

    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0)
        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_lib = self.patch_lib[self.coreset_idx]


    def get_coreset_idx_randomp(self, z_lib, n=1000, eps=0.90, float16=True, force_cpu=False):
        """Returns n coreset idx for given z_lib.
        Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
        CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)
        opts:
            z_lib:      (n, d) tensor of patches.
            n:          Number of patches to select.
            eps:        Agression of the sparse random projection.
            float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
            force_cpu:  Force cpu, useful in case of GPU OOM.
        Returns:
            coreset indices
        """

        print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
        try:
            transformer = random_projection.SparseRandomProjection(eps=eps)
            z_lib = torch.tensor(transformer.fit_transform(z_lib))
            print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
        except ValueError:
            print("   Error: could not project vectors. Please increase `eps`.")

        select_idx = 0
        last_item = z_lib[select_idx:select_idx + 1]
        coreset_idx = [torch.tensor(select_idx)]
        min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        # The line below is not faster than linalg.norm, although i'm keeping it in for
        # future reference.
        # min_distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True)

        if float16:
            last_item = last_item.half()
            z_lib = z_lib.half()
            min_distances = min_distances.half()
        if torch.cuda.is_available() and not force_cpu:
            last_item = last_item.to("cuda")
            z_lib = z_lib.to("cuda")
            min_distances = min_distances.to("cuda")

        for _ in range(n - 1):
            distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)  # broadcasting step
            min_distances = torch.minimum(distances, min_distances)  # iterative step
            select_idx = torch.argmax(min_distances)  # selection step

            # bookkeeping
            last_item = z_lib[select_idx:select_idx + 1]
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx.to("cpu"))
        return torch.stack(coreset_idx)


class SDF_Features(torch.nn.Module):
    def __init__(self, image_size, output_dir=None):
        super().__init__()
        # self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.image_size = image_size
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.blur = KNNGaussianBlur(4)
        set_seeds(0)
        self.resize = torch.nn.AdaptiveAvgPool2d((28, 28))
        self.image_rocauc = 0
        self.pixel_rocauc = 0

        # self.weight = 0
        # self.bias = 0
        # self.origin_f_map = []
        self.sdf_patch_lib = []
        # self.au_pro = 0
        # self.pixel_preds = list()
        # self.image_preds = list()
        # self.image_labels = list()
        # self.pixel_preds = list()
        # self.pixel_labels = list()
        # self.pixels_preds = list()
        # self.GAN_predictions = list()
        # self.gts = list()
        # self.predictions = list()
        # self.pro_score = list()


    def __call__(self, x):
        # Extract the desired feature maps using the backbone model.
        with torch.no_grad():
            feature_maps = self.rgb_feature_extractor(x)

        # feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        return feature_maps


    def count_your_model(model, input):
        flops, params = get_model_complexity_info(model, input, as_strings=True, print_per_layer_stat=True)
        print('Flops: ' + flops)
        print('Params: ' + params)

    def foreground_subsampling(self): 
        self.sdf_patch_lib = torch.cat(self.sdf_patch_lib, 0)
        self.rgb_patch_lib = torch.cat(self.rgb_patch_lib, 0)

        # Remove unused RGB features #
        self.origin_f_map = np.full(shape=(self.rgb_patch_lib.shape[0]),fill_value=-1)
        use_f_idices = torch.unique(torch.cat(self.rgb_f_idx_patch_lib,dim=0))
        self.rgb_patch_lib = self.rgb_patch_lib[use_f_idices]
        self.origin_f_map[use_f_idices] = np.array(range(use_f_idices.shape[0]),dtype=int)
        self.origin_f_map = torch.Tensor(self.origin_f_map).long()
            
    def Find_KNN_feature(self, feature, mode='testing'):
        Dict_features = []
        patch_lib = self.sdf_patch_lib.to(self.device)
        dist = torch.cdist(feature, patch_lib)
        _, knn_idx = torch.topk(dist, k=10+1, largest=False)

        knn_idx = knn_idx.cpu()
        if mode == 'alignment':
            knn_idx = knn_idx[:,1:]
        elif mode == 'testing':
            knn_idx = knn_idx[:,:-1]

        feature = feature.detach().to('cpu')
        for patch in range(knn_idx.shape[0]):
            Knn_features = self.sdf_patch_lib[knn_idx[patch]] #nearest patch in memory bank
            
            code_matrix = sparse_encode(X=feature[patch].view(1,-1), dictionary=Knn_features, algorithm='omp', n_nonzero_coefs=3, alpha=1e-10)
            
            code_matrix = torch.Tensor(code_matrix)
            sparse_feature = torch.matmul(code_matrix, Knn_features) # Sparse representation test rgb feature using the training rgb features stored in the memory.
            Dict_features.append(sparse_feature)
            
        Dict_features = torch.cat(Dict_features, 0)
        NN_feature = self.sdf_patch_lib[knn_idx[:, 0]]   # find the nearest neighbor feature
        pdist = torch.nn.PairwiseDistance(p=2, eps=1e-12)
        min_val = pdist(feature, Dict_features)
        s = torch.max(min_val) # Compute image level anomaly score #
        return NN_feature, Dict_features, knn_idx, s

    def run_coreset(self):
        self.sdf_patch_lib = torch.cat(self.sdf_patch_lib, 0)
        

    def cal_alignment(self):
        # SDF distribution
        sdf_map = np.array(self.sdf_pixel_preds)
        non_zero_indice = np.nonzero(sdf_map)
        non_zero_sdf_map = sdf_map[non_zero_indice]
        sdf_mean = np.mean(non_zero_sdf_map)
        sdf_std = np.std(non_zero_sdf_map)
        sdf_lower = sdf_mean - 3 * sdf_std
        sdf_upper = sdf_mean + 3 * sdf_std
        # RGB distribution
        rgb_map = np.array(self.rgb_pixel_preds)
        non_zero_indice = np.nonzero(rgb_map)
        non_zero_rgb_map = rgb_map[non_zero_indice]
        rgb_mean = np.mean(non_zero_rgb_map)
        rgb_std = np.std(non_zero_rgb_map)
        rgb_lower = rgb_mean - 3 * rgb_std
        rgb_upper = rgb_mean + 3 * rgb_std
        
        self.weight = (sdf_upper - sdf_lower) / (rgb_upper - rgb_lower)
        self.bias = sdf_lower - self.weight * rgb_lower
        # new_rgb_map = rgb_map * self.weight  + self.bias
        # total_score = np.maximum(new_rgb_map, sdf_map)
        # visualize_smap_distribute(total_score, sdf_map, rgb_map, new_rgb_map, self.image_size, output_dir)
        return self.weight, self.bias

    def cal_total_score(self, method='RGB_SDF'):

        if method == 'RGB':
            image_preds = np.stack(self.rgb_image_preds)
            pixel_preds = np.array(self.rgb_pixel_preds)
        elif method == 'SDF':
            image_preds = np.stack(self.sdf_image_preds)
            pixel_preds = np.array(self.sdf_pixel_preds)
        else:
            image_preds = np.stack(self.image_preds)
            pixel_preds = np.array(self.pixel_preds)

        gts = np.array(self.pixel_labels).reshape(-1, self.image_size, self.image_size)
        predictions = pixel_preds.reshape(-1, self.image_size, self.image_size)

        # visualize the distribution of image score and pixel score
        # if len(self.rgb_pixel_preds) != 0 and method == 'RGB_SDF':
        #     sdf_map = np.array(self.sdf_pixel_preds)
        #     rgb_map = np.array(self.rgb_pixel_preds)
        #     new_rgb_map = np.array(self.new_rgb_pixel_preds)
        #     sdf_s = np.array(self.sdf_image_preds)
        #     rgb_s = np.array(self.rgb_image_preds)
        #     label = np.array(self.image_labels)
            # visualize_image_s_distribute(sdf_s, rgb_s, label, output_dir)
            # visualize_smap_distribute(pixel_preds, sdf_map, rgb_map, new_rgb_map, self.image_size, output_dir)
        self.image_rocauc = roc_auc_score(self.image_labels, image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, pixel_preds)
        for pro_integration_limit in self.pro_limit:
            au_pro, _ = calculate_au_pro(gts, predictions, integration_limit=pro_integration_limit)
            self.au_pro[str(pro_integration_limit)] = au_pro

