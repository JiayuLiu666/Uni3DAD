import os
import torch
from sklearn.kernel_approximation import Nystroem
from torch.autograd import Variable
import numpy as np
import useful_func as uf
from train_dist import SPGAN
from Generation.config import opts
from sklearn import linear_model
from copy import deepcopy
from emd_dist.emd_module import emdModule
from scipy.spatial import cKDTree
import cv2
from scipy.spatial.distance import cdist
from utils.au_pro_util import calculate_au_pro
from ChamferDistancePytorch.chamfer_python import distChamfer, distChamfer_raw
from sklearn.metrics import roc_auc_score
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from feature_extractors.features import SDF_Features
import seaborn as sns
from sklearn.pipeline import make_pipeline
#SDF related
from models.sdf_model import SDF_Model
from utils.utils import *
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import hinge_loss

from bayes_opt import BayesianOptimization
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class SDFFeature(object):
    def __init__(self, image_size, BS, POINT_NUM, ckpt_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.image_size = image_size
        self.BS = BS
        self.POINT_NUM = POINT_NUM

        # Launch the session
        self.sdf_model = SDF_Model(self.POINT_NUM).to(self.device)
        self.sdf_model.eval()
        self.blur = KNNGaussianBlur(4)
        # load check point
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.sdf_model.load_state_dict(checkpoint['sdf_model'])

    def get_feature(self, points_all, points_idx, split=None):
        
        total_feature = None

        
        for patch in range(len(points_all)):

            points = points_all[patch].reshape(-1, self.POINT_NUM, 3)
            indices = points_idx[patch].reshape(self.POINT_NUM)
            # extract sdf features
            feature = self.sdf_model.get_feature(points.to(self.device))
            
            if patch == 0:
                total_feature = feature
            else:
                total_feature = torch.cat((total_feature, feature), 0)
                
            
        return total_feature

    def get_score_map(self, feature, points_all, points_idx):

        s_map = np.full((self.image_size*self.image_size), 0, dtype=float)
        for patch in range(len(points_all)):

            if points_all[patch].shape[1] != self.POINT_NUM:
                print('Error!',points_all[patch].shape)
                continue
                    
            points = points_all[patch].reshape(-1, self.POINT_NUM, 3) #(1,500,3)
            indices = points_idx[patch].reshape(-1, self.POINT_NUM) #(1,500)

            point_feature = torch.tile(torch.unsqueeze(feature[patch], 0), [1, self.POINT_NUM, 1])
            index = indices[0].reshape(self.POINT_NUM)
            point_target = points[0,:].reshape(self.BS, self.POINT_NUM, 3)
            point_target = point_target.to(self.device)
            point_feature = point_feature.to(self.device)
            sdf_c = self.sdf_model.get_sdf(point_feature, point_target)

            sdf_c = np.abs(sdf_c.detach().cpu().numpy().reshape(-1))
            index = index.cpu().numpy()

            tmp_map = np.full((self.image_size*self.image_size), 0, dtype=float)
            for L in range(sdf_c.shape[0]):
                tmp_map[index[L]] = sdf_c[L]
                if(s_map[index[L]] == 0) or (s_map[index[L]] > sdf_c[L]):
                    s_map[index[L]] = sdf_c[L]

        s_map = s_map.reshape(1, 1, self.image_size, self.image_size)
        s_map = torch.tensor(s_map)
        sdf_map = self.blur(s_map)
        return sdf_map 

class Shape_GanInversion(SDF_Features):
    def __init__(self, opts):
        super().__init__(opts)
        #GAN Related
        self.emd = emdModule()
        self.opts = opts
        self.ball = None
        self.model = SPGAN(opts)
        checkpoint = torch.load('./pretrain_checkpoints/' + self.opts.choice + '.ckpt', map_location='cuda') 
        self.model.load_state_dict(checkpoint['state_dict'])
        self.G = self.model.G.to('cuda')
        self.G_weight = deepcopy(self.G.state_dict())
        self.G.eval()
        
        self.criterion = uf.DiscriminatorLoss()
        self.ftr_net = self.model.D.to('cuda')
        self.ftr_net.eval()
        self.G.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=self.opts.lr_g, betas=(0.5, 0.99)) # type: ignore
        self.G_scheduler = uf.LRScheduler(self.G.optim)
        self.update_G_stages = self.opts.update_G_stages
       
        self.z = torch.zeros((1, 4096, 128)).normal_(0,0.2).cuda()
        self.z = Variable(self.z, requires_grad=True)
        self.z_optim = torch.optim.Adam([self.z], lr=self.opts.z_lrs[0], betas=(0,0.99))  #change
        self.z_scheduler = uf.LRScheduler(self.z_optim)
        self.ball = None
        
        #SDF Related
        self.sdf = SDFFeature(self.opts.image_size, self.opts.BS, self.opts.POINT_NUM, self.opts.ckpt_path)
        
        
        #Anomaly Score related
        self.gts = list()
        self.predictions = list()
        self.PT_predictions = list()
        self.GAN_predictions = list()
        self.pro_score = list()
        self.pixel_labels = list()
        self.pixels_preds = list()
        self.au_pro = 0
        self.pixel_preds = list()

        self.X_test = list()


    def sphere_generator(self,bs=1,static=True):
        if self.ball is None:
            self.ball = np.loadtxt('template/balls/%d.xyz'%self.opts.np)[:,:3]
            self.ball = uf.pc_normalize(self.ball)
            N = self.ball.shape[0]
            # xx = torch.bmm(x, x.transpose(2,1))
            xx = np.sum(self.ball ** 2, axis=(1)).reshape(N, 1)
            yy = xx.T
            xy = -2 * xx @ yy  # torch.bmm(x, y.permute(0, 2, 1))
            self.ball_dist = xy + xx + yy  # [B, N, N]
        if static:
            ball = np.expand_dims(self.ball, axis=0)
            ball = np.tile(ball, (bs, 1, 1))
        else:
            ball = np.zeros((bs, self.opts.np, 3))
            for i in range(bs):
                idx = np.random.choice(self.ball.shape[0], self.opts.np)
                ball[i] = self.ball[idx]
        ball = Variable(torch.Tensor(ball)).cuda()

        return ball
    
    def _requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag
        
    def reset_G(self, pcd_id = None):
        self.G.load_state_dict(self.G_weight) 
        self.G.eval()


    def reset_z(self):
        self.z = torch.zeros((1, 4096, 128)).normal_(0,0.2).cuda()
        self.z = Variable(self.z, requires_grad=True)
        self.z_optim = torch.optim.Adam([self.z], lr=self.opts.z_lrs[0], betas=(0,0.99))  #change
        self.z_scheduler = uf.LRScheduler(self.z_optim)

    def read_ball(self,sort=False):
        x = np.loadtxt('template/balls/4096.xyz')
        ball = uf.pc_normalize(x)
        N = ball.shape[0]
        # xx = torch.bmm(x, x.transpose(2,1))
        xx = np.sum(x ** 2, axis=(1)).reshape(N, 1)
        yy = xx.T
        xy = -2 * xx @ yy  # torch.bmm(x, y.permute(0, 2, 1))
        dist = xy + xx + yy  # [B, N, N]
        return ball
    
    def do_inversion(self, ith=-1):
        self.reset_z()
        self.reset_G()
        curr_step = 0
        self._requires_grad(self.G, True)
        for stage, iter in enumerate(self.opts.iterations):
            for i in range(iter):
                curr_step += 1
                # Optimizer
                self.G_scheduler.update(curr_step, self.opts.G_lrs[stage])
                self.z_scheduler.update(curr_step, self.opts.z_lrs[stage])
                # self.M_scheduler.update(curr_step, self.opts.M_lrs[stage])
                self.z_optim.zero_grad()
                if self.update_G_stages[stage]:
                    self.G.optim.zero_grad()
                
                #Input Parameters Setting
                ball = self.read_ball()
                x = np.expand_dims(ball, axis=0)
                x = np.tile(x, (1, 1, 1))
                x = Variable(torch.Tensor(x),  requires_grad=False).cuda()
                out = self.G(x, self.z) #1,3,4096
                
                x_map = self.set_mask(self.target, out.transpose(2,1))
                nll = (self.z**2 / 2).mean()

                #----------------cd testing loss--------------------------------
                # ftr_loss = self.criterion(self.ftr_net, x_map.transpose(2,1), self.target.transpose(2,1))  #10-2
                _, dist2 , _, _ = distChamfer(out.transpose(2,1)[:,:,:2].cuda(), self.target[:,:,:2].cuda())
                dist1, _ , _, _ = distChamfer(x_map[:,:,:2].cuda(), self.target[:,:,:2].cuda())
                
                loss = torch.sum(dist1) + torch.sum(dist2) \
                        + 1.0*(self.ftr_net(out) - 1)**2 \
                        + 1 * nll
                
                #----------------EMD loss---------------------------------------
                # emd_loss , a = self.emd(out.transpose(2, 1).cuda(),self.target.cuda(),0.001,100)
                # loss =  cd_loss +  nll + torch.mean(emd_loss)
                #---------------------------------------------------------------
                
                #----------------CD(usefel) loss--------------------------------
                # loss , _ , _ = uf.calc_dcd_full(out.transpose(2, 1).cuda(), self.target.cuda())
                #---------------------------------------------------------------
                
                loss.backward()
                self.z_optim.step()
                
                if self.update_G_stages[stage]:
                    self.G.optim.step()

        self.out = out.transpose(2,1)
        
        return self.out

    def set_target(self, query):
        self.target = query.cuda()


    def set_mask(self, target, x):
        """
        masking based on CD.
        target: (1, N, 3), partial, can be < 4096, 4096, > 4096
        x: (1, 4096, 3)
        x_map: (1, N', 3), N' < 4096
        x_map: v1: 4096, 0 masked points
        """
        knn = self.opts.num_knn

        if knn == 1:
            cd1, cd2, argmin1, argmin2 = distChamfer(target, x) #matrix
            idx = torch.unique(argmin1).type(torch.long)
        elif knn > 1:
            # dist_mat shape (B, 4096, 4096), where B = 1
            dist_mat = distChamfer_raw(target, x)
            # indices (B, 4096, k)
            val, indices = torch.topk(dist_mat, k=knn, dim=2, largest=False)  #the smallest 5

            # union of all the indices
            idx = torch.unique(indices).type(torch.long)

        x_map = x[:, idx]
        return x_map
    
    def calculate_metrics(self):  
        self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
            
    def get_anomaly_area(self, detection_results): #return the clusters that contain incomplete shape  
        if np.count_nonzero(detection_results) > 0:
            detection_results_copy = np.zeros((224,224)).astype(np.float32)
            nonzero_x, nonzero_y = np.nonzero(detection_results)
            # Combine the x and y coordinates
            nonzero_coordinates = list(zip(nonzero_x, nonzero_y))
            X = np.array(nonzero_coordinates)
            db = DBSCAN(eps=8, min_samples=8).fit(X)
            labels = db.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            groups = {}
            for label, coord in zip(labels, X):
                if label in groups:
                    groups[label].append(coord)
                else:
                    groups[label] = [coord]

            # Convert lists to numpy arrays for convenience
            for label in groups:
                groups[label] = np.array(groups[label])
                
            dist_list_num = []
            for i in range(n_clusters_): #3
                # detection_results_copy[groups[i][:,0],groups[i][:,1]] = detection_results[groups[i][:,0],groups[i][:,1]]
                dist_list_num.append(np.max(detection_results[groups[i][:,0],groups[i][:,1]]))
            try:
                max_index_group = np.argmax(np.array(dist_list_num))
                detection_results_copy[groups[max_index_group][:,0],groups[max_index_group][:,1]]  = \
                    detection_results[groups[max_index_group][:,0],groups[max_index_group][:,1]]
                return detection_results_copy
            except ValueError:
                return detection_results
    
        else:
            return detection_results

    def clustering_anomaly_area(self, detection_results): #return the clusters that contain incomplete shape  
        detection_results_copy = np.zeros((224,224)).astype(np.float32)
        nonzero_x, nonzero_y = np.nonzero(detection_results)
        # Combine the x and y coordinates
        nonzero_coordinates = list(zip(nonzero_x, nonzero_y))
        X = np.array(nonzero_coordinates)
        db = DBSCAN(eps=8, min_samples=8).fit(X)
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        groups = {}
        for label, coord in zip(labels, X):
            if label in groups:
                groups[label].append(coord)
            else:
                groups[label] = [coord]

        # Convert lists to numpy arrays for convenience
        for label in groups:
            groups[label] = np.array(groups[label])
            

        for i in range(n_clusters_): #3
            detection_results_copy[groups[i][:,0],groups[i][:,1]] = detection_results[groups[i][:,0],groups[i][:,1]]
                        
        return detection_results_copy
    
    def point_cloud_2_image(self, pt_ano_score_map, nonzero_fg_coords, nonzero_indices, dist_backward):
        pixel_results = self.clustering_anomaly_area(pt_ano_score_map.reshape(224,224))    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        #
        dialated_pixel_s_map = self.opts.coefficient*cv2.dilate(pixel_results, kernel, iterations=3).reshape(-1) #convert to torch
        dialated_pixel_s_map[nonzero_indices] = dist_backward
        dialated_pixel_s_map = dialated_pixel_s_map.reshape(1,224,224)
          
        return dialated_pixel_s_map
    
    def add_pt_feature(self, sample):
        #add sample + compute ocsvm
        nonzero_indice = sample[4]
        fpfh_feature_maps = self.get_fpfh_features(sample[2])
        fpfh_feature_maps_resized = self.resize(self.average(fpfh_feature_maps))
        patch = fpfh_feature_maps_resized.reshape(fpfh_feature_maps_resized.shape[1], -1).T
        feature_s_map = self.compute_train_s_map(patch, fpfh_feature_maps_resized.shape[-2:]) #1,224,224
        return feature_s_map
    
    def ocsvm_classifier(self, s_map_lib, quantile=None, nu=0.5, transform=False):

        s_map_lib = torch.cat(s_map_lib, 0)       
        # self.X_train = s_map_lib
        
        mask = ~((s_map_lib[:, 0] == 0) & (s_map_lib[:, 1] == 0))
        
        # print(f"mean {torch.mean(s_map_lib[mask][:,1]) / torch.mean(s_map_lib[mask][:,0])}")
        
        # print(f'weight {torch.std(self.X_train[mask][:,1]) / torch.std(self.X_train[mask][:,0])}')
        
        # print(f'bias {torch.mean(self.X_train[mask][:,1]) - torch.std(self.X_train[mask][:,1]) / torch.std(self.X_train[mask][:,0]) * torch.mean(self.X_train[mask][:,0])}')
        
        # print(f"max {max(s_map_lib[mask][:,1]) / max(s_map_lib[mask][:,0])}")
        
        # print('quantile GAN', torch.quantile(self.X_train[:,1], 0.8, interpolation='nearest'))
        
        # print('quantile Feature', torch.quantile(self.X_train[:,0], 0.8, interpolation='nearest'))
        
        k_ratio = torch.quantile(s_map_lib[mask][:,1], quantile, interpolation='nearest') / torch.quantile(s_map_lib[mask][:,0], quantile, interpolation='nearest')
        
        s_map_lib[:, 0] = s_map_lib[:, 0] * k_ratio
        
        # print(f'{k * 100}% quantile {ratio}')
                
        clf_sgd = linear_model.SGDOneClassSVM(nu=nu, random_state=42)
        if transform:
            transform = Nystroem(gamma=2, random_state=42)
            seg_fuser = make_pipeline(transform, clf_sgd)
        else:
            seg_fuser = make_pipeline(clf_sgd)
        seg_fuser.fit(s_map_lib[mask])
        # print('adjust point', seg_fuser.score_samples(np.array([0,0]).reshape(-1,2)))
        return seg_fuser, k_ratio
        
    def anomaly_detection(self, sample, resized_organized_pt, nonzero_indices, gan_output, ground_truth=None):
        '''s
        input ->
        sample: to be detected target point cloud
        resized_organized_pt: 
        nonzero_indices: 
        ground_truth:
        
        output:
        distances_backward: anomaly score
        '''
        self.resized_organized_pt = resized_organized_pt
        self.nonzero_indices = nonzero_indices
        # self.ground_truth = ground_truth.squeeze()
        
        test_near_xy = []
        rec_sample_xy = gan_output[0].detach().cpu().numpy()[:,:2]
        test_sample_xy = sample[0].detach().cpu().numpy()[:,:2]
        sample_x_y = sample[0].detach().clone().cpu().numpy()[:,:2]
        
        rec_kd_tree = cKDTree(rec_sample_xy)
        distances_backward, vertex_ids_backward = rec_kd_tree.query(test_sample_xy, p=2, k=1) 
        #distances_backward: every point on test_sample_xy to reconstruction sample, shape: same as test sample
        vertex_ids_backward = np.asarray(vertex_ids_backward) #subset of self.out, the nearest distance from test sample to rec sample (regular grid)
        
        # print(distances_backward.shape) 16470

        test_near_xy.append(rec_sample_xy[vertex_ids_backward].reshape(-1,2)) #lots of repeated points, build from reconstruction pt
        test_near_xy = np.squeeze(np.asarray(test_near_xy),axis=0)
        test_near_kd_tree = cKDTree(test_near_xy) #same as test sample
        d_backward, v_backward = test_near_kd_tree.query(rec_sample_xy, p=2, k=1)
        defect_pt = rec_sample_xy[np.nonzero(d_backward)[0]] #covered area still has points
        
        ## pixel range question?
        ori_x_coordinate = sample_x_y[:,0] #original sample
        ori_y_coordinate = sample_x_y[:,1]
        
        defect_x = defect_pt[:,0] #defect points
        defect_y = defect_pt[:,1]
        
        pixel_y_range = np.floor(self.nonzero_indices / 224).astype(np.int64)
        pixel_x_range = (self.nonzero_indices - 224 * pixel_y_range).astype(np.int64)
        
        min_x_idx, min_y_idx =np.argmin(pixel_x_range), np.argmin(pixel_y_range)
        max_x_idx, max_y_idx =np.argmax(pixel_x_range), np.argmax(pixel_y_range)
        
        min_x_pixel, max_x_pixel = pixel_x_range[min_x_idx], pixel_x_range[max_x_idx]
        min_y_pixel, max_y_pixel = pixel_y_range[min_y_idx], pixel_y_range[max_y_idx]
        
        pix_x_num_interval = max_x_pixel - min_x_pixel
        pix_y_num_interval = max_y_pixel - min_y_pixel

        min_x, max_x = np.min(ori_x_coordinate), np.max(ori_x_coordinate) #point cloud coordinates
        min_y, max_y = np.min(ori_y_coordinate), np.max(ori_y_coordinate)
                
        # min_rec_x, max_rec_x = np.min(defect_x), np.max(defect_x) #reconstruction point cloud coordinates
        # min_rec_y, max_rec_y = np.min(defect_y), np.max(defect_y)
                

        new_x_range, new_y_range = np.linspace(min_x, max_x, pix_x_num_interval) \
            ,np.linspace(min_y, max_y, pix_y_num_interval) #coordinate system change
        
        new_pix_x_range, new_pix_y_range = np.linspace(min_x_pixel,max_x_pixel,int(pix_x_num_interval+1)).astype(np.int64) \
                                            ,np.linspace(min_y_pixel,max_y_pixel,int(pix_y_num_interval+1)).astype(np.int64) # pixel system
                                            
        ##Problem**
        x_lab, y_lab = [],[]
        for i in range(len(defect_x)):
            y_lab.append(uf.binary_search(new_y_range, defect_y[i]))
            x_lab.append(uf.binary_search(new_x_range, defect_x[i]))
            
        xx = new_pix_x_range[x_lab]
        yy = new_pix_y_range[y_lab]
        
        min_val = np.min(cdist(defect_pt, test_sample_xy),axis=1)
        results = torch.zeros((1,224,224)).to(torch.float32)
        
        for i in range(len(xx)):
            results[:,yy[i], xx[i]] = min_val[i]
            
        anomaly_mask = results.numpy().reshape(224,224)  

        # pixel_results_areas = results.numpy().reshape(-1) #50176                 
        ### Calculate Anomaly Score    
        nonzero_defects_coords = np.column_stack(np.where(anomaly_mask > 0.0))
        nonzero_fg_coords = np.column_stack(np.where(self.resized_organized_pt.squeeze()[2,:,:].numpy() > 0.0))
        min_dist = np.min(cdist(nonzero_defects_coords, nonzero_fg_coords), axis=1)
        
        for i in range(nonzero_defects_coords.shape[0]):
            if i in np.where(min_dist > 0.0)[0]:
                pass
            else:
                anomaly_mask[nonzero_defects_coords[i,0], nonzero_defects_coords[i,1]] = 0.0
        
     
        pixel_results_areas = anomaly_mask.reshape(-1)
        if np.count_nonzero(pixel_results_areas) > 0.0:
            full_defects_map = self.point_cloud_2_image(pt_ano_score_map=pixel_results_areas,
                                                        nonzero_fg_coords=nonzero_fg_coords,
                                                        nonzero_indices=self.nonzero_indices,
                                                        dist_backward=distances_backward)
            
            #Calculate Anomaly area    
            pixel_results = self.get_anomaly_area(pixel_results_areas.reshape(224,224))    
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            
            ##OCSVM
            dialated_pixel_s_map = self.opts.coefficient * cv2.dilate(pixel_results, kernel, iterations=3).reshape(-1) #convert to torch
            dialated_pixel_s_map[self.nonzero_indices] = distances_backward
            pixel_s_map = dialated_pixel_s_map.reshape(1,224,224)
            return self.blur(torch.from_numpy(pixel_s_map)), self.blur(torch.from_numpy(full_defects_map))
            
        else:
            pixel_results_areas[self.nonzero_indices] = distances_backward
            pixel_s_map = pixel_results_areas.reshape(1,224,224)
            # pixel_s_map_zero = np.zeros(1,224,224)
            
            #OCSVM
            return self.blur(torch.from_numpy(pixel_results_areas)), self.blur(torch.from_numpy(pixel_results_areas))


    def train_predict(self, sample, mask=None, label=None):
        feature = self.sdf.get_feature(sample[5], sample[6])
        NN_feature, Dict_features, lib_idices, sdf_s = self.Find_KNN_feature(feature)
        sdf_map = self.sdf.get_score_map(Dict_features, sample[5], sample[6]) #(Patch num, 128), sample[1]:(patch,500,3)
        return sdf_map
    
    def cal_single_score_valid(self, s_map_point, pixel_s_map, OCSVM_model, ground_truth, tmp_pre_list, tmp_gt_list, k_ratio=None, mask=None, transform=False):
        if transform:
            ground_truth = ground_truth.squeeze()
            ##OCSVM        
            final_result = torch.cat((s_map_point * k_ratio, pixel_s_map)).permute(1,2,0).reshape(-1,2)
            self.X_test.append(final_result)
            bias = self.seg_fuser.score_samples(np.array([0,0]).reshape(1,-1))
            final_results = abs(torch.from_numpy(self.seg_fuser.score_samples(final_result).reshape(1,224,224)) - bias)
            ##
            self.pixel_preds.extend(final_results.flatten().numpy())
            self.pixel_labels.extend(self.ground_truth.flatten().numpy())
            self.gts.append(self.ground_truth.numpy())
            self.predictions.append(final_results.squeeze().numpy())
            self.GAN_predictions.append((pixel_s_map).squeeze().numpy())
            self.PT_predictions.append((s_map_point).squeeze().numpy())
            
            try:
                proscore, _ = calculate_au_pro([self.ground_truth.numpy()], [final_results.squeeze().numpy()])
                self.pro_score.append(proscore)
            except ZeroDivisionError:
                self.pro_score.append(0)
        else:
            ground_truth = ground_truth.squeeze()
            
            ##Weight------------------------------------------------------------------------------------------
            # s_map_point_flatt = torch.flatten(s_map_point)
            # pixel_s_map_flatt = torch.flatten(pixel_s_map)
            # weight = torch.std(pixel_s_map_flatt[mask]) / torch.std(s_map_point_flatt[mask])
            # bias = torch.mean(pixel_s_map_flatt[mask]) - torch.std(pixel_s_map_flatt[mask]) / torch.std(s_map_point_flatt[mask]) * torch.mean(s_map_point_flatt[mask])
            # final_results = (weight * s_map_point_flatt + bias + pixel_s_map_flatt).reshape(1,224,224)
            # self.pixel_preds.extend(final_results.flatten().numpy())
            # self.pixel_labels.extend(self.ground_truth.flatten().numpy())
            # self.gts.append(self.ground_truth.numpy())
            # self.predictions.append(final_results.squeeze().numpy())
            # self.GAN_predictions.append((pixel_s_map).squeeze().numpy())
            # self.PT_predictions.append((s_map_point).squeeze().numpy())
            # try:
            #     proscore, _ = calculate_au_pro([self.ground_truth.numpy()], [final_results.squeeze().numpy()])
            #     self.pro_score.append(proscore)
            # except ZeroDivisionError:
            #     self.pro_score.append(0)
                    
            ##OCSVM--------------------------------------------------------------------------------------------------------------------------
            final_result = torch.cat((s_map_point * k_ratio, pixel_s_map)).permute(1,2,0).reshape(-1,2)
            final_results = torch.from_numpy(OCSVM_model.score_samples(final_result).reshape(1,224,224))
            tmp_gt_list.append(ground_truth.numpy())
            tmp_pre_list.append(final_results.squeeze().numpy())

    def cal_single_score(self, s_map_point, pixel_s_map, seg_fuser, ground_truth, coefficient=None, mask=None, transform=False):
        if transform:
            self.ground_truth = ground_truth.squeeze()
            
                    
            ##OCSVM        
            final_result = torch.cat((s_map_point, pixel_s_map)).permute(1,2,0).reshape(-1,2)
            self.X_test.append(final_result)
            
            bias = seg_fuser.score_samples(np.array([0,0]).reshape(1,-1))
            final_results = abs(torch.from_numpy(seg_fuser.score_samples(final_result).reshape(1,224,224)) - bias)
            
            ##
            self.pixel_preds.extend(final_results.flatten().numpy())
            self.pixel_labels.extend(self.ground_truth.flatten().numpy())
            self.gts.append(self.ground_truth.numpy())
            self.predictions.append(final_results.squeeze().numpy())
            self.GAN_predictions.append((pixel_s_map).squeeze().numpy())
            self.PT_predictions.append((s_map_point).squeeze().numpy())
            
            try:
                proscore, _ = calculate_au_pro([self.ground_truth.numpy()], [final_results.squeeze().numpy()])
                self.pro_score.append(proscore)
            except ZeroDivisionError:
                self.pro_score.append(0.0)
        else:
            self.ground_truth = ground_truth.squeeze()
            
            ##Weight------------------------------------------------------------------------------------------
            # s_map_point_flatt = torch.flatten(s_map_point)
            # pixel_s_map_flatt = torch.flatten(pixel_s_map)
            # weight = torch.std(pixel_s_map_flatt[mask]) / torch.std(s_map_point_flatt[mask])
            # bias = torch.mean(pixel_s_map_flatt[mask]) - torch.std(pixel_s_map_flatt[mask]) / torch.std(s_map_point_flatt[mask]) * torch.mean(s_map_point_flatt[mask])
            # final_results = (weight * s_map_point_flatt + bias + pixel_s_map_flatt).reshape(1,224,224)
            # self.pixel_preds.extend(final_results.flatten().numpy())
            # self.pixel_labels.extend(self.ground_truth.flatten().numpy())
            # self.gts.append(self.ground_truth.numpy())
            # self.predictions.append(final_results.squeeze().numpy())
            # self.GAN_predictions.append((pixel_s_map).squeeze().numpy())
            # self.PT_predictions.append((s_map_point).squeeze().numpy())
            # try:
            #     proscore, _ = calculate_au_pro([self.ground_truth.numpy()], [final_results.squeeze().numpy()])
            #     self.pro_score.append(proscore)
            # except ZeroDivisionError:
            #     self.pro_score.append(0)
                    
            ##OCSVM--------------------------------------------------------------------------------------------------------------------------
            final_result = torch.cat((s_map_point, pixel_s_map)).permute(1,2,0).reshape(-1,2)
            self.X_test.append(final_result)
            final_results = torch.from_numpy(seg_fuser.score_samples(final_result).reshape(1,224,224))
            self.pixel_preds.extend(final_results.flatten().numpy())
            self.pixel_labels.extend(self.ground_truth.flatten().numpy())
            self.gts.append(self.ground_truth.numpy())
            self.predictions.append(final_results.squeeze().numpy())
            self.GAN_predictions.append((pixel_s_map).squeeze().numpy())
            self.PT_predictions.append((s_map_point).squeeze().numpy())
            try:
                proscore, _ = calculate_au_pro([self.ground_truth.numpy()], [final_results.squeeze().numpy()])
                self.pro_score.append(proscore)
            except ZeroDivisionError:
                self.pro_score.append(0)
    
        
      
    def add_sample_to_mem_bank(self, sample):
        
        sdf_feature = self.sdf.get_feature(sample[5], sample[6], split='train') #enter 1 bagel patches
        self.sdf_patch_lib.append(sdf_feature.to('cpu'))
    
    
    def predict(self, sample, mask=None, label=None):
        feature = self.sdf.get_feature(sample[6], sample[7])
        NN_feature, Dict_features, lib_idices, sdf_s = self.Find_KNN_feature(feature)
        sdf_map = self.sdf.get_score_map(Dict_features, sample[6], sample[7]) #(Patch num, 128), sample[1]:(patch,500,3)
        return sdf_map
    
    def plot_decision_boundray(self, output_path, rgb_path, save_num=5):
        for i in range(max(save_num, len(self.predictions))):
            _, ax = plt.subplots(figsize=(9, 6))
            xx, yy = np.meshgrid(np.linspace(0, 0.5, 100), np.linspace(0, 0.3, 100))
            X = np.concatenate([xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)], axis=1)

            DecisionBoundaryDisplay.from_estimator(
                self.seg_fuser,
                X,
                response_method="decision_function",
                plot_method="contourf",
                ax=ax,
                cmap="PuBu",
            )

            DecisionBoundaryDisplay.from_estimator(
                self.seg_fuser,
                X,
                response_method="decision_function",
                plot_method="contour",
                ax=ax,
                linewidths=2,
                colors="darkred",
                levels=[0],
            )
            
            DecisionBoundaryDisplay.from_estimator(
                self.seg_fuser,
                X,
                response_method="decision_function",
                plot_method="contourf",
                ax=ax,
                colors="palevioletred",
                levels=[0, self.seg_fuser.decision_function(X).max()],
            )
            
            s = 5
            
            mask = ~((self.X_train[:, 0] == 0.0) & (self.X_train[:, 1] == 0.0))
            b1 = ax.scatter(self.X_train[mask][:,0], self.X_train[mask][:,1], c="green", s=s)
            b2 = ax.scatter(self.X_test[i][:,0], self.X_test[i][:,1], c="blueviolet", s=s)
            
            ax.set(
                title="One-Class SVM",
                xlim=(0, 0.5),
                ylim=(0, 0.3),
                ylabel=('GAN Anomaly Score'
            ),
                xlabel=('Feature Based Anomaly Score')
            )
            # ax.legend(
            #     [mlines.Line2D([], [], color="darkred", label="learned frontier"), b1, b2],
            #     [
            #         "learned frontier",
            #         "training observations",
            #         "new regular observations",
            #         "new abnormal observations",
            #     ],
            #     loc="upper left",
            # )
            
            class_dir = os.path.join(output_path, rgb_path[i][0].split('/')[-5])
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

            ad_dir = os.path.join(class_dir, rgb_path[i][0].split('/')[-3])
            if not os.path.exists(ad_dir):  
                os.mkdir(ad_dir)
            
            plt.savefig(os.path.join(ad_dir,  str(round(self.pro_score[i] * 100, 3)) + '_pred_' + rgb_path[i][0].split('/')[-1] + '.jpg'))
            plt.close()

    def plot_PT_histogram(self, output_path, rgb_path, save_num=5):
        
        for i in range(max(save_num, len(self.PT_predictions))): 
            nonzero_indices = np.nonzero(self.PT_predictions[i].flatten())
            non_zero_s_map = self.PT_predictions[i].flatten()[nonzero_indices]
            
            fig, ax = plt.subplots(figsize=(10,10))   
            sns.histplot(non_zero_s_map, bins=40, kde=False, color='b', ax=ax, alpha=0.6, stat='density')
            sns.kdeplot(data=non_zero_s_map, color='crimson', ax=ax)
            ax.set_title('Histogram of Anomaly Score', fontsize=18)
            ax.set_xlabel('Point Features Anomaly Score', fontsize=18, fontfamily='sans-serif', fontstyle='italic')
            ax.set_ylabel('Density', fontsize='x-large', fontstyle='oblique')
                        
            class_dir = os.path.join(output_path, rgb_path[i][0].split('/')[-5])
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

            ad_dir = os.path.join(class_dir, rgb_path[i][0].split('/')[-3])
            if not os.path.exists(ad_dir):
                os.mkdir(ad_dir)
            
            fig.savefig(os.path.join(ad_dir,  str(round(self.pro_score[i] * 100, 3)) + '_pred_' + rgb_path[i][0].split('/')[-1] + '.jpg'))
            plt.close(fig)  

    def plot_GAN_histogram(self, output_path, rgb_path, save_num=5):
        for i in range(max(save_num, len(self.GAN_predictions))): 
            nonzero_indices = np.nonzero(self.GAN_predictions[i].flatten())
            non_zero_s_map = self.GAN_predictions[i].flatten()[nonzero_indices]
            fig, ax = plt.subplots(figsize=(10,10))   
            
            sns.histplot(non_zero_s_map, bins=40, kde=False, color='b', ax=ax, alpha=0.6, stat='density')
            sns.kdeplot(data=non_zero_s_map, color='crimson', ax=ax)
            ax.set_title('Histogram of Anomaly Score', fontsize=18)
            ax.set_xlabel('GAN Anomaly Score', fontsize=18, fontfamily='sans-serif', fontstyle='italic')
            ax.set_ylabel('Density', fontsize='x-large', fontstyle='oblique')
                        
            class_dir = os.path.join(output_path, rgb_path[i][0].split('/')[-5])
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

            ad_dir = os.path.join(class_dir, rgb_path[i][0].split('/')[-3])
            if not os.path.exists(ad_dir):
                os.mkdir(ad_dir)
            
            fig.savefig(os.path.join(ad_dir,  str(round(self.pro_score[i] * 100, 3)) + '_pred_' + rgb_path[i][0].split('/')[-1] + '.jpg'))
            plt.close(fig)  
            
            
    def plot_histogram(self, output_path, rgb_path, save_num=5):
        for i in range(max(save_num, len(self.predictions))): 
            nonzero_indices = np.nonzero(self.predictions[i].flatten())
            non_zero_s_map = self.predictions[i].flatten()[nonzero_indices]
            fig, ax = plt.subplots(figsize=(10,10))   
            
            sns.histplot(non_zero_s_map, bins=40, kde=True, color='b', ax=ax, alpha=0.6, stat='density')
            sns.kdeplot(data=non_zero_s_map, color='crimson', ax=ax)
            ax.set_title('Histogram of Anomaly Score', fontsize=18)
            ax.set_xlabel('GAN Anomaly Score', fontsize=18, fontfamily='sans-serif', fontstyle='italic')
            ax.set_ylabel('Density', fontsize='x-large', fontstyle='oblique')
                        
            class_dir = os.path.join(output_path, rgb_path[i][0].split('/')[-5])
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

            ad_dir = os.path.join(class_dir, rgb_path[i][0].split('/')[-3])
            if not os.path.exists(ad_dir):
                os.mkdir(ad_dir)
            
            fig.savefig(os.path.join(ad_dir,  str(round(self.pro_score[i] * 100, 3)) + '_pred_' + rgb_path[i][0].split('/')[-1] + '.jpg'))
            plt.close(fig) 
    
    # def save_anomaly_score_maps(self, input_list, output_dir, save_num=5):
    #     for i in range(len(input_list)):
                    
    #         ad_dir_test = os.path.join(class_dir_test, rgb_path[i][0].split('/')[-3])
    #         if not os.path.exists(ad_dir_test):
    #             os.mkdir(ad_dir_test)


    #         np.savetxt(os.path.join(ad_dir_test, str(i) + '_test.txt'), input_list[i].numpy(), delimiter=';')
            
        
    def save_prediction_maps(self, output_path, rgb_path, save_num=5):
        for i in range(max(save_num, len(self.predictions))):
            # fig = plt.figure(dpi=300)
            fig = plt.figure(figsize=(8,8))
        
            ax3 = fig.add_subplot(1,3,1)
            rgb = plt.imread(rgb_path[i][0])    
            ax3.imshow(rgb)


            ax2 = fig.add_subplot(1,3,2)
            ax2.imshow(self.gts[i], cmap=plt.cm.gray)
            
            ax = fig.add_subplot(1,3,3)
            ax.imshow(self.predictions[i])

            
            class_dir = os.path.join(output_path, rgb_path[i][0].split('/')[-5])
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

            ad_dir = os.path.join(class_dir, rgb_path[i][0].split('/')[-3])
            if not os.path.exists(ad_dir):  
                os.mkdir(ad_dir)
            
            plt.savefig(os.path.join(ad_dir,  str(round(self.pro_score[i] * 100, 3)) + '_pred_' + rgb_path[i][0].split('/')[-1] + '.jpg'))
            plt.close()
    
    # def objective_function(self, k, s_map_lib, nu=0.8):
    #     s_map_lib = torch.cat(s_map_lib, 0)       
    #     clf_sgd = linear_model.SGDOneClassSVM(nu=nu, random_state=42)
    #     self.seg_fuser = make_pipeline(clf_sgd)
    #     self.seg_fuser.fit(s_map_lib)

    #     sample_score = self.seg_fuser.score_samples(s_map_lib)
    #     label = np.ones_like(sample_score)
    #     return hinge_loss(label, sample_score)

    
    # def run_bayesian(self):
    #     pbounds = {
    #         'k': (0, 10),
    #     }

    #     optimizer = BayesianOptimization(
    #         f=self.objective_function,
    #         pbounds=pbounds,
    #         random_state=42
    #     )

    #     optimizer.maximize(
    #         init_points=1,  # Random exploratory steps
    #         n_iter=10       # Steps of Bayesian Optimization
    #     )
        
    #     return optimizer.max['param']
        

        