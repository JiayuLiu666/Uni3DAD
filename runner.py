import numpy as np
from gan_runner.btf import BTF_GanInversion
from gan_runner.mae import MAE_GanInversion
from gan_runner.shape_guided import Shape_GanInversion
from data.mvtec_dataset import get_data_loader
from utils.au_pro_util import calculate_au_pro
from data.data import get_data_loader_SDF
from Generation.config import opts
import torch
import os
import pandas as pd
import os.path as osp
import re

class Runner(object):
    def __init__(self, opts, data_choice, method):
        self.method_name = method
        self.data_choice = data_choice
        self.opts = opts
        #defind model
        if self.method_name == 'BTF+GAN':
            self.model = BTF_GanInversion(self.opts)
            self.train_loader = get_data_loader("train", class_name=self.data_choice, img_size=224, batch_size=1)
            self.valid_loader = get_data_loader('validation', class_name=self.data_choice, img_size=224, batch_size=1)
            self.test_loader = get_data_loader("test", class_name=self.data_choice, img_size=224, batch_size=1)
        elif self.method_name == 'MAE+GAN':
            self.model = MAE_GanInversion(self.opts)
            self.train_loader = get_data_loader("train", class_name=self.data_choice, img_size=224, batch_size=1)
            self.valid_loader = get_data_loader('validation', class_name=self.data_choice, img_size=224, batch_size=1)
            self.test_loader = get_data_loader("test", class_name=self.data_choice, img_size=224, batch_size=1)
        elif self.method_name == 'Shape+GAN':
            self.model = Shape_GanInversion(self.opts)
            self.train_loader = get_data_loader_SDF("train", class_name=self.data_choice, img_size=224, grid_path=self.opts.grid_path, shuffle=True)
            self.valid_loader = get_data_loader_SDF("validation", class_name = self.data_choice, img_size=224, grid_path=self.opts.grid_path)
            self.test_loader = get_data_loader_SDF("test", class_name = self.data_choice, img_size=224, grid_path=self.opts.grid_path)
            
    def train(self):
        #Train CORESET------------------------------
        with torch.no_grad(): #uncomment when use Shape+GAN
            for sample, _ in self.train_loader:
                self.model.add_sample_to_mem_bank(sample)
            self.model.run_coreset()
        
    def add_trains(self, model_name=None): #prepare to train OCSVM
        ### Train OCSVM
        i = 0
        # tmp_list = []
        saved_training_path = self.opts.saved_training + model_name #change to your own path!!
        if not os.path.isdir(saved_training_path):
            os.makedirs(saved_training_path)
        for sample, _ in self.train_loader:
            if i < 35:
                data = sample[1]
                self.model.reset_G()
                self.model.set_target(data)
                gan_output = self.model.do_inversion()
                
                _, gan_s_map = self.model.anomaly_detection(sample=data, 
                                            resized_organized_pt=sample[2], 
                                            nonzero_indices=sample[4].squeeze().numpy(),
                                            gan_output=gan_output) #1,224,224
                
                feature_s_map = self.model.train_predict(sample)
                # tmp_list.append(torch.cat((feature_s_map, gan_s_map)).permute(1,2,0).view(-1,2))
                
                np.savetxt(osp.join(saved_training_path, str(i) + '_train.txt'), torch.cat((feature_s_map, gan_s_map)).permute(1,2,0).view(-1,2).numpy(), delimiter=';')
            else:
                break
            i += 1
        
        
    def add_valids(self, model_name=None):
        path_list = []
        saved_valid_path = self.opts.saved_validing + model_name
        if not os.path.isdir(saved_valid_path):
            os.makedirs(saved_valid_path)
        for idx, batch in enumerate(self.valid_loader):
            sample, ground_truth, label, rgb_path, ori_sample = batch
            path_list.append(rgb_path) 
            self.model.reset_G()
            self.model.set_target(ori_sample)
            out_put_sample = self.model.do_inversion()
            
            anoscore_ft = self.model.predict(sample, ground_truth, label) #1,224,224
            anoscore_gan, full_anoscore_gan = self.model.anomaly_detection(sample=ori_sample, 
                                        resized_organized_pt=sample[2],  
                                        nonzero_indices=sample[4].squeeze().numpy(), 
                                        gan_output=out_put_sample,
                                        ground_truth=ground_truth) #1,224,224
            
            np.savetxt(os.path.join(saved_valid_path, str(idx) + '_valid.txt'), torch.cat((anoscore_ft, anoscore_gan)).permute(1,2,0).reshape(-1,2).numpy(), delimiter=';')


    def train_OCSVM(self, nu=0.5, quantile_ratio=None, model_name=None):
        ### Train OCSVM
        tmp_list = []
        saved_training_path = self.opts.saved_training + model_name + '/'
        
        all_files = os.listdir(saved_training_path)
        for files in all_files:
            tmp_list.append(torch.from_numpy(np.loadtxt(saved_training_path + files ,delimiter=';')))
        seg_fuser, k_coefficient = self.model.ocsvm_classifier(tmp_list, quantile=quantile_ratio, nu=nu)
        return seg_fuser, k_coefficient
    
    def re_train_OCSVM(self, nu=0.5, quantile_ratio=None, model_name=None):
        ### Train OCSVM
        tmp_list = []
        saved_training_path = self.opts.saved_training + model_name + '/'
        
        all_files = os.listdir(saved_training_path)
        for files in all_files:
            tmp_list.append(torch.from_numpy(np.loadtxt(saved_training_path + files ,delimiter=';')))
        self.seg_fuser, self.k_ratio = self.model.ocsvm_classifier(tmp_list, quantile=quantile_ratio, nu=nu)
        return self.k_ratio


    def run_valid_accuracy(self, OCSVM_model=None, model_name=None, k_coefficient=None):
        self.pro_score_valid = list()
        self.gt_valid = list()
        
        saved_valid = self.opts.saved_validing + model_name + '/'
        for idx, batch in enumerate(self.valid_loader):
            sample, ground_truth, label, rgb_path, ori_sample = batch #shape+GAN:1,1,224,224, MAE:1,1,224,224
            valid_ano_score_map = np.loadtxt(saved_valid + str(idx) + '_valid.txt', delimiter=';')
            anoscore_ft = torch.from_numpy(valid_ano_score_map[:,0].reshape(1,224,224))
            anoscore_gan = torch.from_numpy(valid_ano_score_map[:,1].reshape(1,224,224))
            
            self.model.cal_single_score_valid(anoscore_ft, 
                                              anoscore_gan, 
                                              OCSVM_model=OCSVM_model, 
                                              ground_truth=ground_truth,
                                              tmp_pre_list=self.pro_score_valid, 
                                              tmp_gt_list=self.gt_valid,
                                              k_ratio=k_coefficient)
                
        au_pro, _ = calculate_au_pro(self.gt_valid, self.pro_score_valid)
        return au_pro

    def class_valid_accuracy(self, OCSVM_model=None, model_name=None, k_ratio=None):
        pro_score_valid = list()
        gt_valid = list()
        
        saved_valid = self.opts.saved_validing + model_name + '/'
        for idx, batch in enumerate(self.valid_loader):
            sample, ground_truth, label, rgb_path, ori_sample = batch
            valid_ano_score_map = np.loadtxt(saved_valid + str(idx) + '_valid.txt', delimiter=';')
            anoscore_ft = torch.from_numpy(valid_ano_score_map[:,0].reshape(1,224,224))
            anoscore_gan = torch.from_numpy(valid_ano_score_map[:,1].reshape(1,224,224))
            
            self.model.cal_single_score_valid(anoscore_ft, 
                                              anoscore_gan, 
                                              OCSVM_model=OCSVM_model, 
                                              ground_truth=ground_truth,
                                              tmp_pre_list=pro_score_valid, 
                                              tmp_gt_list=gt_valid,
                                              k_ratio=k_ratio)
    
        au_pro, _ = calculate_au_pro(gt_valid, pro_score_valid)
        return au_pro

    
    def grid_search(self, model_name):     
        param_grid = np.linspace(0.9,1,21)
        best_accuracy = 0.0
        for param in param_grid:
            print(f"Current param: {param}")
            OCSVM_model, k_coefficient = self.train_OCSVM(quantile_ratio=param, model_name=model_name)
            accuracy = self.run_valid_accuracy(OCSVM_model=OCSVM_model, model_name=model_name, k_coefficient=k_coefficient)
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_quantile_ratio = param
                
        return best_quantile_ratio

    def test(self, model_name, seg_fuser, transform=False):
        path_list = []
        anoscore_list = []
        saved_testing = self.opts.saved_testing + model_name + '/' #change to your own path

        for idx, batch in enumerate(self.test_loader):
            sample, ground_truth, label, rgb_path, ori_sample = batch
            path_list.append(rgb_path) 
            
            test_ano_score_map = np.loadtxt(saved_testing + str(idx) + '_test.txt', delimiter=';')

            mask_test = ~((test_ano_score_map[:, 0] == 0) & (test_ano_score_map[:, 1] == 0))
            #debug
            anoscore_ft = torch.from_numpy(test_ano_score_map[:,0].reshape(1,224,224))

            anoscore_gan = torch.from_numpy(test_ano_score_map[:,1].reshape(1,224,224))

            self.model.cal_single_score(anoscore_ft * self.k_ratio,
                                        anoscore_gan, 
                                        seg_fuser, 
                                        ground_truth=ground_truth, 
                                        transform=transform)
            
        save_density = False
        save_FT_density = False
        save_GAN_density = False
        save_decision_boundary = False
        predict_maps = False
        
        if save_decision_boundary:
            save_density_path = '/home/liuj35/UNI3DAD/decision_boundary'
            if not os.path.isdir(save_density_path):
                os.makedirs(save_density_path)
            self.model.plot_decision_boundray(save_density_path, path_list)
        
        if save_FT_density:
            save_density_path = '/home/liuj35/UNI3DAD/density_PT_maps'
            if not os.path.isdir(save_density_path):
                os.makedirs(save_density_path)
            self.model.plot_PT_histogram(save_density_path, path_list)
        
        if save_GAN_density:
            save_density_path = '/home/liuj35/UNI3DAD/density_GAN_maps'
            if not os.path.isdir(save_density_path):
                os.makedirs(save_density_path)
            self.model.plot_GAN_histogram(save_density_path, path_list)
        
        if save_density:
            save_density_path = '/home/liuj35/SP-GAN/density_maps'
            if not os.path.isdir(save_density_path):
                os.makedirs(save_density_path)
            self.model.plot_histogram(save_density_path, path_list)
        
        if predict_maps:
            saved_maps_path = '/home/liuj35/SP-GAN/pred_maps'
            if not os.path.isdir(saved_maps_path):
                os.makedirs(saved_maps_path)
            self.model.save_prediction_maps(saved_maps_path, path_list)
    
    
    def add_tests(self, model_name, coefficient=None):
        path_list = []
        saved_testing_path = self.opts.saved_testing + model_name
        if not os.path.isdir(saved_testing_path):
            os.makedirs(saved_testing_path)
        for idx, batch in enumerate(self.test_loader):
            sample, ground_truth, label, rgb_path, ori_sample = batch
            path_list.append(rgb_path) 
            self.model.reset_G()
            self.model.set_target(ori_sample)
            out_put_sample = self.model.do_inversion()
            anoscore_ft = self.model.predict(sample, ground_truth, label) #1,224,224
            anoscore_gan, full_anoscore_gan = self.model.anomaly_detection(sample=ori_sample, 
                                        resized_organized_pt=sample[2],  
                                        nonzero_indices=sample[4].squeeze().numpy(), 
                                        gan_output=out_put_sample,
                                        ground_truth=ground_truth) #1,224,224
            
            np.savetxt(os.path.join(saved_testing_path, str(idx) + '_test.txt'), torch.cat((anoscore_ft, anoscore_gan)).permute(1,2,0).reshape(-1,2).numpy(), delimiter=';')
            
            # self.model.cal_single_score(anoscore_ft, anoscore_gan, ratio=coefficient, ground_truth=ground_truth, transform=False)
 
    def evaluate(self):
        pixel_rocaucs = dict()
        au_pros = dict()
        self.model.calculate_metrics()
        pixel_rocaucs[self.method_name] = round(self.model.pixel_rocauc, 3)
        au_pros[self.method_name] = round(self.model.au_pro, 3)
        print(
            f'Class: {opts.choice} AU-PRO: {self.model.au_pro:.3f} AUROC {self.model.pixel_rocauc:.3f}'
        )
        return au_pros, pixel_rocaucs
    
    def save_point_cloud(self, model_name):
        path_list = []
        saved_testing_path = '/home/liuj35/SP-GAN/saved_pt/'
        if not os.path.isdir(saved_testing_path):
            os.makedirs(saved_testing_path)
        for idx, batch in enumerate(self.test_loader):
            sample, ground_truth, label, rgb_path, ori_sample = batch
            path_list.append(rgb_path)
             
            class_dir = os.path.join(saved_testing_path, rgb_path[0].split('/')[-5])
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

            ad_dir = os.path.join(class_dir, rgb_path[0].split('/')[-3])
            if not os.path.exists(ad_dir):  
                os.mkdir(ad_dir)
            

            self.model.reset_G()
            self.model.set_target(ori_sample)
            out_put_sample = self.model.do_inversion()
            
            np.savetxt(os.path.join(ad_dir, str(idx) + '_ori_sample.txt'), ori_sample[0].cpu().detach().numpy(), delimiter=';')
            np.savetxt(os.path.join(ad_dir, str(idx) + '_reconstruction.txt'), out_put_sample[0].cpu().detach().numpy(), delimiter=';')

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    # model_choice = ['cable_gland','carrot','cookie','dowel','foam','peach','potato','rope','tire']
    
    model_choice = ['cable_gland']
    
    METHOD_NAMES = [opts.METHOD_NAME]
    print(f"Method name {METHOD_NAMES[0]}")
    pixel_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    au_pros_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    
    mean_list = list()
    for cls in range(len(model_choice)):
        opts.choice = model_choice[cls]
        runner = Runner(opts, opts.choice, method=METHOD_NAMES[0])
        
        # runner.save_point_cloud(model_name=opts.choice)
        runner.train()
        runner.add_trains(model_name =opts.choice)
        runner.add_valids(model_name=opts.choice)
        runner.add_tests(model_name=opts.choice)
        
        quantile_ratio = runner.grid_search(model_name = opts.choice)
        optimal_k_ratio = runner.re_train_OCSVM(quantile_ratio=quantile_ratio, model_name=opts.choice)
        class_accuracy_valid = runner.class_valid_accuracy(OCSVM_model=runner.seg_fuser, 
                                                model_name=opts.choice, 
                                                k_ratio=optimal_k_ratio) #au_pro score of the class
        mean_list.append(class_accuracy_valid)
        
        runner.test(model_name=opts.choice, seg_fuser=runner.seg_fuser)

        au_pros, pixel_rocaucs = runner.evaluate()
        pixel_rocaucs_df[opts.choice] = pixel_rocaucs_df['Method'].map(pixel_rocaucs)
        au_pros_df[opts.choice] = au_pros_df['Method'].map(au_pros)
        
    print(f"validation mean {np.mean(mean_list)}")
    pixel_rocaucs_df['Mean'] = round(pixel_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    au_pros_df['Mean'] = round(au_pros_df.iloc[:, 1:].mean(axis=1),3)
    print("\n\n################################################################################")
    print("############################# Pixel ROCAUC Results #############################")
    print("################################################################################\n")
    print(pixel_rocaucs_df.to_markdown(index=False))

    print("\n\n##########################################################################")
    print("############################# AU PRO Results #############################")
    print("##########################################################################\n")
    print(au_pros_df.to_markdown(index=False))