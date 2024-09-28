import numpy as np
import torch
from torch.autograd import Variable
from ptflops import get_model_complexity_info
import torch.nn.functional as F
import torch.nn as nn


def pytorch_safe_norm(x, epsilon=1e-12, axis=None):
    return torch.sqrt(torch.sum(x ** 2, axis=axis) + epsilon)

# normal encoder with BN
class encoder_BN(nn.Module):
    def __init__(self):
        super(encoder_BN, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        return x  #1,POINT_NUM,128

## decoder with initial
class local_NIF(nn.Module):
    def __init__(self):
        super(local_NIF, self).__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(3, 512)
        self.fc3 = nn.Linear(512*2, 512)
        torch.nn.init.constant_(self.fc3.bias, 0.0)
        torch.nn.init.normal_(self.fc3.weight, 0.0, np.sqrt(2) / np.sqrt(512))

        for i in range(7):
            fc4 = nn.Linear(512, 512)
            torch.nn.init.constant_(fc4.bias, 0.0)
            torch.nn.init.normal_(fc4.weight, 0.0, np.sqrt(2) / np.sqrt(512))
            fc4 = nn.utils.weight_norm(fc4)
            setattr(self, "fc4" + str(i), fc4)

        self.fc5 = nn.Linear(512, 1)
        torch.nn.init.constant_(self.fc5.bias, -0.5)
        torch.nn.init.normal_(self.fc5.weight, mean=2*np.sqrt(np.pi) / np.sqrt(512), std=0.000001)

        #self.bn = nn.BatchNorm1d(512)

    def forward(self, points_feature, input_points):
        feature_f = F.relu(self.fc1(points_feature))
        net = F.relu(self.fc2(input_points))
        net = torch.concat([net, feature_f], dim=2)
        net = F.relu(self.fc3(net))
        for i in range(7):
            fc4 = getattr(self, "fc4" + str(i))
            net = F.relu(fc4(net))

        sdf = self.fc5(net)
        return sdf

    def get_gradient(self, points_feature, input_points):
        input_points.requires_grad_(True)
        sdf = self.forward(points_feature, input_points)
        gradient = torch.autograd.grad(
            sdf,
            input_points,
            torch.ones_like(sdf, requires_grad=False, device=sdf.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)
        normal_p_lenght = torch.unsqueeze(
            pytorch_safe_norm(gradient[0], axis=-1), -1)
        grad_norm = gradient[0] / normal_p_lenght
        g_point = input_points - sdf * grad_norm
        return g_point

class theta2(nn.Module):    
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.NIF = local_NIF()

    def forward(self, point_feature, input_points):
        g_points = self.NIF.get_gradient(point_feature, input_points)
        return g_points #input_points_3d - sdf * grad_norm

    def get_sdf(self, point_feature, input_points): #sdf
        return self.NIF(point_feature, input_points)

    def freeze_model(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        for p in self.NIF.parameters():
            p.requires_grad_(False)

class SDF_Model(nn.Module): #theta1 + theta2
    def __init__(self, point_num):
        super(SDF_Model, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = encoder_BN() #theta1
        self.NIF = local_NIF() #theta2
        self.point_num = point_num

    def forward(self, gt, input_points):
        gt = torch.permute(gt, (0, 2, 1))
        feature = self.encoder(gt)
        point_feature = torch.tile(torch.unsqueeze(
            feature, 1), [1, self.point_num, 1])
        g_points = self.NIF.get_gradient(point_feature, input_points)
        return g_points #input_points_3d - sdf * grad_norm

    def get_feature(self, point):
        point = torch.permute(point, (0, 2, 1))
        return self.encoder(point)

    def get_sdf(self, point_feature, input_points): #sdf
        return self.NIF(point_feature, input_points)

    def freeze_model(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        for p in self.NIF.parameters():
            p.requires_grad_(False)

class global_decoder(nn.Module): #theta3 + theta2
    def __init__(self, point_num, SHAPE_NUM=10):
        super().__init__()
        
        self.fc4 = nn.Linear(SHAPE_NUM, 512)
        self.fc5 = nn.Linear(3, 512)
        self.fc6 = nn.Linear(512*2, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, SHAPE_NUM)  #think
        self.fc10 = nn.Linear(SHAPE_NUM,128)
        self.fc9 = nn.Linear(512, 3)
        checkpoint = torch.load('/home/liuj35/Shape-Guided/checkpoint/best_ckpt/ckpt_000601.pth')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pretrain_sdf = SDF_Model(point_num)
        self.theta2 = theta2()
        self.pretrain_sdf.load_state_dict(checkpoint['sdf_model'])
        self.pretrain_dict =  {k: v for k, v in self.pretrain_sdf.state_dict().items() if k in self.theta2.state_dict()} 
        self.theta2.state_dict().update(self.pretrain_dict) 
        self.theta2.load_state_dict(self.theta2.state_dict()) 
        
    def forward(self, feature_global_f, input_points_3d_global_f):  
        '''
        input:
        -----
        feature_global_f:feature create manually: 1,500,10
        input_points_3d_global_f: randomly sample points from noise point cloud: 1,X,3
        
        output:
        -----
            g_points_g: gradient (vector)
            sdf_g: distance
        '''
        feature_g = F.relu(self.fc4(feature_global_f))
        net_g = F.relu(self.fc5(input_points_3d_global_f))
        net_g = torch.concat([net_g, feature_g], dim=2)
        net_g = F.relu(self.fc6(net_g))
        for i in range(8):
            net_g = self.fc7(F.relu(net_g))
             
        feature_output = self.fc8(F.relu(net_g)) #1,500,10
        
        
        feature_output = F.relu(self.fc10(feature_output))#1,500,128
        
        d_output = self.fc9(F.relu(net_g)) #1,500,3

        g_points_g = self.theta2(feature_output, input_points_3d_global_f + d_output)
        
        sdf_g = self.theta2.get_sdf(feature_output, input_points_3d_global_f + d_output)
        return g_points_g, sdf_g
    
    def freeze_model(self):
        for p in self.theta2.parameters():
            p.requires_grad_(False)



def prepare_input(resolution):
    x1 = torch.FloatTensor(1, 500, 3)
    x2 = torch.FloatTensor(1, 500, 3)
    return dict(gt = x1, input_points = x2)
 
if __name__ == '__main__':
    gt = Variable(torch.rand(32, 500, 3))
    sampled_point = Variable(torch.rand(32, 500, 3),  requires_grad=True)
    sdf_model = SDF_Model(128)
    print(sdf_model.get_feature(gt))

    # flops, params = get_model_complexity_info(sdf_model, (1, 500, 3), input_constructor=prepare_input, as_strings=True, print_per_layer_stat=True)
    # print('Flops: ' + flops)
    # print('Params: ' + params)
    ###
    # g = global_decoder(500,128)
    # feature_global_f = torch.rand(1,500,128)
    # input_points_3d_global_f = torch.rand(1,500,3)
    # g(feature_global_f,input_points_3d_global_f)
    # print('enter here')
