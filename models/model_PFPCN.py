import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Convlayer(nn.Module):
    def __init__(self,point_scales):
        super(Convlayer,self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)
    def forward(self,x):
        x = torch.unsqueeze(x,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_512 = F.relu(self.bn5(self.conv5(x_256)))
        x_1024 = F.relu(self.bn6(self.conv6(x_512)))
        x_128 = torch.squeeze(self.maxpool(x_128),2)
        x_256 = torch.squeeze(self.maxpool(x_256),2)
        x_512 = torch.squeeze(self.maxpool(x_512),2)
        x_1024 = torch.squeeze(self.maxpool(x_1024),2)
        L = [x_1024,x_512,x_256,x_128]
        x = torch.cat(L,1)
        return x

class Latentfeature(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list):
        super(Latentfeature,self).__init__()
        self.num_scales = num_scales
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[0]) for i in range(self.each_scales_size)])
        self.Convlayers2 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[1]) for i in range(self.each_scales_size)])
        self.Convlayers3 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[2]) for i in range(self.each_scales_size)])
        self.conv1 = torch.nn.Conv1d(3,1,1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self,x):
        outs = []
        for i in range(self.each_scales_size):
            outs.append(self.Convlayers1[i](x[0]))
        for j in range(self.each_scales_size):
            outs.append(self.Convlayers2[j](x[1]))
        for k in range(self.each_scales_size):
            outs.append(self.Convlayers3[k](x[2]))
        # for i in range(len(outs)):
        #     print(outs[i].shape)
        latentfeature = torch.cat(outs,2)
        latentfeature = latentfeature.transpose(1,2)
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))
        latentfeature = torch.squeeze(latentfeature,1)
#        latentfeature_64 = F.relu(self.bn1(self.conv1(latentfeature)))  
#        latentfeature = F.relu(self.bn2(self.conv2(latentfeature_64)))
#        latentfeature = F.relu(self.bn3(self.conv3(latentfeature)))
#        latentfeature = latentfeature + latentfeature_64
#        latentfeature_256 = F.relu(self.bn4(self.conv4(latentfeature)))
#        latentfeature = F.relu(self.bn5(self.conv5(latentfeature_256)))
#        latentfeature = F.relu(self.bn6(self.conv6(latentfeature)))
#        latentfeature = latentfeature + latentfeature_256
#        latentfeature = F.relu(self.bn7(self.conv7(latentfeature)))
#        latentfeature = F.relu(self.bn8(self.conv8(latentfeature)))
#        latentfeature = self.maxpool(latentfeature)
#        latentfeature = torch.squeeze(latentfeature,2)
        return latentfeature

class PointcloudCls(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list,k=40):
        super(PointcloudCls,self).__init__()
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.latentfeature(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))        
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class PCN(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)
    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4):
        super().__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, xyz):
        B, N, _ = xyz.shape
        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 1024)
        
        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), coarse point cloud
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()

class PCNEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.latent_dim = latent_dim

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )
    
    def forward(self, xyz):
        B, N, _ = xyz.shape
        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 1024)
        return feature_global

class PCNDecoder(nn.Module):
    def __init__(self, latent_dim=1024, num_dense=6144, grid_size=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_dense = num_dense
        self.grid_size = grid_size
        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.final_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, feature_global, coarse):
        B, N, _ = coarse.shape
        # print("1. coarse", coarse.shape)
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        # print("2. point_feat", point_feat.shape)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return fine.transpose(1, 2).contiguous()

class _netG_PCN(nn.Module):
    def  __init__(self,num_scales,each_scales_size,point_scales_list,crop_point_num):
        super(_netG_PCN,self).__init__()
        self.crop_point_num = crop_point_num
        self.pcn_encoder = PCNEncoder(latent_dim=1024)
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        self.feat_fc = nn.Linear(1920+1024, 1024)
        self.fc1 = nn.Linear(1920,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        
        # self.fc1_1 = nn.Linear(1024,128*512)
        self.fc2_1 = nn.Linear(512,64*128)#nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256,64*3)
        
#        self.bn1 = nn.BatchNorm1d(1024)
#        self.bn2 = nn.BatchNorm1d(512)
#        self.bn3 = nn.BatchNorm1d(256)#nn.BatchNorm1d(64*256) !
#        self.bn4 = nn.BatchNorm1d(128*512)#nn.BatchNorm1d(256)
#        self.bn5 = nn.BatchNorm1d(64*128)
#        
        # self.conv1_1 = torch.nn.Conv1d(512,512,1)#torch.nn.Conv1d(256,256,1) !
        # self.conv1_2 = torch.nn.Conv1d(512,256,1)
        # self.conv1_3 = torch.nn.Conv1d(256,int((self.crop_point_num*3)/128),1)
        self.conv2_1 = torch.nn.Conv1d(128,18,1)#torch.nn.Conv1d(256,12,1) !
        
#        self.bn1_ = nn.BatchNorm1d(512)
#        self.bn2_ = nn.BatchNorm1d(256)
        self.pcn_decoder = PCNDecoder(latent_dim=1024, num_dense=self.crop_point_num, grid_size=4)
        
    def forward(self,x):
        l_feat = self.latentfeature(x)
        x_1 = F.relu(self.fc1(l_feat)) #1024
        x_2 = F.relu(self.fc2(x_1)) #512
        x_3 = F.relu(self.fc3(x_2))  #256
        
        pc1_feat = self.fc3_1(x_3)
        # print("0. pc1_feat",pc1_feat.shape)
        pc1_xyz = pc1_feat.reshape(-1,64,3) #64x3 center1
        # print("1. pc1_xyz",pc1_xyz.shape)
        
        pc2_feat = F.relu(self.fc2_1(x_2))
        # print("2. pc2_feat",pc2_feat.shape)
        pc2_feat = pc2_feat.reshape(-1,128,64)
        # print("3. pc2_feat",pc2_feat.shape)
        pc2_xyz =self.conv2_1(pc2_feat) #6x64 center2
        # print("4. pc2_xyz",pc2_xyz.shape)

        # pc3_feat = F.relu(self.fc1_1(x_1))
        # print("5. pc3_feat",pc3_feat.shape)
        # pc3_feat = pc3_feat.reshape(-1,512,128)
        # print("6. pc3_feat",pc3_feat.shape)
        # pc3_feat = F.relu(self.conv1_1(pc3_feat))
        # print("7. pc3_feat",pc3_feat.shape)
        # pc3_feat = F.relu(self.conv1_2(pc3_feat))
        # print("8. pc3_feat",pc3_feat.shape)
        # pc3_xyz = self.conv1_3(pc3_feat) #144x128 fine
        # print("9. pc3_xyz",pc3_xyz.shape)
        
        pc1_xyz_expand = torch.unsqueeze(pc1_xyz,2)
        # print("10. pc1_xyz_expand",pc1_xyz_expand.shape)
        pc2_xyz = pc2_xyz.transpose(1,2)
        # print("11. pc2_xyz",pc2_xyz.shape)
        pc2_xyz = pc2_xyz.reshape(-1,64,6,3)
        # print("12. pc2_xyz",pc2_xyz.shape)
        pc2_xyz = pc1_xyz_expand+pc2_xyz
        # print("13. pc2_xyz",pc2_xyz.shape)
        pc2_xyz = pc2_xyz.reshape(-1,384,3) 
        # print("14. pc2_xyz",pc2_xyz.shape)
        
        # pc2_xyz_expand = torch.unsqueeze(pc2_xyz,2)
        # print("15. pc2_xyz_expand",pc2_xyz_expand.shape)
        # pc3_xyz = pc3_xyz.transpose(1,2)
        # print("16. pc3_xyz",pc3_xyz.shape)
        # pc3_xyz = pc3_xyz.reshape(-1,128,int(self.crop_point_num/128),3)
        # print("17. pc3_xyz",pc3_xyz.shape)
        # pc3_xyz = pc2_xyz_expand+pc3_xyz
        # print("18. pc3_xyz",pc3_xyz.shape)
        # pc3_xyz = pc3_xyz.reshape(-1,self.crop_point_num,3)
        # print("19. pc3_xyz",pc3_xyz.shape)
        
        p_feat = self.pcn_encoder(x[0])
        feat = torch.cat((p_feat, l_feat), 1)
        feat = self.feat_fc(feat)
        pc3_xyz = self.pcn_decoder(feat, pc2_xyz)
        
        return pc1_xyz,pc2_xyz,pc3_xyz #center1 ,center2 ,fine

class _netlocalD(nn.Module):
    def __init__(self,crop_point_num):
        super(_netlocalD,self).__init__()
        self.crop_point_num = crop_point_num
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.maxpool = torch.nn.MaxPool2d((self.crop_point_num, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(448,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,16)
        self.fc4 = nn.Linear(16,1)
        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x_64 = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x_64)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_64 = torch.squeeze(self.maxpool(x_64))
        x_128 = torch.squeeze(self.maxpool(x_128))
        x_256 = torch.squeeze(self.maxpool(x_256))
        Layers = [x_256,x_128,x_64]
        x = torch.cat(Layers,1)
        x = F.relu(self.bn_1(self.fc1(x)))
        x = F.relu(self.bn_2(self.fc2(x)))
        x = F.relu(self.bn_3(self.fc3(x)))
        x = self.fc4(x)
        return x

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input1 = torch.randn(1,6144,3).to(device)
    input2 = torch.randn(1,384,3).to(device)
    input3 = torch.randn(1,512,3).to(device)
    input_ = [input1,input2,input3]
    netG=_netG_PCN(3,1,[6144,384,512],6144).to(device)
    print(count_parameters(netG))
    lFeature = Latentfeature(3,1,[6144,384,512])
    print(count_parameters(lFeature))
    pcn = PCNEncoder()
    print(count_parameters(pcn))
    output = netG(input_)
    cen1, cen2, fine = output
    print(cen1.shape, cen2.shape, fine.shape)
    
