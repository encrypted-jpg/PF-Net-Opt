import os
import sys
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
from utils import PointLoss
from extensions.chamfer_dist import ChamferDistanceL1
from utils import distance_squre
import data_utils as d_utils
import ModelNet40Loader
import shapenet_part_loader
from models.model_PFNet import _netlocalD,_netG
from datasets.dfaustDataset import DFaustDataset
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
import time
from visual import plot_pcd_one_view



parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',  default="C:/Users/valla/Desktop/BTP/BTP2/", help='path to dataset')
parser.add_argument('--json', default="data.json", help='path to json file')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
parser.add_argument('--pnum', type=int, default=6144, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=6144,help='0 means do not use else use with this weight')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=201, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--D_choose',type=int, default=1, help='0 not use D-net,1 use D-net')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[6144,1024,512],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.95,help='0 means do not use else use with this weight')
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
parser.add_argument('--save_dir', default = 'checkpoints', help = 'save directory')
opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USE_CUDA = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)
point_netD = _netlocalD(opt.crop_point_num)
cudnn.benchmark = True
resume_epoch=0

def dataLoaders(opt):
    print("[+] Loading the data...")
    folder = opt.dataroot
    json = opt.json
    batch_size = opt.batchSize
    gt_points = opt.pnum
    trainDataset = DFaustDataset(folder, json, partition="train", gt_num_points=gt_points)
    testDataset = DFaustDataset(folder, json, partition="test", gt_num_points=gt_points)
    # valDataset = DFaustDataset(folder, json, partition="val")
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=True)
    # valLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
    return trainLoader, testLoader, trainDataset, testDataset

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_pcd(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0) 

if USE_CUDA:       
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # point_netG = torch.nn.DataParallel(point_netG)
    # point_netD = torch.nn.DataParallel(point_netD)
    point_netG.to(device) 
    point_netG.apply(weights_init_normal)
    point_netD.to(device)
    point_netD.apply(weights_init_normal)

if opt.netG != '' :
    checkpoint = torch.load(opt.netG,map_location=lambda storage, location: storage)
    point_netG.load_state_dict(checkpoint['state_dict'])
    resume_epoch = checkpoint['epoch']
    if 'loss' in checkpoint.keys():
        loss = checkpoint['loss']
    else:
        loss = None
    print("Loaded netG Model from epoch {} with loss {}".format(resume_epoch, loss))

if opt.netD != '' :
    checkpoint = torch.load(opt.netD,map_location=lambda storage, location: storage)
    point_netD.load_state_dict(checkpoint['state_dict'])
    resume_epoch = checkpoint['epoch']
    if 'loss' in checkpoint.keys():
        loss = checkpoint['loss']
    else:
        loss = None
    print("Loaded netD Model from epoch {} with loss {}".format(resume_epoch, loss))

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)

transforms = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
    ]
)

# dset = shapenet_part_loader.PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=opt.pnum, split='train')
# assert dset
# dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,
#                                          shuffle=True,num_workers = int(opt.workers))


# test_dset = shapenet_part_loader.PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=opt.pnum, split='test')
# test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
#                                          shuffle=True,num_workers = int(opt.workers))

dataloader, test_dataloader, dset, test_dset = dataLoaders(opt)

#dset = ModelNet40Loader.ModelNet40Cls(opt.pnum, train=True, transforms=transforms, download = False)
#assert dset
#dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,
#                                         shuffle=True,num_workers = int(opt.workers))
#
#
#test_dset = ModelNet40Loader.ModelNet40Cls(opt.pnum, train=False, transforms=transforms, download = False)
#test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
#                                         shuffle=True,num_workers = int(opt.workers))

#pointcls_net.apply(weights_init)
print(point_netG)
print("[+] Total Number of Parameters: {}".format(count_parameters(point_netG)))
print(point_netD)
print("[+] Total Number of Parameters: {}".format(count_parameters(point_netD)))

criterion = torch.nn.BCEWithLogitsLoss().to(device)
chamfer_loss = ChamferDistanceL1().to(device)

# setup optimizer
optimizerD = torch.optim.Adam(point_netD.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05,weight_decay=opt.weight_decay)
optimizerG = torch.optim.Adam(point_netG.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05 ,weight_decay=opt.weight_decay)
schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.2)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)

real_label = 1
fake_label = 0

crop_point_num = int(opt.crop_point_num)
# input_cropped1 = torch.FloatTensor(opt.batchSize, opt.pnum, 3)
label = torch.FloatTensor(opt.batchSize)


num_batch = len(dset) / opt.batchSize
###########################
#  G-NET and T-NET
##########################  
if opt.D_choose == 1:
    best_loss = 1000000
    for epoch in range(resume_epoch,opt.niter):
        if epoch<15:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch<20:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2
        
        trainG_loss = 0.0
        trainD_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            start = time.time()
            real_point, target = data
            # Float32
            real_point = real_point.float()
            # Reduce Density
            # real_point = real_point[:,::opt.sample_num,:]

            batch_size = real_point.size()[0]
            real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)       
            input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
            input_cropped1 = input_cropped1.data.copy_(real_point)
            real_point = torch.unsqueeze(real_point, 1)
            input_cropped1 = torch.unsqueeze(input_cropped1,1)
            p_origin = [0,0,0]
            if opt.cropmethod == 'random_center':
                #Set viewpoints
                choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
                for m in range(batch_size):
                    index = random.sample(choice,1)#Random choose one of the viewpoint
                    distance_list = []
                    p_center = index[0]
                    for n in range(opt.pnum):
                        distance_list.append(distance_squre(real_point[m,0,n],p_center))
                    distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])
                    
                    for sp in range(opt.crop_point_num):
                        input_cropped1.data[m,0,distance_order[sp][0]] = torch.FloatTensor([0,0,0])
                        real_center.data[m,0,sp] = real_point[m,0,distance_order[sp][0]]
            label.resize_([batch_size,1]).fill_(real_label)
            real_point = real_point.to(device)
            real_center = real_center.to(device)
            input_cropped1 = input_cropped1.to(device)
            label = label.to(device)
            ############################
            # (1) data prepare
            ###########################      
            real_center = Variable(real_center,requires_grad=True)
            real_center = torch.squeeze(real_center,1)
            real_center_key1_idx = utils.farthest_point_sample(real_center,64,RAN = False)
            real_center_key1 = utils.index_points(real_center,real_center_key1_idx)
            real_center_key1 =Variable(real_center_key1,requires_grad=True)

            real_center_key2_idx = utils.farthest_point_sample(real_center,128,RAN = True)
            real_center_key2 = utils.index_points(real_center,real_center_key2_idx)
            real_center_key2 =Variable(real_center_key2,requires_grad=True)

            input_cropped1 = torch.squeeze(input_cropped1,1)
            input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
            input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
            input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
            input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
            input_cropped1 = Variable(input_cropped1,requires_grad=True)
            input_cropped2 = Variable(input_cropped2,requires_grad=True)
            input_cropped3 = Variable(input_cropped3,requires_grad=True)
            input_cropped2 = input_cropped2.to(device)
            input_cropped3 = input_cropped3.to(device)      
            input_cropped  = [input_cropped1,input_cropped2,input_cropped3]
            
            # print(input_cropped1.shape, input_cropped2.shape, input_cropped3.shape)
            
            point_netG = point_netG.train()
            point_netD = point_netD.train()
            ############################
            # (2) Update D network
            ###########################        
            point_netD.zero_grad()
            real_center = torch.unsqueeze(real_center,1)   
            output = point_netD(real_center)
            errD_real = criterion(output,label)
            errD_real.backward()
            fake_center1,fake_center2,fake  =point_netG(input_cropped)
            # print(fake_center1.shape, fake_center2.shape, fake.shape)
            fake = torch.unsqueeze(fake,1)
            label.data.fill_(fake_label)
            output = point_netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()
            trainD_loss += errD.item()
            ############################
            # (3) Update G network: maximize log(D(G(z)))
            ###########################
            point_netG.zero_grad()
            label.data.fill_(real_label)
            output = point_netD(fake)
            errG_D = criterion(output, label)
            errG_l2 = 0
            # print(real_center.shape, fake.shape)
            CD_LOSS = chamfer_loss(torch.squeeze(fake,1),torch.squeeze(real_center,1))
       
            errG_l2 = chamfer_loss(torch.squeeze(fake,1),torch.squeeze(real_center,1))\
            +alpha1*chamfer_loss(fake_center1,real_center_key1)\
            +alpha2*chamfer_loss(fake_center2,real_center_key2)
            
            errG = (1-opt.wtl2) * errG_D + opt.wtl2 * errG_l2
            trainG_loss += errG.item()
            errG.backward()
            optimizerG.step()
            end = time.time()
            b_time = end - start
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f/ %.4f, batch time: %.4f'
                  % (epoch, opt.niter, i, len(dataloader),  
                     errD.data, errG_D.data,errG_l2,errG,CD_LOSS,b_time))
            f=open(os.path.join(opt.save_dir, 'loss_PFNet.txt'),'a')
            f.write('\n'+'[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f /%.4f, batch time: %.4f'
                  % (epoch, opt.niter, i, len(dataloader), 
                     errD.data, errG_D.data,errG_l2,errG,CD_LOSS, b_time))

            if i % 20 == 0:
                if not os.path.exists(os.path.join(opt.save_dir, 'pcds')):
                    os.makedirs(os.path.join(opt.save_dir, 'pcds'))
                if not os.path.exists(os.path.join(opt.save_dir, 'imgs')):
                    os.makedirs(os.path.join(opt.save_dir, 'imgs'))
                save_pcd(real_center[0].cpu().detach().numpy().reshape(-1, 3), '{}/train_{}_{}_real.pcd'.format(os.path.join(opt.save_dir, 'pcds'), epoch, i))
                save_pcd(fake[0].cpu().detach().numpy().reshape(-1, 3), '{}/train_{}_{}_fake.pcd'.format(os.path.join(opt.save_dir, 'temp'), epoch, i))
                plot_pcd_one_view(os.path.join(opt.save_dir, 'imgs', 'train_{}_{}.png'.format(epoch, i)),
                                    [real_center[0].cpu().detach().numpy().reshape(-1, 3), fake[0].cpu().detach().numpy().reshape(-1, 3)],
                                    ['real', 'fake'], xlim=[-1, 1], ylim=[-1, 1], zlim=[-1, 1])

        trainG_loss /= len(dataloader)
        trainD_loss /= len(dataloader)
        print('Epoch: %d, trainG_loss: %.4f, trainD_loss: %.4f' % (epoch, trainG_loss, trainD_loss))
        f.write('\n'+'Epoch: %d, trainG_loss: %.4f, trainD_loss: %.4f' % (epoch, trainG_loss, trainD_loss))
        start = time.time()
        test_loss = 0.0
        rid = np.random.randint(0, len(test_dataloader))
        for i, data in enumerate(test_dataloader, 0):
            real_point, target = data
            
            real_point = real_point.float()
            batch_size = real_point.size()[0]
            real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
            input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
            input_cropped1 = input_cropped1.data.copy_(real_point)
            real_point = torch.unsqueeze(real_point, 1)
            input_cropped1 = torch.unsqueeze(input_cropped1,1)
            
            p_origin = [0,0,0]
            
            if opt.cropmethod == 'random_center':
                choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
                
                for m in range(batch_size):
                    index = random.sample(choice,1)
                    distance_list = []
                    p_center = index[0]
                    for n in range(opt.pnum):
                        distance_list.append(distance_squre(real_point[m,0,n],p_center))
                    distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])                         
                    for sp in range(opt.crop_point_num):
                        input_cropped1.data[m,0,distance_order[sp][0]] = torch.FloatTensor([0,0,0])
                        real_center.data[m,0,sp] = real_point[m,0,distance_order[sp][0]]  
            real_center = real_center.to(device)
            real_center = torch.squeeze(real_center,1)
            input_cropped1 = input_cropped1.to(device) 
            input_cropped1 = torch.squeeze(input_cropped1,1)
            input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
            input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
            input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
            input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
            input_cropped1 = Variable(input_cropped1,requires_grad=False)
            input_cropped2 = Variable(input_cropped2,requires_grad=False)
            input_cropped3 = Variable(input_cropped3,requires_grad=False)
            input_cropped2 = input_cropped2.to(device)
            input_cropped3 = input_cropped3.to(device)      
            input_cropped  = [input_cropped1,input_cropped2,input_cropped3]
            point_netG.eval()
            fake_center1,fake_center2,fake  =point_netG(input_cropped)
            if rid == i:
                if not os.path.exists(os.path.join(opt.save_dir, 'pcds')):
                    os.makedirs(os.path.join(opt.save_dir, 'pcds'))
                if not os.path.exists(os.path.join(opt.save_dir, 'imgs')):
                    os.makedirs(os.path.join(opt.save_dir, 'imgs'))
                save_pcd(real_center[0].cpu().detach().numpy().reshape(-1, 3), '{}/test_{}_real.pcd'.format(os.path.join(opt.save_dir, 'pcds'),epoch))
                save_pcd(fake[0].cpu().detach().numpy().reshape(-1, 3), '{}/test_{}_fake.pcd'.format(os.path.join(opt.save_dir, 'temp'),epoch))
                plot_pcd_one_view(os.path.join(opt.save_dir, 'imgs', 'test_{}.png'.format(epoch)),
                                    [real_center[0].cpu().detach().numpy().reshape(-1, 3), fake[0].cpu().detach().numpy().reshape(-1, 3)],
                                    ['real', 'fake'], xlim=[-1, 1], ylim=[-1, 1], zlim=[-1, 1])
            CD_loss = chamfer_loss(torch.squeeze(fake,1),torch.squeeze(real_center,1))
            test_loss += CD_loss.item()
        
        end = time.time()
        test_loss /= len(test_dataloader)
        print('Epoch: %d, test loss: %.4f, Time: %.4f' % (epoch, test_loss,end-start))
        f.write('\n'+'Epoch: %d, test loss: %.4f, Time: %.4f' % (epoch, test_loss,end-start))
        
        f.close()
        schedulerD.step()
        schedulerG.step()
        # os.mkdir(opt.save_dir, exist_ok=True)
        torch.save({'epoch':epoch+1,
                    'loss': test_loss,
                    'state_dict':point_netG.state_dict()},
                    os.path.join(opt.save_dir, 'point_netG_last.pth' ))
        torch.save({'epoch':epoch+1,
                    'loss': test_loss,
                    'state_dict':point_netD.state_dict()},
                    os.path.join(opt.save_dir, 'point_netD_last.pth' ))
        print('Saved Last Model')
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({'epoch':epoch+1,
                        'loss': test_loss,
                        'state_dict':point_netG.state_dict()},
                        os.path.join(opt.save_dir, 'point_netG_best.pth' ))
            torch.save({'epoch':epoch+1,
                        'loss': test_loss,
                        'state_dict':point_netD.state_dict()},
                        os.path.join(opt.save_dir, 'point_netD_best.pth' ))
            print('Saved Best Model')

#
#############################
## ONLY G-NET
############################ 
else:
    for epoch in range(resume_epoch,opt.niter):
        if epoch<30:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch<80:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2
        
        for i, data in enumerate(dataloader, 0):
            
            real_point, target = data
            
            real_point = real_point.float()
            batch_size = real_point.size()[0]
            real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)       
            input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
            input_cropped1 = input_cropped1.data.copy_(real_point)
            real_point = torch.unsqueeze(real_point, 1)
            input_cropped1 = torch.unsqueeze(input_cropped1,1)
            p_origin = [0,0,0]
            if opt.cropmethod == 'random_center':
                choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
                for m in range(batch_size):
                    index = random.sample(choice,1)
                    distance_list = []
                    p_center = index[0]
                    for n in range(opt.pnum):
                        distance_list.append(distance_squre(real_point[m,0,n],p_center))
                    distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])
                    
                    for sp in range(opt.crop_point_num):
                        input_cropped1.data[m,0,distance_order[sp][0]] = torch.FloatTensor([0,0,0])
                        real_center.data[m,0,sp] = real_point[m,0,distance_order[sp][0]]
            real_point = real_point.to(device)
            real_center = real_center.to(device)
            input_cropped1 = input_cropped1.to(device)
            ############################
            # (1) data prepare
            ###########################      
            real_center = Variable(real_center,requires_grad=True)
            real_center = torch.squeeze(real_center,1)
            real_center_key1_idx = utils.farthest_point_sample(real_center,64,RAN = False)
            real_center_key1 = utils.index_points(real_center,real_center_key1_idx)
            real_center_key1 =Variable(real_center_key1,requires_grad=True)

            real_center_key2_idx = utils.farthest_point_sample(real_center,128,RAN = True)
            real_center_key2 = utils.index_points(real_center,real_center_key2_idx)
            real_center_key2 =Variable(real_center_key2,requires_grad=True)
            
            input_cropped1 = torch.squeeze(input_cropped1,1)
            input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
            input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
            input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
            input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
            input_cropped1 = Variable(input_cropped1,requires_grad=True)
            input_cropped2 = Variable(input_cropped2,requires_grad=True)
            input_cropped3 = Variable(input_cropped3,requires_grad=True)
            input_cropped2 = input_cropped2.to(device)
            input_cropped3 = input_cropped3.to(device)      
            input_cropped  = [input_cropped1,input_cropped2,input_cropped3]
            point_netG = point_netG.train()
            point_netG.zero_grad()
            fake_center1,fake_center2,fake  =point_netG(input_cropped)
            fake = torch.unsqueeze(fake,1)
            ############################
            # (3) Update G network: maximize log(D(G(z)))
            ###########################
            
            CD_LOSS = chamfer_loss(torch.squeeze(fake,1),torch.squeeze(real_center,1))
            
            errG_l2 = chamfer_loss(torch.squeeze(fake,1),torch.squeeze(real_center,1))\
            +alpha1*chamfer_loss(fake_center1,real_center_key1)\
            +alpha2*chamfer_loss(fake_center2,real_center_key2)

            errG_l2.backward()
            optimizerG.step()
            print('[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                  % (epoch, opt.niter, i, len(dataloader), 
                      errG_l2,CD_LOSS))
            f=open(os.path.join(opt.save_dir, 'loss_PFNet.txt'),'a')
            f.write('\n'+'[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                  % (epoch, opt.niter, i, len(dataloader), 
                      errG_l2,CD_LOSS))
            f.close()
        schedulerD.step()
        schedulerG.step()
        
        if epoch% 10 == 0:   
            torch.save({'epoch':epoch+1,
                        'state_dict':point_netG.state_dict()},
                        'Checkpoint/point_netG'+str(epoch)+'.pth' )
 

    
        
