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
from model_PFNet import _netlocalD,_netG
from model_PFPCN import _netlocalD,_netG_PCN
from dfaustDataset import DFaustDataset
from dfaustPreDataset import DFaustPreDataset
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
import time
from visual import plot_pcd_one_view
import datetime
from tqdm import tqdm


def print_log(fd,  message, time=True):
    if time:
        message = ' ==> '.join([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message])
    fd.write(message + '\n')
    fd.flush()
    print(message)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',  default=".", help='path to dataset')
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
    parser.add_argument('--pcn', action='store_true', help='use point cloud network')
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
    parser.add_argument('--preprocess', action='store_true', help='preprocess the data')
    parser.add_argument('--img_freq', type=int, default=100, help='frequency of saving images')
    opt = parser.parse_args()
    return opt

def get_models(opt):
    if opt.pcn:
        point_netG = _netG_PCN(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)
    else:
        point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)
    point_netD = _netlocalD(opt.crop_point_num)
    return point_netG, point_netD

def get_dataLoaders(f, opt):
    print_log(f, "Loading the data...")
    folder = opt.dataroot
    json = opt.json
    batch_size = opt.batchSize
    gt_points = opt.pnum
    if opt.preprocess:
        trainDataset = DFaustDataset(folder, json, opt, partition="train", gt_num_points=gt_points)
        testDataset = DFaustDataset(folder, json, opt, partition="test", gt_num_points=gt_points)
        # valDataset = DFaustDataset(folder, json, partition="val")
    else:
        trainDataset = DFaustPreDataset(folder, json, opt, partition="train", gt_num_points=gt_points)
        testDataset = DFaustPreDataset(folder, json, opt, partition="test", gt_num_points=gt_points)
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

def load_model(model, path, f):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    resume_epoch = checkpoint['epoch']
    if 'loss' in checkpoint.keys():
        loss = checkpoint['loss']
    else:
        loss = None
    print_log(f, "Loaded {} Model with Epoch {} Loss {}".format(path, resume_epoch, loss))
    return model

def pre_ops(opt):
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    # print_log(f, "Random Seed: " + str(opt.manualSeed))
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    if not os.path.exists(os.path.join(opt.save_dir, 'pcds')):
        os.makedirs(os.path.join(opt.save_dir, 'pcds'))
    if not os.path.exists(os.path.join(opt.save_dir, 'imgs')):
        os.makedirs(os.path.join(opt.save_dir, 'imgs'))
   
def train(point_netG, point_netD, trainLoader, testLoader, trainDataset, testDataset, opt):
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    chamfer_loss = ChamferDistanceL1().to(device)

    f=open(os.path.join(opt.save_dir, 'loss_PFNet.txt'),'a')

    # setup optimizer
    optimizerD = torch.optim.Adam(point_netD.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05,weight_decay=opt.weight_decay)
    optimizerG = torch.optim.Adam(point_netG.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05 ,weight_decay=opt.weight_decay)
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.2)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)

    best_loss = 1000000
    for epoch in range(0, opt.niter):
        if epoch<15:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch<20:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2
        
        trainG_loss, trainD_loss = train_epoch(point_netG, point_netD, trainLoader, epoch, alpha1, alpha2, criterion, chamfer_loss, optimizerG, optimizerD, opt)
        
        test_loss = test_epoch(point_netG, testLoader, epoch, chamfer_loss, opt)

        schedulerD.step()
        schedulerG.step()
        
        torch.save({'epoch':epoch+1,
                    'loss': test_loss,
                    'state_dict':point_netG.state_dict()},
                    os.path.join(opt.save_dir, 'point_netG_last.pth' ))
        torch.save({'epoch':epoch+1,
                    'loss': test_loss,
                    'state_dict':point_netD.state_dict()},
                    os.path.join(opt.save_dir, 'point_netD_last.pth' ))
        print_log(f, 'Saved Last Model')

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
            print_log(f, 'Saved Best Model')

def preprocess(data, opt):

    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    real_point = data
    real_point = real_point.float()

    batch_size = real_point.size()[0]
    real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)       
    input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
    input_cropped1 = input_cropped1.data.copy_(real_point)
    real_point = torch.unsqueeze(real_point, 1)
    input_cropped1 = torch.unsqueeze(input_cropped1,1)

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
    return input_cropped1, input_cropped2, input_cropped3, real_center_key1, real_center_key2, real_center, label

def train_epoch(point_netG, point_netD, trainLoader, epoch, alpha1, alpha2, criterion, chamfer_loss, optimizerG, optimizerD, opt):

    f=open(os.path.join(opt.save_dir, 'loss_PFNet.txt'),'a')

    real_label = 1
    fake_label = 0

    point_netG.train()
    point_netD.train()
    
    trainG_loss = 0.0
    trainD_loss = 0.0
    start = time.time()
    for i, data in enumerate(trainLoader, 0):
        
        if opt.preprocess:
            cdata = preprocess(data)
        else:
            cdata = data

        input_cropped1, input_cropped2, input_cropped3, real_center_key1, real_center_key2, real_center, label = cdata
        
        input_cropped1 = torch.squeeze(input_cropped1).to(device)
        input_cropped2 = torch.squeeze(input_cropped2).to(device)
        input_cropped3 = torch.squeeze(input_cropped3).to(device)
        input_cropped = [input_cropped1,input_cropped2,input_cropped3]
        real_center_key1 = torch.squeeze(real_center_key1).to(device)
        real_center_key2 = torch.squeeze(real_center_key2).to(device)
        real_center = torch.squeeze(real_center).to(device)
        label = torch.squeeze(label).reshape(-1, 1).to(device)

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
        errG.backward()
        optimizerG.step()
        trainG_loss += errG.item()
        # print_log(f, '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f/ %.4f, batch time: %.4f'
        #     % (epoch, opt.niter, i, len(trainLoader),  
        #         errD.data, errG_D.data,errG_l2,errG,CD_LOSS,b_time))

        if i % opt.img_freq == 0:
            print_log(f, '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Time Elapsed: %.4f'
                % (epoch, opt.niter, i, len(trainLoader), trainD_loss/(i+1), trainG_loss/(i+1), time.time()-start))
            # if not os.path.exists(os.path.join(opt.save_dir, 'pcds')):
            #     os.makedirs(os.path.join(opt.save_dir, 'pcds'))
            if not os.path.exists(os.path.join(opt.save_dir, 'imgs')):
                os.makedirs(os.path.join(opt.save_dir, 'imgs'))
            # save_pcd(real_center[0].cpu().detach().numpy().reshape(-1, 3), '{}/train_{}_{}_real.pcd'.format(os.path.join(opt.save_dir, 'pcds'), epoch, i))
            # save_pcd(fake[0].cpu().detach().numpy().reshape(-1, 3), '{}/train_{}_{}_fake.pcd'.format(os.path.join(opt.save_dir, 'temp'), epoch, i))
            plot_pcd_one_view(os.path.join(opt.save_dir, 'imgs', 'train_{}_{}.png'.format(epoch, i)),
                                [real_center[0].cpu().detach().numpy().reshape(-1, 3), fake[0].cpu().detach().numpy().reshape(-1, 3)],
                                ['real', 'fake'], xlim=[-1, 1], ylim=[-1, 1], zlim=[-1, 1])

    trainG_loss /= len(trainLoader)
    trainD_loss /= len(trainLoader)
    print_log(f, 'Epoch: %d, trainD_loss: %.4f, trainG_loss: %.4f, Time Elapsed: %.4f'
        % (epoch, trainD_loss, trainG_loss, time.time()-start))
    f.close()
    return trainG_loss, trainD_loss

def test_epoch(point_netG, testLoader, epoch, chamfer_loss, opt):
    f = open(os.path.join(opt.save_dir, 'loss_PFNet.txt'),'a')
    label = torch.FloatTensor(opt.batchSize)
    start = time.time()
    test_loss = 0.0
    for i, data in enumerate(testLoader, 0):

        input_cropped1, input_cropped2, input_cropped3, real_center_key1, real_center_key2, real_center, label = data
        
        input_cropped1 = torch.squeeze(input_cropped1).to(device)
        input_cropped2 = torch.squeeze(input_cropped2).to(device)
        input_cropped3 = torch.squeeze(input_cropped3).to(device)
        input_cropped = [input_cropped1,input_cropped2,input_cropped3]
        real_center_key1 = torch.squeeze(real_center_key1).to(device)
        real_center_key2 = torch.squeeze(real_center_key2).to(device)
        real_center = torch.squeeze(real_center).to(device)
        label = torch.squeeze(label).reshape(-1, 1).to(device)
        
        point_netG.eval()
        fake_center1,fake_center2,fake = point_netG(input_cropped)
        CD_loss = chamfer_loss(torch.squeeze(fake,1),torch.squeeze(real_center,1))
        test_loss += CD_loss.item()

        if i == 0:
            # if not os.path.exists(os.path.join(opt.save_dir, 'pcds')):
            #     os.makedirs(os.path.join(opt.save_dir, 'pcds'))
            if not os.path.exists(os.path.join(opt.save_dir, 'imgs')):
                os.makedirs(os.path.join(opt.save_dir, 'imgs'))
            # save_pcd(real_center[0].cpu().detach().numpy().reshape(-1, 3), '{}/test_{}_real.pcd'.format(os.path.join(opt.save_dir, 'pcds'),epoch))
            # save_pcd(fake[0].cpu().detach().numpy().reshape(-1, 3), '{}/test_{}_fake.pcd'.format(os.path.join(opt.save_dir, 'temp'),epoch))
            plot_pcd_one_view(os.path.join(opt.save_dir, 'imgs', 'test_{}.png'.format(epoch)),
                                [real_center[0].cpu().detach().numpy().reshape(-1, 3), fake[0].cpu().detach().numpy().reshape(-1, 3)],
                                ['real', 'fake'], xlim=[-1, 1], ylim=[-1, 1], zlim=[-1, 1])
    
    end = time.time()
    test_loss /= len(testLoader)
    print_log(f, 'Epoch: %d, test loss: %.4f, Time: %.4f' % (epoch, test_loss,end-start))
    
    f.close()
    return test_loss


if __name__ == "__main__":
    opt = get_parser()
    pre_ops(opt)
    f = open(os.path.join(opt.save_dir, 'loss_PFNet.txt'),'w')
    print_log(f, str(opt))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainLoader, testLoader, trainDataset, testDataset = get_dataLoaders(f, opt)

    # start = time.time()
    # for i, data in tqdm(enumerate(trainLoader, 0)):
    #     pass
    # end = time.time()
    # print_log(f, 'Time for loading data: %.4f' % (end-start))
    
    point_netG, point_netD = get_models(opt)

    if torch.cuda.is_available():
        print_log(f, "Using GPU")
        point_netG.to(device)
        point_netG.apply(weights_init_normal)
        point_netD.to(device)
        point_netD.apply(weights_init_normal)

    if opt.netG != '':
        point_netG = load_model(point_netG, opt.netG, f)
    
    if opt.netD != '':
        point_netD = load_model(point_netD, opt.netD, f)
    
    # print(point_netG)
    print_log(f, "Total Number of Parameters in netG: {:.3f}M".format(count_parameters(point_netG)/1e6))
    # print(point_netD)
    print_log(f, "Total Number of Parameters in netD: {:.3f}M".format(count_parameters(point_netD)/1e6))

    f.close()

    train(point_netG, point_netD, trainLoader, testLoader, trainDataset, testDataset, opt)



