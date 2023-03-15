import os
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import random
from utils import distance_squre
from torch.autograd import Variable
import utils

class DFaustPreDataset(Dataset):
    def __init__(self, folder, jsonFile, opt, partition="train", gt_num_points=6144, transform=None):
        self.folder = folder
        self.transform = transform
        self.partial = []
        self.gts = []
        self.labels = []
        self.gt_num_points = gt_num_points
        self.opt = opt
        count = 1
        self.sdata = []
        if "D-Faust_Pre" not in folder:
            jfolder = os.path.join(folder, "D-Faust_Pre")
        with open(os.path.join(jfolder, jsonFile), 'r') as f:
            data = json.load(f)
        for name in tqdm(data[partition]):
            self.sdata.extend(torch.load(os.path.join(jfolder, name)))
        print("Partition:", partition, " Loaded ", len(self.sdata), " samples")

    def __len__(self):
        return len(self.sdata)

    def __getitem__(self, idx):
        return self.sdata[idx]