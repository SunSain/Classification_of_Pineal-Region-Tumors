
import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import SimpleITK as sitk
from scipy.ndimage import rotate
import nibabel as nib
import numpy as np
import os


class New_Feature_Folder(torch.utils.data.Dataset):
    def __init__(self, num_classes,t1_csv_path,t2_csv_path,t1c_csv_path):
        print("t1_csv_path,t2_csv_path,t1c_csv_path: ",t1_csv_path,t2_csv_path,t1c_csv_path)
        self.num_classes = num_classes
        self.t1_csv_path,self.t2_csv_path,self.t1c_csv_path = t1_csv_path,t2_csv_path,t1c_csv_path
        self.t1_dataset=[]
        self.t2_dataset=[]
        self.t1c_dataset=[]
        self.count=0
        self.y=[]
        self.sid_dict =[]
        self.all_feat=[]
        self.process()
    
    def process(self):
        t1c_f = open(self.t1c_csv_path,"r")
        t1c_csv = np.array(pd.read_csv(t1c_f,header=None))
        #print("t1c_csv: ",t1c_csv)
        t1c_data_box,t1c_target_box = self.single_process(t1c_csv)
        #print("t1c_data_box: ",t1c_data_box)
        t1c_f.close()
        
        t1_f = open(self.t1_csv_path,"r")
        t1_csv = np.array(pd.read_csv(t1_f,header=None))
        t1_data_box,t1_target_box = self.single_process(t1_csv)
        #print("t1_data_box: ",t1_data_box)
        t1_f.close()
        
        t2_f = open(self.t2_csv_path,"r")
        t2_csv = np.array(pd.read_csv(t2_f,header=None))
        t2_data_box,t2_target_box = self.single_process(t2_csv)
        #print("t2_data_box: ",t2_data_box)
        t2_f.close()
        

        print("t1_data_box.key: ",t1_data_box.keys())
        print("t2_data_box.key: ",t2_data_box.keys())
        print("t1c_data_box.key: ",t1c_data_box.keys())
        for key in t1c_data_box.keys():
            try:
                t1_feat = t1_data_box[key]
                t2_feat = t2_data_box[key]
                t1c_feat = t1c_data_box[key]
            except:
                continue
            self.sid_dict.append(key)
            self.all_feat.append( {'t1':t1_feat,'t2':t2_feat,'t1c':t1c_feat,'sid':key,'target':t1_target_box[key]})
            
    def single_process(self,t_csv):
        data_box={}
        target_box={}
        for i, item in enumerate(t_csv):
            sid =int( item[0])
            target = int(item[-1])
            feat = item[1:-1]
            data_box[sid ]=feat
            target_box[sid] = target
        return data_box,target_box
    
    def getsid(self):
        return self.sid_dict
    
    
    def __len__(self):
        return len(self.all_feat)


    def __add__(self,sid,t1_feature,t2_feature,t1c_feature,target):
        self.t1_dataset.append(t1_feature)
        self.t2_dataset.append(t2_feature)
        self.t1c_dataset.append(t1c_feature)
        self.y.append(target)
        self.sid_dict.append(sid)

    def __getitem__(self,index):
        return (torch.FloatTensor(self.all_feat[index]['t1']),torch.FloatTensor(self.all_feat[index]['t2']),torch.FloatTensor(self.all_feat[index]['t1c']),self.all_feat[index]['sid'],torch.tensor(self.all_feat[index]['target']))
    
    def gety(self):
        return self.y


