
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
from utils2.config_3class import opt
from sklearn.model_selection import train_test_split
import torchio as tio
from utils2.preprocessor import Preprocessor,array2image

from utils2.diy_dataset import Sub_DiyFolder
from csv import reader
import re
from scipy import stats
from utils2.DIY_radiomics import DiyRadiomics

def nii_loader(path):
    img = nib.load(str(path))
    return img

def read_table(path):
    return(pd.read_excel(path).values) # default to first sheet



class DIY_Folder_3M(torch.utils.data.Dataset):
    def __init__(self, num_classes,data_path, loader=nii_loader,train_form = "None",root_mask_radiomics="",use_radiomics=True,istest=False,sec_pair_orig=False,multiclass = True,vice_path = None,multi_M= False
                 ,t1_path ="", t2_path="", t1c_path ="",extra_test=False,valid_sid_box=None):
        self.num_classes = num_classes
        self.root = data_path
        self.sub_fns = sorted(os.listdir(self.root))

        self.root_mask_radiomics = root_mask_radiomics
        self.use_radiomics = use_radiomics
        self.istest = istest
        self.multi_M = multi_M
        self.extra_test=extra_test
        self.valid_sid_box=valid_sid_box
        
        #self.table_refer = read_table(excel_path)
        self.loader = loader
        self.dataset={}
        self.stastic = [0,0,0]
        self.stastic2=[0,0,0]
        self.origi=[0,0,0]
        self.count=0
        self.y=[]
        self.files={}
        self.infors={}
        self.ages = []
        self.filedict={}
        
        self.radiomics_pair_path={} # {sid:{'imgpath': , 'maskpath': }, }
        self.radiomics_features={}
        self.radiomics_root=[]
        self.masked_img={} #{sid:img,}
        self.third_img = {}
        self.sec_pair_orig=sec_pair_orig
        self.multiclass = multiclass
        self.vice_path = vice_path
        self.t1_path = t1_path
        self.t2_path = t2_path
        self.t1c_path = t1c_path
        
        self.delete_box=[]
        self.complete_file = []

        if vice_path!=None and vice_path!="":
            try:
                self.vice_subfns = sorted(os.listdir(self.vice_path))
                print("vice_subfns: ",self.vice_subfns)
                print("self.sub_fns: ",self.sub_fns)
                self.sub_fns.extend(self.vice_subfns)
                print("self.sub_fns2: ",self.sub_fns)
            except:
                self.vice_subfns = None
                print("No vice_path")

        self.preprocessor = Preprocessor(target_spacing=[0.45,0.45,6.5])
        self.train_form = train_form
        self.build_dict()
        self.get_img_files()
        if self.use_radiomics:
            self.get_radiomics()
        self.statistics() # clinical
        print(self.filedict.keys())
        print("self.complete_file: ")
        print("dict_keys(",self.complete_file)
        print("self.delete_box: ",self.delete_box)
        self.process() 

    def get_img_files(self):#统计图像路径，方便之后直接从数组取
        if  self.radiomics_root !=[]:
            print("Already get pair_radiomics_files, self.files")
            return
        for i, f in enumerate(self.sub_fns):
    
            fn = f.split("_")
            fn = fn[0].split(".")  
            try:
                    sid=int(fn[0])
            except:
                print("it's the_select_file")
                continue
            if sid in self.delete_box:
                print("this file is in delete_box: ",sid)
                continue
            # no more new data
            if sid>122:
                if not self.extra_test:
                    continue
            # no more new data
            if self.valid_sid_box!=None and not sid in self.valid_sid_box:
                continue
            sub_path = os.path.join(self.root, f)
            self.files[sid]=(f)
            
            pair_file = {'imgpath':sub_path}
            self.radiomics_pair_path[sid]=pair_file
            
        mask_root = self.root_mask_radiomics
        if not os.path.exists(mask_root):
            print("[ERROR]: No Mask_path , plz check if you want to use Radiomics")
            return 
        print("sofar self.radiomics_pair_path",self.radiomics_pair_path)
        instance_mask_file = sorted(os.listdir(mask_root))
        for i, f in enumerate(instance_mask_file):
            fn = f.split("_")
            fn = fn[0].split(".")
            try:
                    sid=int(fn[0])
            except:
                print("it's the_select_file")
                continue
            # no more new data
            if sid>122:
                if not self.extra_test:
                    continue
            # no more new data
            print("sid: ",sid)
            
            sub_path = os.path.join(mask_root, f)
            try:
                pair_file = self.radiomics_pair_path[sid]
                pair_file['maskpath']=sub_path
                self.radiomics_pair_path[sid] = pair_file
            except:
                print("sid has missing T1/T2: ",sid)
        print('self.radiomics_pair_path: ',self.radiomics_pair_path)
            
        

    def get_radiomics(self): #获得所有图像的radiomics
        if self.radiomics_pair_path == []:
            print("No radiomics...")
            return
        my_radiomics = DiyRadiomics(self.radiomics_pair_path)
        self.radiomics_features = my_radiomics.get_features()
       
        try:
            first_img_feature = self.radiomics_features[0]
            print("PyRadiomics Result: ",first_img_feature)
        except:
            print("No Radiomics fearure get!")
        
        
        
    def statistics(self): #统计姓名年龄信息
        with open(opt.infor_path, 'r',encoding='UTF-8') as read_obj:
            next(read_obj)
            csv_reader = read_obj.readlines()
            for row in csv_reader:
                print(row)
                reg = r"[\t,' ',\n]+"
                row = re.split(reg,row)
                if row==None or row[0]=='': continue
                print("row2: ",row)
                try:
                    sid = int(row[0])
                except:
                    print("it's not valid row: ",row)
                    continue
                # no more new data
                if sid>122:
                    if not self.extra_test:
                        continue
                # no more new data
                
                gender = row[3]
                age = int(re.findall(r'\d+', row[4])[0])
                unit = re.findall(r'[\u4e00-\u9fa5]', row[4])[0]
                if gender=='男':
                    gender=0.
                else:
                    gender=1.
                if unit == '月':
                    age = age/12.0
                inforbox=[gender,age]
                self.ages.append(age)
                    
                self.infors[sid]=inforbox
            print("self.infors: ",self.infors)
            self.ages=stats.zscore(self.ages)
            
            x=0
            for i,sid in enumerate(self.infors):
                
                self.infors[sid][1] = self.ages[x]
                x+=1
            print("self.infors: ",self.infors)
        
    def detail_process(self,sid,sub_path):   
        
        img, brain_mask_img = self.preprocessor.run(img_path=sub_path)
        print(sid,"_Spacing2: ",img.GetSpacing())
        
        img=self.normalizeImg(img,brain_mask_img)
        img_array = sitk.GetArrayFromImage(img)
        #if not self.istest:
            #print("This is train&valid box, use seg mask!")
        try:
            bbx_part_path= self.radiomics_pair_path.get(sid).get('maskpath')
            bbx_part_img,bbx_brain_mask_img = self.preprocessor.run(img_path=bbx_part_path)
            
            bbx_part_img=self.normalizeImg(bbx_part_img,bbx_brain_mask_img)
            bbx_part_img_array = sitk.GetArrayFromImage(bbx_part_img)
            #print("bbx_part_img_array: ",bbx_part_img_array)
            if self.sec_pair_orig:
                bbx_img_array=img_array*bbx_part_img_array # pair_orig
            else:
                bbx_img_array=bbx_part_img_array
            bbx_img = array2image(bbx_img_array,img)
        except:
            print("[bbx]: no bbx path")
            bbx_img = None
            bbx_part_img=None
        
        self.masked_img[sid] = bbx_img 
        self.third_img[sid] = bbx_part_img # must be bbx_img
        
        return img
        #img = array2image(img_array,img)
        #sitk.WriteImage(img,"/opt/chenxingru/test_playground/gt_nifts/"+f)
        
                

    # 预处理(只是导入数据、归一化，没有数据增强)
    def process(self):
        for i, f in enumerate(self.sub_fns):
            #if i>63:break
            fn = f.split("_")
            fn = fn[0].split(".")
            #get label: 0<sid<61: label =0 | 61<sid: label=1
            try:
                    sid=int(fn[0])
            except:
                print("it's the_select_file")
                continue
            if sid in self.delete_box:
                continue
            # no more new data
            if sid>122:
                if not self.extra_test:
                        continue
            if self.valid_sid_box!=None and not sid in self.valid_sid_box:
                continue
            # no more new data
            self.count+=1
            
            if self.multiclass:
                if sid<27 or (sid>127 and sid<131): 
                    slabel=0
                    self.y.append(0)
                    self.stastic[0]+=1
                elif sid<61 or (sid>122 and sid<128): 
                    slabel=1
                    self.y.append(1)
                    self.stastic[1]+=1
                else: 
                    slabel = 2
                    self.y.append(2)
                    self.stastic[2]+=1
            else:
                if sid<61 or (sid<131 and sid>122): 
                    slabel=0
                    self.y.append(0)
                    self.stastic[1]+=1
                else: 
                    slabel = 1
                    self.y.append(1)
                    self.stastic[2]+=1

            
            inforbox = self.infors.get(sid) 
            radiomics = []
            try:
                paths = self.radiomics_pair_path.get(sid)
                t1_sub_path = self.filedict[sid].get("T1")
                t2_sub_path = self.filedict[sid].get("T2")
                t1c_sub_path = self.filedict[sid].get("T1C")
                radiomics = self.radiomics_features.get(sid)
            except:
                print("No radiomics or T1 or T2")
                assert(0==1)
            
            print(sid)
            t1_img = self.detail_process(sid,t1_sub_path)
            t2_img = self.detail_process(sid,t2_sub_path)
            t1c_img = self.detail_process(sid,t1c_sub_path)
            
                                     
            original_data = (t1_img,t2_img,t1c_img, sid, slabel, inforbox,paths)
            self.__add__(sid,original_data)
            i+=1
    
    def normalizeImg(self,img, brain_mask_img):
        img_data=sitk.GetArrayFromImage(img)
        mask_data = sitk.GetArrayFromImage(brain_mask_img)
        brain_mask = mask_data > 0
        brain_mean = img_data[brain_mask].mean()
        brain_std = img_data[brain_mask].std()
        img_data = (img_data - brain_mean) / brain_std

        img= array2image(img_data,img)

        croporpad = tio.CropOrPad(
        (400,400,23),
        mask_name='brain_mask',
        )
        img = croporpad(img)
        return img

    def __len__(self):
        return len(self.dataset)


    def __add__(self,sid,other):
        self.dataset[sid]=other
        
    def _pad_or_crop_to_img_size(self, image, img_size, mode="constant"):
        """Image cropping to img_size
        """
        rank = len(img_size)
        
        # Create placeholders for the new shape
        from_indices = [[0, image.shape[dim]] for dim in range(rank)]  # [ [0, 0], [0, 1], [0, 2] ]
        to_padding = [[0, 0] for dim in range(rank)]
        
        slicer = [slice(None)] * rank
        
        # for each dimensions find whether it is supposed to be cropped or padded
        for i in range(rank):
            if image.shape[i] <= img_size[i]:
                to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
                to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
            else:
                slice_start = np.random.randint(0, image.shape[i] - img_size[i] + 1)
                from_indices[i][0] = slice_start
                from_indices[i][1] = from_indices[i][0] + img_size[i]
            
            # Create slicer object to crop or leach each dimension
            slicer[i] = slice(from_indices[i][0], from_indices[i][1])
        
        padded_img = np.pad(image[tuple(slicer)], to_padding, mode=mode, constant_values=0)
        #padded_seg = np.pad(seg[tuple(slicer)], to_padding, mode=mode, constant_values=0)
        
        return {
            "padded_img": padded_img
        }
    
    def getsid(self, data_idx=[]):
        sids=[]
        for i, idx in enumerate(data_idx):
            (origin_img,sid,slabel,sinforbox,paths) = self.dataset[idx]
            a1 = (origin_img,sid,slabel,sinforbox,paths)
            #print("idx: ",idx," ;sid: ",sid," ; paths: ",paths)
            sids.append(sid)
        return sids
            
            
    def select_dataset(self, data_idx=[], aug=True, aug_form="Random_Mask",use_secondimg=False,noSameRM=False,usethird=True,comman_total_file=[]):
        print("data_idx: ",data_idx)
        print("comman_total_file: ",comman_total_file)
        subdata=[]
        submaskimgs=[]
        suby=[]
        subfiles=[]
        thirdimg =[]
        sub_radiomics=[]
        sub_complete_file = []
        total_sid=[]
        for i, idx in enumerate(data_idx):
           total_sid.append(comman_total_file[idx])
           
        for i, idx in enumerate(data_idx):
            (t1_img,t2_img,t1c_img,sid,slabel,sinforbox,paths) = self.dataset[comman_total_file[idx]]
            a1 = (t1_img,t2_img,t1c_img,sid,slabel,sinforbox,paths)
            
            print("idx: ",idx," ;sid: ",sid,"; target: ",slabel," ; paths: ",paths)
            submaskimgs.append(self.masked_img[sid])
            subdata.append(a1)
            subfiles.append(self.files[sid])
            thirdimg.append(self.third_img[sid])
            
            if self.use_radiomics:
                sub_radiomics.append(torch.tensor(self.radiomics_features.get(sid),dtype=torch.float32))
        print("subdata: ",subdata)
        print("sub_radiomics: ",sub_radiomics)
                    
        idx_data = Sub_DiyFolder(subdata,subfiles,aug, aug_form,train_form = self.train_form,use_radiomics = self.use_radiomics,submaskimgs=submaskimgs,use_secondimg=use_secondimg,noSameRM=noSameRM,thirdimg= thirdimg,usethird =usethird,radiomics_feature=sub_radiomics,multi_modality= self.multi_M)
        return idx_data,total_sid
    
    def gety(self):
        return self.y

    def build_dict(self):
        t1_sub_fns = sorted(os.listdir(self.t1_path))
        t2_sub_fns = sorted(os.listdir(self.t2_path))
        t1c_sub_fns = sorted(os.listdir(self.t1c_path))
        self.append_dict(t1_sub_fns,"T1",self.t1_path)
        self.append_dict(t2_sub_fns,"T2",self.t2_path)
        self.append_dict(t1c_sub_fns,"T1C",self.t1c_path)

        print("self.filedict: ",self.filedict)
        for i, f in enumerate(self.sub_fns): 
            fn = f.split("_")
            fn = fn[0].split(".")
            try:
                    sid=int(fn[0])
            except:
                print("select file")
                continue
            # no more new data
            if sid>122:
                if not self.extra_test:
                    continue
            # no more new data            
            item = self.filedict[sid]
            t1_path=self.t1_path
            t2_path=self.t2_path
            t1c_path=self.t1c_path
            try:
                t1_path =  item['T1']
                t2_path =  item['T2']
                t1c_path =  item['T1C']
                self.count+=1      
            except:
                print("Got file missing!")
                self.delete_box.append(sid)
                continue  
            
            self.complete_file.append(sid)
                    
        print("All file number: ",self.count)
        print("Missing files [ T1/T2/T1C ]: ",self.delete_box) 
        print(self.filedict.keys())
        for i,sid in enumerate(self.delete_box):
            del self.filedict[sid]
        print(self.filedict.keys())
        print("self.complete_file: ",self.complete_file)
        return  self.complete_file, self.delete_box


    def append_dict(self,sub_fns,key_name,root):
        for i, f in enumerate(sub_fns):
            fn = f.split("_")
            fn = fn[0].split(".")
            try:
                    sid=int(fn[0])
            except:
                print("it's the_select_file")
                continue
            # no more new data
            if sid>122:
                if not self.extra_test:
                    continue
            # no more new data
            if  not sid in self.filedict:
                self.filedict[sid]={}
            self.filedict[sid][key_name]=os.path.join(root,f)
            print("self.filedict[sid][key_name] : ",self.filedict[sid][key_name])
  


# 测试
if __name__=="__main__":
        

    excel_path="/opt/chenxingru/Pineal_region/Pineal_0410.xlsx/"
    data_path="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/imagesTr/"
   


    transform_dict = {
        tio.RandomElasticDeformation(),
        tio.RandomNoise(),
        tio.RandomFlip(),
        tio.RandomBlur(),
        tio.RandomAffine(),
        tio.RandomMotion(),
        tio.RandomSwap()
        }



    data = DIY_Folder_3M(data_path=data_path, loader=nii_loader,train_form = "None",num_classes=2,root_mask_radiomics="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/labelsTr/",use_radiomics=False,istest=False,sec_pair_orig=False)
    print("data.form: ",data)
