

import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from utils2.preprocessor import Preprocessor,array2image
import torchio as tio

class Allfile():
    def __init__(self,t1path,t2path,t1cpath,saveroot) :
        self.t1path,self.t2path,self.t1cpath ,self.saveroot= t1path,t2path,t1cpath,saveroot
        self.filedict = {}
        self.count=0
        self.complete_file=[]
        self.missing_T1=[]
        self.missing_T2=[]
        self.missing_T1c=[]
        self.build_dict()
        
    def build_dict(self):
        t1_sub_fns = sorted(os.listdir(self.t1path))
        t2_sub_fns = sorted(os.listdir(self.t2path))
        t1c_sub_fns = sorted(os.listdir(self.t1cpath))
        self.append_dict(t1_sub_fns,"T1",self.t1path)
        self.append_dict(t2_sub_fns,"T2",self.t2path)
        self.append_dict(t1c_sub_fns,"T1c",self.t1cpath)


    def append_dict(self,sub_fns,key_name,root):
        for i, f in enumerate(sub_fns):
            fn = f.split("_")
            try:
                    sid=int(fn[0])
            except:
                print("it's the_select_file")
                continue
            if  not sid in self.filedict:
                self.filedict[sid]={}
            self.filedict[sid][key_name]=os.path.join(root,f)
            print("self.filedict[sid][key_name] : ",self.filedict[sid][key_name])
    
        
    def build_4_modality_file(self):
        #origin_imag_path="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task01_BrainTumour/imagesTr/BRATS_484.nii.gz"
        #origin_composed_img = sitk.ReadImage(origin_imag_path)
        delete_box=[]
        t1c_box=[]
        print("self.filedict: ",self.filedict)
        for i, idx in enumerate(self.filedict): 
            #print("idx: ",idx)
            item = self.filedict[idx]
            t1_path=""
            t2_path=""
            t1c_path=""
            try:
                t1_path =  item['T1']
                t2_path =  item['T2']
                t1c_path =  item['T1c']
                self.count+=1    
                t1c_box.append(idx)  
            except:
                print("Got file missing!")
                if "T2" in item:
                    t1c_box.append(idx)
                    print("idx: ",idx)
                    if "T1c" not in item or "T1" not in item:
                        self.missing_T1.append(idx)
                        delete_box.append(idx)
                if "T2" not in item:
                    self.missing_T1c.append(idx)
                    delete_box.append(idx)
                continue  
            
            self.complete_file.append(idx)
                    
        print("All file number: ",self.count)
        print("Missing files [ T1c ]: ",self.missing_T1)                     
        print("Missing files [ T2 ]: ",self.missing_T2)  
        print("Missing files [ T1 ]: ",self.missing_T1c)
        print("len all: ",len(self.complete_file))
        print("len delete: ",len(delete_box))
        print("t1c_box: ",t1c_box)
        print(self.filedict.keys())
        nothing = [0, 3, 4, 8, 16, 20, 22, 25, 26, 38, 45, 47, 50, 53, 56, 64, 65, 66, 75, 89, 93, 94, 98, 99, 101, 111, 120]
        new_box=[]
        print("nothing: ",nothing)
        for i, idx in enumerate(nothing):
            if idx not in t1c_box:
                new_box.append(idx)
                
        print("missing files: ",new_box)
        for i,idx in enumerate(delete_box):
            del self.filedict[idx]
        print(self.filedict.keys())
        return  self.complete_file
                 
    def get_array_from_img(self,path):
        img = sitk.ReadImage(path)
        img = sitk.DICOMOrient(img, 'LPS')
        print("origin_img.data: ",img.GetSize()," ; spacing: ",img.GetSpacing())
        print("img.Direction() : ",img.GetDirection())

        img = sitk.GetArrayFromImage(img).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img, dtype= np.float32)
        return img
            
def show_file_info(path):
    sub_fns = sorted(os.listdir(path))
    for i, f in enumerate(sub_fns):
            fn = f.split("_")
            img_path = os.path.join(path, f)
            img = sitk.ReadImage(img_path)
            print("[%s]: "%(fn[0])," ; size: ",img.GetSize()," ; spacing: ",img.GetSpacing())
                
def crop_file(path, save_root):
    sub_fns = sorted(os.listdir(path))
    for i, f in enumerate(sub_fns):
            fn = f.split("_")
            img_path = os.path.join(path, f)
            img = sitk.ReadImage(img_path)
            #print("[%s]: "%(fn[0])," ; size: ",img.GetSize()," ; spacing: ",img.GetSpacing())
            croporpad = tio.CropOrPad(
                    (430,430,23)
                    )
            img = croporpad(img)
            savepath = os.path.join(save_root,f)
            if not os.path.exists(saveroot):
                os.mkdir(saveroot)
            sitk.WriteImage(img,savepath)
    
def count_file_num(path):
    sub = sorted(os.listdir(path))
    count = 0
    class_count=[0,0,0]
    for i, f in enumerate(sub):
        fn = f.split("_")
        try:
            sid = int(fn[0])
        except:
            print("select file")
            continue
        if sid<27:
            class_count[0]+=1
        elif sid <61:
            class_count[1]+=1
        else:
            class_count[2]+=1
        count+=1
    print("total count: ",count)
    print("sub_class: ",class_count)

if __name__=="__main__":
    t1_path="/opt/chenxingru/Pineal_region/after_12_08/230305_data/11_23_No_regis_T1_best_Notest_No103/"
    t2_path="/opt/chenxingru/Pineal_region/after_12_08/230305_data/12_03_No_regis_T2_best_Notest/"
    t1c_path="/opt/chenxingru/Pineal_region/after_12_08/230305_data/12_03_No_regis_T1+C_best_Notest/"
    saveroot="/opt/chenxingru/Pineal_region/after_12_08/230305_data/new_img/"
   
    t1c_test_path = "/opt/chenxingru/Pineal_region/after_12_08/230305_data/12_03_No_regis_T1+C_TEST_No22/"
    #t1c_path=t1c_test_path
    t1_test_path="/opt/chenxingru/Pineal_region/11_23_No_regis_T1_best_TEST/"
    #t1_path=t1_test_path
    orig_path = "/opt/chenxingru/Pineal_region/0509_fresh_all_data/select_every_kind/T2_detailed_select_0528/AX_and_TRA/"
    #count_file_num(orig_path)
    t2_test_path="/opt/chenxingru/Pineal_region/after_12_08/230305_data/12_03_No_regis_T2_best_TEST/"
    t2_path = t2_test_path
    allfile = Allfile(t1_path,t2_path,t1c_path,saveroot)
    allfile.build_4_modality_file()
    
    path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task004_MyTotal/imagesTs/"
    #show_file_info(path)
    saveroot = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task007_MyTotal/imagesTs/"
    #crop_file(path, saveroot)
    #show_file_info(saveroot)