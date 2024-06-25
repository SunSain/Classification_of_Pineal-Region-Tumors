from sklearn.model_selection import StratifiedKFold
from load_data_23 import DIY_Folder
import numpy as np
from utils2.DIY_radiomics import DiyRadiomics
import os
import utils2.config_3class as opt
import pandas as pd
import csv

k=3
splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)

# to record metrics: the best_acc of k folds, best_acc of each fold

#================= begin to train, choose 1 of k folds as validation =================================
print("======================== start train ================================================ \n")

data_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C"
test_data_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C"
train_radiomics_path="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task13_old_T1C/seg_all_results/train/"
test_radiomics_path="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task13_old_T1C/seg_all_results/test/"

a=[3,4,5,2,1,6]
print(sorted(a))

comman_total_file=[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]
y_box = [0 if i<61 else 1 for i in comman_total_file]



def get_img_files(root,root_mask_radiomics,common_box):#统计图像路径，方便之后直接从数组取
    files={}
    radiomics_pair_path={}
    sub_fns=sorted(os.listdir(root))
    for i, f in enumerate(sub_fns):

        fn = f.split("_")  
        fn = fn[0].split(".")
        try:
                sid=int(fn[0])
        except:
            print("it's the_select_file")
            continue
        if not sid in common_box:
            continue
        # no more new data
        if sid>122:
            continue
        # no more new data
        
        sub_path = os.path.join(root, f)
        files[sid]=(f)
        
        pair_file = {'imgpath':sub_path}
        radiomics_pair_path[sid]=pair_file
        
    mask_root = root_mask_radiomics
    if not os.path.exists(mask_root):
        print("[ERROR]: No Mask_path , plz check if you want to use Radiomics")
        return 
    print("sofar self.radiomics_pair_path",radiomics_pair_path)
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
        if not sid in common_box:
            continue
        if sid>122:
            continue
        # no more new data
        print("sid: ",sid)
        

        sub_path = os.path.join(mask_root, f)
        pair_file = radiomics_pair_path[sid]
        pair_file['maskpath']=sub_path
        radiomics_pair_path[sid] = pair_file
    print('self.radiomics_pair_path: ',radiomics_pair_path)
    return radiomics_pair_path

def write_radiomics_features(features,sids_box=[],all_or_split=True,save_root="",filename=""):
    
    common_box_dir={'train':[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122],
                    'test':[3, 4, 8, 16, 20, 22, 25, 26, 38, 45, 47, 56, 64, 65, 66, 75, 93, 94, 98, 99, 101, 111, 120]}
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    common_box=sids_box
    new_box=[]
    for i, sid in enumerate(common_box):
        feature=[sid]
        feature.extend(features.get(sid))
        if sid<61:
            feature.append(0)
        else:
            feature.append(1)
        print("len(feature): ",len(feature))
        print("feature: ",feature)
        new_box.append(feature)
    print("len(new_box): ",len(new_box))
    new_box=pd.DataFrame(data=new_box)
    new_box.to_csv(os.path.join(save_root,filename),mode='w',index=None,header=None)
    

def get_radiomics(radiomics_pair_path): #获得所有图像的radiomics
    if radiomics_pair_path == []:
        print("No radiomics...")
        return
    my_radiomics = DiyRadiomics(radiomics_pair_path)
    radiomics_features = my_radiomics.get_features()
    empty_radiomics_box = my_radiomics.get_empty_box()
    
    try:
        first_img_feature = radiomics_features[0]
        print("PyRadiomics Result: ",first_img_feature)
    except:
        print("No Radiomics fearure get!")
    return radiomics_features,empty_radiomics_box


#total_file = DIY_Folder(num_classes =  3,data_path=data_path,train_form =  None,root_mask_radiomics =  "",use_radiomics= False,istest=False,sec_pair_orig= False,multiclass =  False,vice_path = "")
if __name__=="__main__":
    
    """
    total_file = DIY_Folder(num_classes = 2,data_path=data_path,train_form = None,root_mask_radiomics ="",use_radiomics=False,istest=False,sec_pair_orig=False,multiclass = False,vice_path = "")

    total_comman_total_file=[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]
    y_box = [0 if i<61 else 1 for i in total_comman_total_file] 
    k=3
    splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
    
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(total_comman_total_file)),y_box)):
        print("Get valid set")
        vali_data,vali_sids=total_file.select_dataset(data_idx=val_idx, aug=False,use_secondimg=False,noSameRM=False, usethird=False,comman_total_file=total_comman_total_file)
        print("Got train set")
        print('vali_sids: ',vali_sids)
        train_data,train_sids=total_file.select_dataset(data_idx=train_idx, aug=True,aug_form=None,use_secondimg=False,noSameRM=False, usethird=False,comman_total_file=total_comman_total_file)
        print('train_data: ',train_sids)
        print("empty_radiomics_box: ",total_file.empty_radiomics_box)

    """
    
    
    text = '''FKEY,TYPE,EFF,DATA
    34787,2,Y,2022.03.20,1088825
    14787,1,Y,2022.03.20,1088825
    34787,2,Y,2022.03.20,1088825
    14787,1,Y,2022.03.20,1088825
    14787,1,Y,2022.03.20,1088825
    34787,2,Y,2022.03.20,1088825
    14787,1,Y,2022.03.20,1088825
    14787,1,Y,2022.03.20,1088825'''

    typedata = '2'

    #with open('d:/Python/testfile.csv') as in_file, open('d:/Python/Data.csv', 'w') as out_file:
    in_file = text.splitlines()
    reader = csv.reader(in_file)
    #writer = csv.writer(out_file)
    for row in reader:
        if int(row['FKEY']) >2000:
            print(row)
        if typedata not in row:
            print(row)
            #writer.writerow(row)
    """
    root = data_path
    root_mask_radiomics = train_radiomics_path
    common_box=[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]
    radiomics_pair_path=get_img_files(root=root, root_mask_radiomics=root_mask_radiomics,common_box=common_box)
    
    radiomics_features,empty_radiomics_box =get_radiomics(radiomics_pair_path)
    write_radiomics_features(features=radiomics_features,sids_box=common_box,all_or_split=True,save_root="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/",filename="all_train.csv")


    common_box=[10, 11, 12, 13, 14, 21, 27, 31, 36, 37, 40, 44, 51, 57, 58, 63, 68, 72, 79, 80, 85, 86, 92, 102, 105, 112, 113, 116, 118, 119, 122]
    #radiomics_features,empty_radiomics_box =get_radiomics(radiomics_pair_path)
    write_radiomics_features(features=radiomics_features,sids_box=common_box,all_or_split=True,save_root="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/",filename="fold_0_valid.csv")

    common_box=[2, 5, 15, 18, 19, 24, 28, 30, 43, 48, 52, 54, 55, 59, 62, 69, 74, 77, 81, 82, 88, 90, 96, 97, 104, 108, 114, 115, 117, 121]
    #radiomics_features,empty_radiomics_box =get_radiomics(radiomics_pair_path)
    write_radiomics_features(features=radiomics_features,sids_box=common_box,all_or_split=True,save_root="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/",filename="fold_1_valid.csv")

    common_box=[1, 6, 7, 9, 17, 23, 29, 34, 35, 39, 42, 46, 49, 60, 61, 67, 70, 71, 73, 76, 78, 84, 87, 91, 95, 100, 106, 107, 109, 110]
    #radiomics_features,empty_radiomics_box =get_radiomics(radiomics_pair_path)
    write_radiomics_features(features=radiomics_features,sids_box=common_box,all_or_split=True,save_root="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/",filename="fold_2_valid.csv")


    common_box=[1, 2, 5, 6, 7, 9, 15, 17, 18, 19, 23, 24, 28, 29, 30, 34, 35, 39, 42, 43, 46, 48, 49, 52, 54, 55, 59, 60, 61, 62, 67, 69, 70, 71, 73, 74, 76, 77, 78, 81, 82, 84, 87, 88, 90, 91, 95, 96, 97, 100, 104, 106, 107, 108, 109, 110, 114, 115, 117, 121]
    #radiomics_features,empty_radiomics_box =get_radiomics(radiomics_pair_path)
    write_radiomics_features(features=radiomics_features,sids_box=common_box,all_or_split=True,save_root="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/",filename="fold_0_train.csv")

    common_box= [1, 6, 7, 9, 10, 11, 12, 13, 14, 17, 21, 23, 27, 29, 31, 34, 35, 36, 37, 39, 40, 42, 44, 46, 49, 51, 57, 58, 60, 61, 63, 67, 68, 70, 71, 72, 73, 76, 78, 79, 80, 84, 85, 86, 87, 91, 92, 95, 100, 102, 105, 106, 107, 109, 110, 112, 113, 116, 118, 119, 122]
    #radiomics_features,empty_radiomics_box =get_radiomics(radiomics_pair_path)
    write_radiomics_features(features=radiomics_features,sids_box=common_box,all_or_split=True,save_root="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/",filename="fold_1_train.csv")

    common_box=[2, 5, 10, 11, 12, 13, 14, 15, 18, 19, 21, 24, 27, 28, 30, 31, 36, 37, 40, 43, 44, 48, 51, 52, 54, 55, 57, 58, 59, 62, 63, 68, 69, 72, 74, 77, 79, 80, 81, 82, 85, 86, 88, 90, 92, 96, 97, 102, 104, 105, 108, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]
    #radiomics_features,empty_radiomics_box =get_radiomics(radiomics_pair_path)
    write_radiomics_features(features=radiomics_features,sids_box=common_box,all_or_split=True,save_root="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/",filename="fold_2_train.csv")


    root = test_data_path
    root_mask_radiomics = test_radiomics_path
    common_box=[3, 4, 8, 16, 20, 22, 25, 26, 38, 45, 47, 56, 64, 65, 66, 75, 93, 94, 98, 99, 101, 111, 120]
    radiomics_pair_path=get_img_files(root=root, root_mask_radiomics=root_mask_radiomics,common_box=common_box)
    
    radiomics_features,empty_radiomics_box =get_radiomics(radiomics_pair_path)
    write_radiomics_features(features=radiomics_features,sids_box=common_box,all_or_split=True,save_root="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/",filename="all_test.csv")
    """
  
#[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 46, 48, 49, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]
#[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]
#[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]