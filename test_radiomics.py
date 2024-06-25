from utils2.DIY_radiomics import DiyRadiomics
import os
 
def get_img_files(root,mask_root):#统计图像路径，方便之后直接从数组取
        radiomics_pair_path={}
        sub_fns = sorted(os.listdir(root))
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
                continue
            if sid==53:
                continue
            if sid<97:
                continue
            sub_path = os.path.join(root, f)
            
            pair_file = {'imgpath':sub_path}
            radiomics_pair_path[sid]=pair_file
            
 
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
            if sid>122:
                continue
            if sid<97:
                continue
            # no more new data
            sub_path = os.path.join(mask_root, f)
            pair_file = radiomics_pair_path[sid]
            pair_file['maskpath']=sub_path
            radiomics_pair_path[sid] = pair_file
        return radiomics_pair_path
        print('self.radiomics_pair_path: ',radiomics_pair_path)

if __name__=="__main__":
        

    excel_path="/opt/chenxingru/Pineal_region/Pineal_0410.xlsx/"
    data_path="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/imagesTr/"
   
    root='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C'
    mask_root="/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_seg_No_regis_T1+C_best_Notest"
    filelist=get_img_files(root,mask_root)
    
    savepath="/home/chenxr/radiomics/train_T1C"
    my_radiomics = DiyRadiomics(filelist,savepath)
    radiomics_features = my_radiomics.get_features()
    empty_radiomics_box = my_radiomics.get_empty_box()
    print("PyRadiomics Result1: ",radiomics_features)
    try:
        first_img_feature =  radiomics_features[0]
        print("PyRadiomics Result2: ",first_img_feature)
    except:
        print("ERROR: ",i)