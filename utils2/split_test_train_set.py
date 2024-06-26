
import os
import shutil

train_sid_box=[]

def get_train_sid_box():
<<<<<<< HEAD
    path = "/opt/chenxingru/Pineal_region/after_12_08/230305_data/12_03_No_regis_T1+C_best_Notest/"
=======
    #path = "/opt/chenxingru/Pineal_region/after_12_08/230305_data/12_03_No_regis_T1+C_best_Notest/"
    # no more new data
    path = "/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C"
>>>>>>> 3a4a4f2 (20240625-code)
    sub_fns = sorted(os.listdir(path))
    for i, f in enumerate(sub_fns):
            old_path = os.path.join(path,f)
            fn = f.split("_")
            #get label: 0<sid<61: label =0 | 61<sid: label=1
            try:
                    sid=int(fn[0])
            except:
                print("it's the_select_file")
                continue
            train_sid_box.append(sid)
    print("train_sid_box: ",train_sid_box)

def split_files(root_path, save_train_path, save_test_path):
    check_box=[]
    test_box=[]
    if not os.path.exists(save_test_path):
        os.makedirs(save_test_path)
    if not os.path.exists(save_train_path):
        os.makedirs(save_train_path)
        
    sub_fns = sorted(os.listdir(root_path))
    for i, f in enumerate(sub_fns):
            old_path = os.path.join(root_path,f)
            fn = f.split("_")
<<<<<<< HEAD
            #get label: 0<sid<61: label =0 | 61<sid: label=1
            try:
                    sid=int(fn[0])
            except:
                print("it's the_select_file")
                continue
=======
            fnn = fn[0].split(".")
            #get label: 0<sid<61: label =0 | 61<sid: label=1
            try:
                    sid=int(fnn[0])
            except:
                print("it's the_select_file")
                continue
            print("sid: ",sid)
>>>>>>> 3a4a4f2 (20240625-code)
            if sid in train_sid_box:
                new_path = os.path.join(save_train_path,f)
                shutil.copyfile(old_path,new_path)
                check_box.append(sid)
            else:
                new_path = os.path.join(save_test_path,f)
                shutil.copyfile(old_path,new_path)
                test_box.append(sid)
    print("test_box: ",test_box)
<<<<<<< HEAD
    if check_box !=train_sid_box:
        print("Miss Something!")
        print("check_box: ",check_box)
        print("train_sid_box: ",train_sid_box)
=======
    if set(check_box) != set(train_sid_box):
        print("Miss Something!")
        print("check_box: ",set(check_box))
        print("train_sid_box: ",set(train_sid_box))
>>>>>>> 3a4a4f2 (20240625-code)
            

if __name__ =="__main__":
    get_train_sid_box()

<<<<<<< HEAD
=======
    """
>>>>>>> 3a4a4f2 (20240625-code)
    orig_root_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/imagesTs0_noSkullStrip/"
    save_train_path="/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_No_regis_T1+C_best_Notest/"
    save_test_path = "/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_No_regis_T1+C_best_Test/"
    
    split_files(orig_root_path,save_train_path,save_test_path)


    skullstriped_root_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/imagesTs_skullStripped/"
    save_train_path="/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_skullstriped_No_regis_T1+C_best_Notest/"
    save_test_path = "/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_skullstriped_No_regis_T1+C_best_Test/"
    
    split_files(skullstriped_root_path,save_train_path,save_test_path)

    seg_root_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/inferTs_skullStripped/"
    save_train_path="/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_seg_No_regis_T1+C_best_Notest/"
    save_test_path = "/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_seg_No_regis_T1+C_best_Test/"
    
    split_files(seg_root_path,save_train_path,save_test_path)


    boundingbox_root_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/masked_bounding/"
    save_train_path="/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_bbox_No_regis_T1+C_best_Notest/"
    save_test_path = "/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_bbox_No_regis_T1+C_best_Test/"
    
    split_files(boundingbox_root_path,save_train_path,save_test_path)
<<<<<<< HEAD

=======
    """
    
    seg_root_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task13_old_T1C/seg_all_results/"
    save_train_path="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task13_old_T1C/seg_all_results/train/"
    save_test_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task13_old_T1C/seg_all_results/test/"
    split_files(seg_root_path,save_train_path,save_test_path)
>>>>>>> 3a4a4f2 (20240625-code)

