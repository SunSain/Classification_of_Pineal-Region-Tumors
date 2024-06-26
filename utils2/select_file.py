
import os
import shutil

def find_file(filepath,f):
    box=[]
    if os.path.isdir(filepath):
        sub_fns = sorted(os.listdir(filepath))
        for i, f in enumerate(sub_fns):
            vice = os.path.join(filepath,f)
            single_file = find_file(vice,f)
            box.extend(single_file)
    elif not os.path.isfile(filepath):
        return []
    else:
        box.append({'filepath':filepath,'filename':f})
        return box

def simply_select_file(filepath, saveroot):
    sub_fns = sorted(os.listdir(filepath))
    for i, f in enumerate(sub_fns):
        vice = os.path.join(filepath,f)
        file = find_file(vice)
        filepath = file[0].get('filepath')
        f = file[0].get('filename')
        new_path = os.path.join(saveroot,f)
        shutil.copy(filepath,new_path)

def correct_lastfix(filepath, savepath):
    sub_fns = sorted(os.listdir(filepath))
    for i, f in enumerate(sub_fns):
        new_f = f
        if f.endswith(".nii"):
            new_f = f+".gz"
        if "-"in new_f:
            print("y")
            new_f=new_f.replace("-","_")

        orig_path = os.path.join(filepath,f)
        new_path = os.path.join(savepath,new_f)
        shutil.copy(orig_path,new_path)

def spilt_train_valid(filepath,saveroot,box,ps):
    saveroot_train = os.path.join(saveroot,"test"+ps)
    saveroot_valid = os.path.join(saveroot,"train"+ps)
    
    if not os.path.exists(saveroot_train):
        os.mkdir(saveroot_train)
    if not os.path.exists(saveroot_valid):
        os.mkdir(saveroot_valid)
    subfns= sorted(os.listdir(filepath))
    for i, f in enumerate(subfns):
        if not f.endswith(".nii.gz") and not f.endswith(".nii"):
            print("invalid file ",f)
            continue
        fns = f.split("_")
        fns = fns[0].split(".")
        try: sid = int(fns[0])
        except:
            print("invalid file ",f)
            continue
        if sid in box:
            savepath = os.path.join(saveroot_train,f)
        else:
            savepath = os.path.join(saveroot_valid,f)
        origpath = os.path.join(filepath,f)
        shutil.copyfile(origpath,savepath)
               
def rename_file(root,f):
    fns = f.split("_")
    fns_0 = fns[0].split(".")
    print("fns_0: ",fns_0)
    try:
        sid = int(fns_0[0])
    except:
        print("invalid file ",f)
        return 
    new_str_sid = ""
    if sid<10:
        new_str_sid="00"+str(sid)+".nii.gz"
    elif sid<100:
        new_str_sid="0"+str(sid)+".nii.gz"
    else:
        new_str_sid = str(sid)+".nii.gz"
    my_source = os.path.join(root,f)
    my_dest = os.path.join(root,new_str_sid)
    os.rename(my_source, my_dest)

def rename_ends_file(root,f):
    fns = f.split("_")
    fns_0 = fns[0].split(".")
    print("fns_0: ",fns_0)
    try:
        sid = int(fns_0[0])
    except:
        print("invalid file ",f)
        return 
    new_str_sid = ""
    if not f.endswith("_0000.nii.gz"):
        new_str_f = f.replace(".nii.gz","_0000.nii.gz")
    else:
        return 
    my_source = os.path.join(root,f)
    my_dest = os.path.join(root,new_str_f)
    os.rename(my_source, my_dest)


def get_box(filepath):
    box=[]
    subfns= sorted(os.listdir(filepath))
    for i, f in enumerate(subfns):
        fns = f.split("_")
        fns_0 = fns[0].split(".")
        print("fns_0: ",fns_0)
        try: sid = int(fns[0])
        except:
            print("invalid file ",f)
            continue
        box.append(sid)
    return box
        

if __name__=="__main__":

    #simply_select_file() 
    #correct_lastfix(filepath, saveroot)
    #box= [11, 19,34,42,44,56,64,76,79,112,114,117,121,143,153,2,3,24,27,41,46,49,57,58,68,71,92,104,144,152]
    """
    test_path = "/opt/chenxingru/Pineal_region/11_23_No_regis_T1_best_TEST/"
    saveroot="/opt/chenxingru/Pineal_region/after_12_08/0704_data/"
    t1_path = "/home/chenxr/new_nii_data/all_T1"
    t1_ps = "_T1"
    t2_path = "/home/chenxr/new_nii_data/all_T2"
    t2_ps = "_T2"
    t1c_path = "/home/chenxr/new_nii_data/all_T1C"
    t1c_ps = "_T1C"
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)
        
    box = get_box(test_path)
    print("len(box): ",len(box))
    spilt_train_valid(t1_path,saveroot,box,t1_ps)
    spilt_train_valid(t2_path,saveroot,box,t2_ps)
    spilt_train_valid(t1c_path,saveroot,box,t1c_ps)
    """
    """
    filepath="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task012_all_T1C/for_classifying/"
    path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C"
    saveroot="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task012_all_T1C/for_classifying/"
    box = get_box(path)
    print(box)
    print("len(box): ",len(box))
    spilt_train_valid(filepath,saveroot,box,ps="")
    
    
    """
    root="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task13_old_T1C/imagesTs/"
    root_fns = sorted(os.listdir(root))
    for i,f in enumerate(root_fns):
        rename_ends_file(root, f)  

        
        
        
#nnUNet_predict -i /opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task012_all_T1C/imagesTs/ -o /opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task012_all_T1C/inferTs/ -m 3d_fullres -t 12   
        