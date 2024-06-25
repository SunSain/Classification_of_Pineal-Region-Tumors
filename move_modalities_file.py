
import os
from shutil import copyfile
from create_four_modalityfile import Allfile

def filename_del(target_dir,keepIndex):
    try:
        for filename in os.listdir(target_dir):
            # file = os.path.splitext()

            # print(type(filename))
            index = filename[-8]
            print("index: ",index)
            if index == keepIndex:
                os.remove(target_dir + filename)
    except:
        print('error')
        
def file_extract(target_dir,keepIndex,saveroot):
    try:
        for filename in os.listdir(target_dir):

            index = filename[-8]
            print("index: ",index)
            if index == keepIndex:
                old_name=os.path.join(target_dir,filename)
                print(old_name)
                print("filename: ",filename)
                newName=filename.replace("0002","0000")
                print("newName: ",newName)
                new_name=os.path.join(saveroot,newName)
                if not os.path.exists(saveroot):
                    os.mkdir(saveroot)
                print("old_name: ",old_name)
                print("new_name: ",new_name)
                copyfile(old_name, new_name)
                #os.remove(target_dir + filename)
    except:
        print('error')

def file_rename(path):
    sub_fns=sorted(os.listdir(path))
    for i, f in enumerate(sub_fns):
        old_name = os.path.join(path, f)
        fn = f.split("_")
        try:
            sid=int(fn[0])
        except:
            print("it's slected file!")
        newName = fn[0]+"_"+fn[-1]
        new_name=os.path.join(path,newName)
        print("new_name: ",new_name)
        os.rename(old_name, new_name)
 
def file_repalce_name(target_dir,keepIndex="2"):
    try:
        for filename in os.listdir(target_dir):

            index = filename[-8]
            print("index: ",index)
            if index == keepIndex:
                old_name=os.path.join(target_dir,filename)
                print(old_name)
                print("filename: ",filename)
                newName=filename.replace("0002","0000")
                print("newName: ",newName)
                new_name=os.path.join(target_dir,newName)
                copyfile(old_name, new_name)
    except:
        print('error')
        
def file_move(idx_box,path,new_root):
    sub_fns=sorted(os.listdir(path))
    for i, f in enumerate(sub_fns):
        old_path = os.path.join(path, f)
        fn = f.split("_")
        try:
            sid=int(fn[0])
        except:
            print("it's slected file!")
        if sid in idx_box:
            new_path = os.path.join(new_root,f)
            copyfile(old_path, new_path)
        else:
            print("not in idx_box: ",fn[0])
       
def  move_files(idx_box,flair_path, t1_path,t2_path,t1c_path,new_root):
    file_move(idx_box, flair_path, new_root)
    file_move(idx_box, t1_path, new_root)
    file_move(idx_box, t2_path, new_root)
    file_move(idx_box, t1c_path, new_root)
                  
 
if __name__=="__main__":
    path="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task000_T1-FLAIR/"
    #filename_del(path+'imagesTr/', '1')
    #filename_del(path+'imagesTs/', '1')
    #filename_del(path+'imagesVal/', '1')
    #file_rename(path+'imagesTs/')
    t1_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task001_t1/"
    t2_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task002_t2/"
    t1c_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task003_t1c/"
    #allfile = Allfile(t1_path+'imagesTs/',t2_path+'imagesTs/',t1c_path+'imagesTs/',"")
    #conjunct_file = allfile.build_4_modality_file()
    #print("conjunct_file: ",conjunct_file)
    #flair_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task000_T1-FLAIR/"
    #new_root = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task004_MyTotal/imagesTs/"
    #if not os.path.exists(new_root):
        #os.mkdir(new_root)
    #move_files(conjunct_file,flair_path+'imagesTs/' ,t1_path+'imagesTs/',t2_path+'imagesTs/',t1c_path+'imagesTs/',new_root)
    total_path="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task007_MyTotal/imagesTs/"
    
    saveroot="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task010_BTumour_T1C/labelsTs/"   
    
    public_data_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task008_BrainTumour/"
    #file_repalce_name(public_data_path+'imagesTr/', '2')
    #file_repalce_name(public_data_path+'imagesTs/', '2')
    #file_repalce_name(public_data_path+'imagesVal/', '2')
    file_extract(public_data_path+'labelsTs/',"2",saveroot)
