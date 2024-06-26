import SimpleITK as sitk
import os
import pydicom
import re
import csv
import traceback
import shutil
import sys
print("sys_path: ",sys.path)

MriData_categories=['T1_cor','T1_sag','T1_ax','T1_tra','T2_cor','T2_sag','T2_ax','T2_tra','T1+c_cor','T1+c_sag','T1+c_ax','T1+c_tra','T1','T2','T1+c','others','total']

   
def save_nii(every_MRI,folderPath, total_save_path,every_patient,f,base): 
    box_num=0 
    illegal=0
    #print("folderpath: ",folderPath)
    pp_total_sta=[0 for i in range(len(MriData_categories))]
        
    step_blogs=re.split("_",every_MRI)
    if step_blogs[1]=="cxr":
        print("cxr need to be changed")
        step_blogs[1]=every_patient.split("_")[1]
    print("step_blog: ",step_blogs," ;every_MRI: ",every_MRI)
    #f.writelines("      _____MRI_date: ",step_blogs,"_____\n")

    f.writelines("  [everyMRI: "+folderPath+"/" +str("_".join(step_blogs))+"]: name_length: "+ str(len(step_blogs))+"\n")        
    

    print("\n[Ridiculous]\n")
    tmp_path = folderPath  
    if not os.path.exists(total_save_path):
        print("\n[built0]\n")
        os.makedirs(total_save_path)
        print("\n]]\n")

    
    t1_path=os.path.join(total_save_path,'T1W')
    if not os.path.exists(t1_path):
        print("\n[built1]\n")
        os.makedirs(t1_path)
        print("\n]]\n")

    t2_path=os.path.join(total_save_path,'T2')
    if not os.path.exists(t2_path):
        print("\n[built2]\n")
        os.makedirs(t2_path)
        print("\n]]\n")

    t1c_path=os.path.join(total_save_path,'T1+C')
    if not os.path.exists(t1c_path):
        print("\n[built3]\n") 
        os.makedirs(t1c_path)
        print("\n]]\n")

    other_path=os.path.join(total_save_path,'Others')
    if not os.path.exists(other_path):
        print("\n[built4]\n")   
        os.makedirs(other_path)
        print("\n]]\n")
    every_patient = str(base)+"_"+every_patient
    print("\n[Ridiculous]\n")
    if"t1+c" in every_MRI or"+c" in every_MRI:
        f.writelines("          [T1+C] :%s"%every_MRI)
        path_save = t1c_path+'/'+every_patient+'/'+every_MRI # chenxingru/.../001_liu/7397192/3_2_T1.nii.gz
        dir_save= os.path.join(t1c_path,every_patient)
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        print("\n[T1+C]  tmp_path: "+tmp_path+" \n save_file:  "+path_save)
        shutil.copyfile(tmp_path,path_save)
    elif "t1" in every_MRI or"t1w" in every_MRI:
        f.writelines("          [T1w] :%s"%every_MRI)
        path_save = t1_path +'/'+every_patient+'/'+every_MRI # chenxingru/.../001_liu/7397192/3_2_T1.nii.gz
        dir_save= os.path.join(t1_path,every_patient)
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)        
        print("\n[T1]  tmp_path: "+tmp_path+" \n save_file:  "+path_save)
        shutil.copyfile(tmp_path,path_save) 
    elif "t2"in every_MRI  or "t2w" in every_MRI:
        path_save = t2_path +'/'+every_patient+'/'+every_MRI # chenxingru/.../001_liu/7397192/3_2_T1.nii.gz
        dir_save= os.path.join(t2_path,every_patient)
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        print("\n[T2]  tmp_path: "+tmp_path+" \n save_file:  "+path_save)
        f.writelines("          [T2] :%s"%every_MRI)           
        shutil.copyfile(tmp_path,path_save)
    else:
        path_save = other_path +'/'+every_patient+'/'+every_MRI # chenxingru/.../001_liu/7397192/3_2_T1.nii.gz
        dir_save= os.path.join(other_path,every_patient)
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        print("\n[Others]  tmp_path: "+tmp_path+" \n save_file:  "+path_save)
        f.writelines("          [NotThem] :%s"%every_MRI)
        shutil.copyfile(tmp_path,path_save)
    print("\n[Byebye]\n")

no_tra_t1c_box=[]
    
def select_t1c_oax_tra(filepath, savepath):
    for every_patient in os.listdir(filepath):
        maxvol = 0
        mrifile = None
        vice_path = os.path.join(filepath,every_patient)
        print("vice_path: ",vice_path)
        patient_name = every_patient.split("_")[2]
        baseID = every_patient.split("_")[0]
        print("patient_name: ",patient_name,"baseID: ",baseID)
        for every_item in os.listdir(vice_path):
            descri = every_item.split("_")
            print("descri: ",descri)
            if 'ax' in descri or 'oax' in descri or 'tra' in descri or 'otra' in descri:
                mrifilepath = os.path.join(vice_path,every_item)
                size = os.path.getsize(mrifilepath)
                if size>=maxvol:
                    mrifile = every_item
                    maxvol = size
        if mrifile==None:
            no_tra_t1c_box.append(every_patient)
            print("There is no tra/ax")
            continue
        mrifilepath = os.path.join(vice_path,mrifile)
        try:
            mrifile="01_01_"+mrifile
            blogs = mrifile.split("_")
            print("mrifile: ",mrifile)
            print("every_patient: ",every_patient)
            blogs[0]=every_patient.split("_")[0]
            blogs[1]=every_patient.split("_")[1]
            blogs[3]=every_patient.split("_")[2]
            mrifile="_".join(blogs)
            print("mrifile2: ",mrifile)
        except:
            print("There is no cxr, "+patient_name)
        
        save_path = os.path.join(savepath,mrifile)
        shutil.copyfile(mrifilepath,save_path)
            
            
        


        
        

if __name__ == '__main__':
 
    
    all_file="/home/chenxr/new_NII/select_every_kind/T1W/"
    total_save_path="/home/chenxr/new_NII/T1_ax_tra/"
    if not os.path.exists(total_save_path):
        os.makedirs(total_save_path)
    
    select_t1c_oax_tra(all_file,total_save_path)
    print("no_tra_t1c_box: ",no_tra_t1c_box)
    
    """
    f = open("/home/chenxr/new_NII/others_MRI_info_T1_T2_T1+C.txt", 'w')
    f.writelines("2023/06/23\n")

    Becount=0
    Count = 0

    i=0

    base = 122
    
    for category in os.listdir(all_file):
        
        
        total_file_path = os.path.join(all_file,category)
        print("category: ",category)
        if not os.path.exists(total_file_path) or not os.path.isdir(total_file_path):
            continue
        try:
            kind = int(category.split("_")[0])
        except:
            print("select dir")
            continue
        if kind ==1:
            base=122
        elif kind ==0:
            base=127
        else:
            base = 130
        for every_patient in os.listdir(total_file_path):
            base+=1
            vice_file_path=os.path.join(total_file_path,every_patient)
            f.writelines("\n[Patient: "+every_patient+"]\n")
            print("\n[every_patient]: ",str(base)+every_patient+"\n")

            if not re.search(r'\d',vice_file_path): # not dir_type:/31729371098/
                    continue
            if not os.path.isdir(vice_file_path):
                        continue
            for every_item in os.listdir(vice_file_path):

                file_path = os.path.join(vice_file_path,every_item)

                #f.writelines("      _____MRI_date: "+every_item+"_____\n")

                try:
                    save_nii(every_item,file_path, total_save_path,every_patient, f,base)
                except:
                    traceback.print_exc(file=f)


    print("total : %d"%i)
    f.writelines("total files: %d\n"%(i))
    f.close()
    """



    