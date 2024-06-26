import os
import re
import numpy as np
import shutil
import torch

torch.cuda.empty_cache()
print(torch.cuda.is_available())

def move_extra_test():
    total_t1c_root="/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C/"
    total_t1_root="/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1/"
    total_t2_root="/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T2/"
    
    test_t1c_root="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C/"
    test_t1_root="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1/"
    test_t2_root="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T2/"
    
    subfns = sorted(os.listdir(total_t1c_root))
    for file in subfns:
        sid = int(file.split("_")[0])
        if sid<123:
            continue
        src_path = os.path.join(total_t1c_root,file)
        des_path = os.path.join(test_t1c_root,file)
        shutil.copy(src_path,des_path)
        
    subfns = sorted(os.listdir(total_t1_root))
    for file in subfns:
        sid = int(file.split("_")[0])
        if sid<123:
            continue
        src_path = os.path.join(total_t1_root,file)
        des_path = os.path.join(test_t1_root,file)
        shutil.copy(src_path,des_path)

    subfns = sorted(os.listdir(total_t2_root))
    for file in subfns:
        sid = int(file.split("_")[0])
        if sid<123:
            continue
        src_path = os.path.join(total_t2_root,file)
        des_path = os.path.join(test_t2_root,file)
        shutil.copy(src_path,des_path)

def counting_num(root):
    subfns = sorted(os.listdir(root))
    count=0
    numset=set()
    for fn in subfns:
        sid = int(fn.split("_")[0])
        if sid>122: continue
        count+=1
        numset.add(sid)
    print("counting_result: ",count,len(numset),root)

def counting_gender(txtfile):
    gender_0=[] #male
    gender_1=[]
    ages={}

    with open(txtfile) as f:
        lines = f.readlines()
        i=0
        for line in lines:
            if i==0: 
                i+=1
                continue
            else: i+=1
            reg = r"[\t,' ',\n]+"
            row = re.split(reg,line)
            print("row: ",row)
            if row==None or row[0]=='': continue
            print("row2: ",row)
            try:
                sid = int(row[0])
            except:
                print("it's not valid row: ",row)
                continue
            # no more new data
            # no more new data
            
            gender = row[3]
            age = int(re.findall(r'\d+', row[4])[0])
            unit = re.findall(r'[\u4e00-\u9fa5]', row[4])[0]
            print("gender: ",gender)
            if gender=='男':
                gender=0.
                gender_0.append(sid)
            else:
                gender=1.
                gender_1.append(sid)
            if unit == '月':
                age = age/12.0
            print("age: ",age)
            ages[sid]=age
    print("\n男:女: ",len(gender_0),':',len(gender_1))
    print("\nsid<=122 男:女: ",len([a for a in gender_0 if a<=122 ]),':',len([a for a in gender_1 if a<=122 ]))   
    print("\nbinary_label_0 男:女: ",len([a for a in gender_0 if a<61 ]),':',len([a for a in gender_1 if a<61 ]))   
    print("\nbinary_label_1 男:女: ",len([a for a in gender_0 if a<=122 and a>60 ]),':',len([a for a in gender_1 if a<=122 and a>60]))   
    print("\nAge:")
    #print("label_0 avg age: ",np.mean(np.array([ages[a] for a in ages.keys() if a<61])), max(min(ages[a]  for a in ages.keys() if a<61)))
    mean_age_label_0=np.mean(np.array([ages[a] for a in ages.keys() if a<61]))
    min_age_label_0=min(ages[a]  for a in ages.keys() if a<61)
    max_age_label_0=max(ages[a]  for a in ages.keys() if a<61)
    print("label_0 avg age: ",mean_age_label_0)
    print("label_0 +- age: ",mean_age_label_0-min_age_label_0,max_age_label_0-mean_age_label_0,max(mean_age_label_0-min_age_label_0,max_age_label_0-mean_age_label_0))
    
    
    
    mean_age_label_1=np.mean(np.array([ages[a] for a in ages.keys() if a<=122 and a>60]))
    min_age_label_1=min(ages[a]  for a in ages.keys() if a<=122 and a>60)
    max_age_label_1=max(ages[a]  for a in ages.keys() if a<=122 and a>60)
    print("label_1 avg age: ",mean_age_label_1)
    print("label_1 +- age: ",mean_age_label_1-min_age_label_1,max_age_label_1-mean_age_label_1,max(mean_age_label_1-min_age_label_1,max_age_label_1-mean_age_label_1))

    print("Total<123 avg age: ",np.mean(np.array([ages[a] for a in ages.keys() if a<=122])))
    mean_age_label_1=np.mean(np.array([ages[a] for a in ages.keys() if a<=122]))
    min_age_label_1=min(ages[a]  for a in ages.keys() if a<=122)
    max_age_label_1=max(ages[a]  for a in ages.keys() if a<=122)
    print("total avg age: ",mean_age_label_1)
    print("total +- age: ",mean_age_label_1-min_age_label_1,max_age_label_1-mean_age_label_1,max(mean_age_label_1-min_age_label_1,max_age_label_1-mean_age_label_1))

    #print("男sid: ",gender_0)
    #print("女sid: ",gender_1)

def analyze_txt(root):
    txtfile = os.path.join(root,"model_result/ResNet18_BatchAvg_0401_T1__4_record.txt")
    try: 
        f= open(txtfile,"r") 
    except:
        try:
            txtfile=os.path.join(root,'model_result/ResNet18_SingleBatchAvg_0401_T1__4_record.txt')
            f= open(txtfile,"r") 
        except:
            try:
                txtfile=os.path.join(root,'model_result/ResNet18_CE_0401_T1__4_record.txt')
                
                f= open(txtfile,"r") 
            except:
                try:
                    txtfile=os.path.join(root,'model_result/ResNet18_SelfKL_0401_T1__4_record.txt')
                    f= open(txtfile,"r")
                except:
                    try:
                        txtfile=os.path.join(root,'model_result/ResNet10_CE_0401_T1__4_record.txt')
                        f= open(txtfile,"r")
                    except:
                        try:
                            txtfile=os.path.join(root,'model_result/ResNet34_CE_0401_T1__4_record.txt')
                            f= open(txtfile,"r")
                        except:
                            try:
                                txtfile=os.path.join(root,'model_result/ResNet18_BatchAvg_0401_T1__4_record.txt')
                                f= open(txtfile,"r")
                            except:
                                try:
                    
                                    txtfile=os.path.join(root,'model_result/ResNet18_SingleBatchAvg_0401_T1__4_record.txt')
                                    f= open(txtfile,"r")
                                except:
                                    txtfile=os.path.join(root,'model_result/ResNet18_SingleBatchAvg_selfKL_0401_T1__4_record.txt')
                                    f= open(txtfile,"r")

    lines = f.readlines()
    train_static={}
    valid_static={}
    test_static={}
    for line in lines:
        reg = r"['<==','==>']+"
        linebox = re.split(reg,line)
        if "Train<== acc records" in line:
            #print(linebox)
            #print("\n=======\n")
            for i,item in enumerate(linebox):
                if 'avg:' in item:
                    #acc records , avg: 1.000000 
                    marker=linebox[i-1].split('records')[0]
                    score = float(item.split("avg:")[1].strip())
                    #print("Train marker: ",marker,score)
                    train_static[marker]=score
                    
        elif "vALID<== acc records" in line:
            #print(linebox)
            #print("\n=======\n")
            for i,item in enumerate(linebox):
                if 'avg:' in item:
                    #acc records , avg: 1.000000 
                    marker=linebox[i-1].split('records')[0]
                    score = float(item.split("avg:")[1].strip())
                    #print("Train marker: ",marker,score)
                    valid_static[marker]=score
                    
        elif "TEST<== acc records" in line:
            #print(linebox)
            #print("\n=======\n")
            for i,item in enumerate(linebox):
                if 'avg:' in item:
                    #acc records , avg: 1.000000 
                    marker=linebox[i-1].split('records')[0]
                    score = float(item.split("avg:")[1].strip())
                    #print("Train marker: ",marker,score)
                    test_static[marker]=score
        else:continue         
    print("Train: ",train_static)
    print("Valid: ",valid_static)
    print("TEST: ",test_static)

if __name__=="__main__":
    """
    t1c_train_root="/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C" # 95
    counting_num(t1c_train_root)
    t1c_test_root='/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C' #23
    counting_num(t1c_test_root)
    t1c_total_root="/home/chenxr/new_nii_data/all_T1C/"
    counting_num(t1c_total_root)

    t1c_train_root="/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1" # 95
    counting_num(t1c_train_root)
    t1c_test_root='/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1' #23
    counting_num(t1c_test_root)

    t1c_train_root="/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T2" # 95
    counting_num(t1c_train_root)
    t1c_test_root='/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T2' #23
    counting_num(t1c_test_root)
    """
    #pineal_txt="/home/chenxr/pineal_data_statistic_final/pineal_csv.txt"
    #counting_gender(pineal_txt)
    
    t1c_record_root="/home/chenxr/Pineal_region/after_12_08/Results/old_T1C/Two/pretrained_constrain_composed_batchavg_ResNet18/"
    analyze_txt(t1c_record_root)
    t2_root="/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_batchavg_constrain_composed_ResNet18/"
    analyze_txt(t2_root)
    t1_root="/home/chenxr/Pineal_region/after_12_08/Results/old_T1/Two/new_real_batchavg_constrain_composed_ResNet18"
    analyze_txt(t1_root)
    
    print("three_modality_best_attention_SingelBatchavg")
    total_best_root='/home/chenxr/Pineal_region/after_12_08/Results/New_attention_3_cnn+attention/Two/three_modality_best_attention_SingelBatchavg/'
    analyze_txt(total_best_root)
    
    print("new_arch_attention_SingelBatchavg")
    new_arch_attention="/home/chenxr/Pineal_region/after_12_08/Results/new_arch_New_attention_3_cnn+attention/Two/three_modality_best_attention_SingelBatchavg/"
    analyze_txt(new_arch_attention)
    
    print("SelfAttention_concatenation_mlp_BatchAvg")
    SelfAttention_concatenation_mlp_root="/home/chenxr/Pineal_region/after_12_08/Results/SelfAttention_concatenation_mlp/Two/three_modality_BatchAvg/"
    analyze_txt(SelfAttention_concatenation_mlp_root)

    print("New_concatenation_mlp_selfKL")
    New_concatenation_mlp_selfKL_root="/home/chenxr/Pineal_region/after_12_08/Results/New_concatenation_mlp/Two/three_modality_SelfKL/"
    analyze_txt(New_concatenation_mlp_selfKL_root)
    
    print("New_concatenation_mlp_BatchAvg")
    New_concatenation_mlp_BatchAvg_root="/home/chenxr/Pineal_region/after_12_08/Results/New_concatenation_mlp/Two/three_modality_BatchAvg/"
    analyze_txt(New_concatenation_mlp_BatchAvg_root)

    print("New_concatenation_mlp_CE")
    New_concatenation_mlp_CE_root="/home/chenxr/Pineal_region/after_12_08/Results/New_concatenation_mlp/Two/three_modality_CE/"
    analyze_txt(New_concatenation_mlp_CE_root)
    
    print("Res_CrossAttention_mlp")
    root="/home/chenxr/Pineal_region/after_12_08/Results/Res_CrossAttention_mlp/Two/three_modality_Batchavg/"
    analyze_txt(root)
    print("SelfAttention_CrossAttention_mlp")
    root="/home/chenxr/Pineal_region/after_12_08/Results/SelfAttention_CrossAttention_mlp/Two/three_modality_Batchavg/"
    analyze_txt(root)
    print("SelfAttention_concatenation_mlp_singleBatchavg")
    root="/home/chenxr/Pineal_region/after_12_08/Results/SelfAttention_concatenation_mlp/Two/three_modality_SingleBatchAvg/"
    analyze_txt(root)
    
    print("SelfAttention_concatenation_mlp_CE")
    root="/home/chenxr/Pineal_region/after_12_08/Results/SelfAttention_concatenation_mlp/Two/three_modality_CE/"
    analyze_txt(root)
    print("t1c")
    t1c="/home/chenxr/Pineal_region/after_12_08/Results/old_T1C/Two/pretrained_constrain_composed_batchavg_ResNet18/"
    analyze_txt(t1c)
    print("t2")
    t2="/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_batchavg_constrain_composed_ResNet18/"
    analyze_txt(t2)
    print("t1")
    t1="/home/chenxr/Pineal_region/after_12_08/Results/old_T1/Two/new_real_batchavg_constrain_composed_ResNet18/"
    analyze_txt(t1)
    
    print("clinical_1layer")
    clinical_1layer="/home/chenxr/Pineal_region/after_12_08/Results/Clinical_pure_binaryRegression/Two/three_modality/"
    analyze_txt(clinical_1layer)
    print("clinical_3layer")
    clinical_3layer="/home/chenxr/Pineal_region/after_12_08/Results/Clinical_pure_binaryRegression_3Layer/Two/three_modality/"
    analyze_txt(clinical_3layer)
    print("clinical_7layer")
    clinical_7layer="/home/chenxr/Pineal_region/after_12_08/Results/Clinical_pure_binaryRegression_7Layer/Two/three_modality/"
    analyze_txt(clinical_7layer)

    print("\n=======================================================================================\n")
    print("pure_img: t1c_ce")
    t1c_ce="/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_composed_CE/"
    analyze_txt(t1c_ce)

    print("pure_img: t1c_mask_ce")
    t1c_mask_ce="/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_mask1_composed_RM_CE/"
    analyze_txt(t1c_mask_ce)
    
    print("pure_img: t1c_mask_singleBatchavg")
    t1c_mask_singleBatchavg="/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_mask1_composed_RM_SingleBatchAvg/"
    analyze_txt(t1c_mask_singleBatchavg)
    
    print("pure_img: t1c_singlebatchavg")
    t1c_singlebatchavg="/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_SingleBatchAvg_composed_ResNet18/"
    analyze_txt(t1c_singlebatchavg)

    print("pure_img: t2_mask_ce")
    t2_mask_ce="/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_mask1_composed_RM_CE/"
    analyze_txt(t2_mask_ce)
    print("pure_img: t2_sing_matrix")
    t2_sing_matrix="/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_mask1_composed_RM_SingleBatchAvg_selfKL/"
    analyze_txt(t2_sing_matrix)
    
    print("pure_img: t1_composed")
    t1_composed="/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_composed_CE/"
    analyze_txt(t1_composed)
    
    print("pure_img: t1_mask1_ce")
    t1_mask1_ce="/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_mask1_composed_RM_CE/"
    analyze_txt(t1_mask1_ce)
    
    print("pure_img: t1_matrix")
    t1_matrix="/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_selfKL_composed_ResNet18/"
    analyze_txt(t1_matrix)
    
    print("\n.............................................................................................................\n")

    print("pure_img: selfattention_CE")
    selfattention_CE="/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/SelfAttention_concatenation_mlp/Two/three_modality_composed_CE/"
    analyze_txt(selfattention_CE)

    print("pure_img: selfattention_SingleBatchavg")
    selfattention_SingleBatchavg="/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/SelfAttention_concatenation_mlp/Two/three_modality_composed_SingleBatchavg/"
    analyze_txt(selfattention_SingleBatchavg)
    
    print("pure_img: selfattention_mask_ce")
    selfattention_mask_ce="/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/SelfAttention_concatenation_mlp/Two/three_modality_mask1_composed_RM_CE"
    analyze_txt(selfattention_mask_ce)
    
    print("pure_img: selfattention_mask_singlebatchavg")
    selfattention_mask_singlebatchavg="/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/SelfAttention_concatenation_mlp/Two/three_modality_mask1_composed_RM_SingleBatchavg"
    analyze_txt(selfattention_mask_singlebatchavg)


    "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/SelfAttention_concatenation_mlp/Two/three_modality_composed_CE/"
    
    augmentation_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_ce/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_Flip_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_Affine_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_composed_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_mask1_composed_RM_CE/",
        
        
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_ce/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_Flip_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_Affine_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_composed_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_mask1_composed_RM_CE/",
        
        
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_ce/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_Flip_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_Affine_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_composed_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_mask1_composed_RM_CE/"
    ]
    t1c_clinical_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/use_clinical/Two/T1C_age_ce/",
        "/home/chenxr/Pineal_region/after_12_08/Results/use_clinical/Two/T1C_gender_ce/",
        "/home/chenxr/Pineal_region/after_12_08/Results/use_clinical/Two/T1C_clinical_ce/",
    ]
    
    box=t1c_clinical_box
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)
    
    t1c_singlebatchavg_clinical_composedRM_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/Use_Clinical/Two/T1C_composed_RM/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_19",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_55",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_91",  
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_37" , 
    ]
    box=t1c_singlebatchavg_clinical_composedRM_box
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)


    
    t1c_batchavg_clinical_composedRM_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/Use_Clinical/Two/T1C_composed_RM/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_mask1_composed_RM_Batchavg_ce_91/", 
    ]
    box=t1c_batchavg_clinical_composedRM_box
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)
        
        
    randommaask=[
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_mask1_ce/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_mask1_ce/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_mask1_ce/",
    ]
    box=randommaask
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)
    batchavg_box=[
        
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_mask1_composed_RM_Batchavg_ce_91/",
        
    ]
    box=batchavg_box
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)
    t2_singlebatchavg_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_composed_ce/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Single_batchavg/Two/T2_singlebatchavg_composed_ResNet18/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_mask1_composed_RM_Singlebatchavg_37/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_singlebatchavg_ce",
        "/home/chenxr/Pineal_region/after_12_08/Results/weightd_clinical/Two/T2_mask1_composed_RM_singlebatchavg_ce_64/",
        
    ]
    box=t2_singlebatchavg_box
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)

    t1_singlebatchavg_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_ce/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Single_batchavg/Two/T1_singlebatchavg_composed_ResNet18/", 
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_mask1_composed_RM_Singlebatchavg_37/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weightd_clinical/Two/T1_mask1_composed_RM_singlebatchavg_ce_91/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weightd_clinical/Two/T1_mask1_composed_RM_singlebatchavg_ce_19/"
    ]
    box=t1_singlebatchavg_box
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)
        
    resnetasyer=[
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRIz/Two/T1C_resnet10_ce/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_ce/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRIz/Two/T1C_resnet34_ce/", 
    ]
    box=resnetasyer
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)

    t1_batchavg_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_RM_2/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_ce/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Single_batchavg/Two/T1_pretrained_batchavg_constrain_composed_ResNet18/", 
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T1/Two/new_real_batchavg_constrain_composed_ResNet18/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weightd_clinical/Two/T1_mask1_composed_RM_batchavg_ce_91/",
        
    ]
    box=t1_batchavg_box
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)

    T2_Batchavg_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_batchavg_constrain_composed_ResNet18/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical_T2/Two/T2_composed_RM_batchavg_91/",
    ]
    box=T2_Batchavg_box
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)

    T2_SelfKL_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_selfKL_composed_ResNet18/",
        
    ]
        
    "/home/chenxr/Pineal_region/after_12_08/Results/New_attention_3_cnn+attention/Two/three_modality_best_attention_selfKL",
    "/home/chenxr/Pineal_region/after_12_08/Results/New_attention_3_cnn+attention/Two/three_modality_best_attention_SingelBatchavg",
    "/home/chenxr/Pineal_region/after_12_08/Results/New_attention_3_cnn+attention/Two/three_modality_best_attention_RM/model_result",
        
        
    box=T2_SelfKL_box
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)
        
        
    attention_weighted=[
        "/home/chenxr/Pineal_region/after_12_08/Results/test_mixattention/Two/self_crossatention_19_three_modality_best_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/test_mixattention/Two/self_crossatention_55_three_modality_best_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/test_mixattention/Two/crossatention_three_modality_best_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/test_mixattention/Two/new_three_modality_best_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/test_mixattention/Two/self_crossatention_91_three_modality_best_CE/"

    ]
    box=attention_weighted
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)
    
    
    simple_mlp=[
        "/home/chenxr/Pineal_region/after_12_08/Results/New_concatenation_mlp/Two/three_modality_BatchAvg/",
        "/home/chenxr/Pineal_region/after_12_08/Results/New_concatenation_mlp/Two/three_modality_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/New_concatenation_mlp/Two/three_modality_SelfKL/",
        
    ]
    box=simple_mlp
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)

    selfkl=[
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfKL/Two/T1_mask1_composed_RM_1090/",
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfKL/Two/T2_mask1_composed_RM_1090/",
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_selfKL_composed_ResNet18/",
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfKL/Two/T1C_mask1_composed_RM_1090/",
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfKL/Two/T1C_pretrained_mask1_composed_RM_1090/",
        "/home/chenxr/Pineal_region/after_12_08/Results/concatenation/Two/Three_modality/",
        "/home/chenxr/Pineal_region/after_12_08/Results/New_concatenation_mlp/Two/three_modality_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfAttention_concatenation_mlp/Two/three_modality_BatchAvg/",
        "/home/chenxr/Pineal_region/after_12_08/Results/New_attention_3_cnn+attention/Two/three_modality_best_attention/",
        "/home/chenxr/Pineal_region/after_12_08/Results/New_attention_3_cnn+attention/Two/three_modality_best_attention_selfKL/",
        "/home/chenxr/Pineal_region/after_12_08/Results/New_attention_3_cnn+attention/Two/three_modality_best_attention_SingelBatchavg/",
        "/home/chenxr/Pineal_region/after_12_08/Results/New_attention_3_cnn+attention/Two/three_modality_best_attention_RM",
        "/home/chenxr/Pineal_region/after_12_08/Results/New_attention_3_cnn+attention/Two/three_modality_best_attention_SingelBatchavg_RM",
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfAttention_MLP_Transformer/Two/CE_new/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_RM_2/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_RM_Matrix_MGDA/",
    ]

    box=selfkl
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)

    selfatten_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfAttention_concatenation_mlp/Two/three_modality_SingleBatchAvg_two_stage/",
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfAttention_concatenation_mlp/Two/three_modality_SingleBatchAvg",
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfAttention_concatenation_mlp/Two/three_modality_SelfKL",
        
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfAttention_concatenation_mlp/Two/three_modality_CE",
        
        
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfAttention_concatenation_mlp/Two/three_modality_BatchAvg",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_111_CE/",
    ]

    box=selfatten_box
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)
        
        
    all_loss_box=[
         "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_37",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_13/",
    ]
    box=all_loss_box
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)
        
    t1_singlebatchavg_new_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical_T1/Two/T1_composed_RM_SingleBatchavg_37/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical_T1/Two/T1_composed_RM_SingleBatchavg_19/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical_T1/Two/T1_composed_RM_SingleBatchavg_11/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical_T1/Two/T1_composed_RM_SingleBatchavg_91/",
        
    ]
    box=t1_singlebatchavg_new_box
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)

    t1c_matrix_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/Use_Clinical/Two/T1C_composed_RM/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Matrix_MGDA/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Matrix_MGDA_2/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Matrix_MGDA_3/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Matrix_MGDA_4/",
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfKL/Two/T1C_pretrained_mask1_composed_RM_1090/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_SelfKL_7/",
    ]
    box=t1c_matrix_box
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)
        
    mixattention=[
                "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_13/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_433_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_555_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_333_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_433_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_811_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_122_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_100_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_622_CE/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_111_CE/",
    ]
    box=mixattention
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)
    t1c_sing_batch=[
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_2",
        
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_3",
        
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_4",


        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_6",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_7",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_8",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_9",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_10",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_11",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_12",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_13",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_14",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_15",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_16",
    
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_17",
        
    ]
    box=t1c_sing_batch
    for i, path in enumerate(box):
        print("path : ",path)
        analyze_txt(path)
    #move_extra_test()