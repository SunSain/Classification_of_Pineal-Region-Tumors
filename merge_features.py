import pandas as pd
import os
import csv

OUTPUT_DIR_T1C_1="/home/chenxr/Pineal_region/after_12_08/Results/united/old_T1C/Two/pretrained_batchavg_constrain_ResNet18/"
OUTPUT_DIR_T1C_2="/home/chenxr/Pineal_region/after_12_08/Results/old_T1C/Two/pretraine_composed_SelfKL_BatchAvg_ResNet18/"
OUTPUT_DIR_T1C_3="/home/chenxr/Pineal_region/after_12_08/Results/old_T1C/Two/pretrained_constrain_composed_batchavg_ResNet18/"
        
OUTPUT_DIR_T1_1="/home/chenxr/Pineal_region/after_12_08/Results/united/old_T1/Two/pretrained_batchavg_constrain_composed_ResNet18/"
OUTPUT_DIR_T1_2="/home/chenxr/Pineal_region/after_12_08/Results/old_T1/Two/new_real_batchavg_constrain_composed_ResNet18/"
OUTPUT_DIR_T1_3="/home/chenxr/Pineal_region/after_12_08/Results/old_T1/Two/new_pretrained_selfKL_composed_ResNet18/"

OUTPUT_DIR_T2_1="/home/chenxr/Pineal_region/after_12_08/Results/united/old_T2/Two/pretrained_batchavg_constrain_ResNet18/"
OUTPUT_DIR_T2_2="/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_batchavg_constrain_composed_ResNet18/"
OUTPUT_DIR_T2_3="/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_selfKL_composed_ResNet18/"

save_root="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics"
tbox=["1","2","3"]
mode_box=["train","valid","test"]


def add_headers(csv_f,new_csv_f,sorted_f,feature_number):#feature_number: 512 or 93

    file = pd.read_csv(csv_f)

    print("csv_f: ",csv_f)
    print("new_csv_f: ",new_csv_f)
    headerList = ['sid']
    if feature_number==512:
        colums=['features_'+str(i+1) for i in range(feature_number)]
    else:
        colums=['radiomics_features_'+str(i+1) for i in range(feature_number)]
    headerList.extend(colums)
    if feature_number==512:
        headerList.append('gender')
        headerList.append('age')
    headerList.append('class')
    print("headerList: ",headerList)
    # converting data frame to csv
    #file.to_csv(new_csv_f, header=headerList, index=False)
    #new_file = pd.read_csv(new_csv_f)
    
    
    with open(new_csv_f, 'w', newline='') as out_f:
        writer = csv.writer(out_f)

        with open(csv_f, newline='') as in_f:
            reader = csv.reader(in_f)

            # Read the first row
            first_row = next(reader)
            # Count the columns in first row; equivalent to your `for i in range(len(first_row)): ...`
            header = headerList

            # Write header and first row
            writer.writerow(header)
            writer.writerow(first_row)

            # Write rest of rows
            for row in reader:
                writer.writerow(row)
    #writer.close()
    new_file = pd.read_csv(new_csv_f)
    #print("new_file: ",new_file)
    new_file=new_file.sort_values(by=["sid"],ascending=[True], inplace=False)
    print("new_file: ",new_file)
    new_file.to_csv(sorted_f, index=False)
  


def combine(A_file,B_file,C_file,fold,mode,save_root, save_file_name):
    """
    file_name = "new_"+str(fold)+"_"+mode+"_sorted.csv"
    A_file=os.path.join(A,file_name)
    B_file=os.path.join(B,file_name)
    C_file=os.path.join(C,file_name) 
    """
    print("A_file: ",A_file)
    df_a = pd.read_csv(A_file)
    df_b = pd.read_csv(B_file)
    df_c=pd.read_csv(C_file)
    merged_df = pd.merge(df_a, df_b,on="sid")
    
    ins_save_file_name= "ins_"+"t1c_t1_t2_fold_"+str(fold)+"_"+mode+".csv"
    save_file_name = "t1c_t1_t2_fold__"+str(fold)+"_"+mode+".csv"
    ins_save_file_path = os.path.join(save_root,ins_save_file_name)
    save_path=os.path.join(save_root,save_file_name)
    
    merged_df.to_csv(ins_save_file_path, index=False)
    merged_df_c = pd.merge(merged_df, df_c,on="sid")
    merged_df_c.to_csv(save_path, index=False)
 
 

def merge_radiomics(t1c_t1_t2_path, radiomics_path,save_root):
     for fold in range(3):
        for i,mode in enumerate(mode_box):
            t1c_t1_t2_file=os.path.join(t1c_t1_t2_path,"none_composed_"+str(fold)+"_"+mode+".csv")
            if not mode=='test':
                radiomics_file = os.path.join(radiomics_path,"sorted_fold_"+str(fold)+"_"+mode+".csv")
            else:
                radiomics_file = os.path.join(radiomics_path,"sorted_all_test.csv")
            print("t1c_t1_t2_file: ",t1c_t1_t2_file)
            print("radiomics_file: ",radiomics_file)
            df_a = pd.read_csv(t1c_t1_t2_file)
            df_b = pd.read_csv(radiomics_file)
            merged_df = pd.merge(df_a, df_b,on="sid")
            save_file_name="t1c_t1_t2_fold_"+str(fold)+'_'+mode+".csv"
            ins_save_file_path = os.path.join(save_root,save_file_name)
            merged_df.to_csv(ins_save_file_path, index=False)

def combine_modality_radiomics(save_root,t1c_t1_t2_path):
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    #add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/all_test.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/new_all_test.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_all_test.csv")
    #add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/all_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/new_all_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_all_train.csv")
    #add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/fold_0_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/new_fold_0_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_fold_0_train.csv")
    #add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/fold_1_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/new_fold_1_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_fold_1_train.csv")
    #add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/fold_2_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/new_fold_2_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_fold_2_train.csv")
    #add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/fold_0_valid.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/new_fold_0_valid.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_fold_0_valid.csv")
    #add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/fold_1_valid.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/new_fold_1_valid.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_fold_1_valid.csv")
    #add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/fold_2_valid.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/new_fold_2_valid.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_fold_2_valid.csv")



    #t1c_t1_t2_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/"
    radiomics_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final"

    merge_radiomics(t1c_t1_t2_path,radiomics_path,save_root)

def merge_t1_t1c_t2(save_root):
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    A=os.path.join(OUTPUT_DIR_T1C_1,"pic_result")
    B=os.path.join(OUTPUT_DIR_T1_1,"pic_result")
    C=os.path.join(OUTPUT_DIR_T2_1,"pic_result")
    save_file_name="none_composed"

    #add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/all_test.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/new_all_test.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_all_test.csv")
    #add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/all_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/new_all_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_all_train.csv")
    #add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/fold_0_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/new_fold_0_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_fold_0_train.csv")
    #add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/fold_1_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/new_fold_1_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_fold_1_train.csv")
    #add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/fold_2_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/new_fold_2_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_fold_2_train.csv")

    for fold in range(3):
        for i,mode in enumerate(mode_box):
                Apath=os.path.join(A,"test_data_2")
                Bpath=os.path.join(B,"test_data_2")
                Cpath=os.path.join(C,"test_data_2")
                
                file_name = "_"+str(fold)+"_"+mode+".csv"
                A_file=os.path.join(Apath,file_name)
                B_file=os.path.join(Bpath,file_name)
                C_file=os.path.join(Cpath,file_name) 
                
                spath=os.path.join(save_root,save_file_name)
                if not os.path.exists(spath):
                    os.mkdir(spath)
                spath = os.path.join(spath,"New")
                if not os.path.exists(spath):
                    os.mkdir(spath)
                
                Asorted_file_name = "T1C"+str(fold)+"_"+mode+"_new.csv"
                Bsorted_file_name = "T1"+str(fold)+"_"+mode+"_new.csv"
                Csorted_file_name = "T2"+str(fold)+"_"+mode+"_new.csv"
                new_A_file=os.path.join(spath,Asorted_file_name)
                new_B_file=os.path.join(spath,Bsorted_file_name)
                new_C_file=os.path.join(spath,Csorted_file_name) 
                
                spath=os.path.join(save_root,save_file_name)
                if not os.path.exists(spath):
                    os.mkdir(spath)
                spath = os.path.join(spath,"Orig")
                if not os.path.exists(spath):
                    os.mkdir(spath)
                
                Asorted_file_name = "T1C"+str(fold)+"_"+mode+"_sorted.csv"
                Bsorted_file_name = "T1"+str(fold)+"_"+mode+"_sorted.csv"
                Csorted_file_name = "T2"+str(fold)+"_"+mode+"_sorted.csv"
                sorted_A_file=os.path.join(spath,Asorted_file_name)
                sorted_B_file=os.path.join(spath,Bsorted_file_name)
                sorted_C_file=os.path.join(spath,Csorted_file_name) 

                feat_num=512
                add_headers(A_file,new_A_file,sorted_A_file,feat_num)
                add_headers(B_file,new_B_file,sorted_B_file,feat_num)
                add_headers(C_file,new_C_file,sorted_C_file,feat_num)         
                
                combine(sorted_A_file,sorted_B_file,sorted_C_file,fold,mode,save_root, save_file_name)
                
save_root1="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_batchavg_constrain" 
merge_t1_t1c_t2(save_root1)
save_root2="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics_batchavg_constrain"
combine_modality_radiomics(save_root2, save_root1)

"""
A=os.path.join(OUTPUT_DIR_T1C_2,"pic_result")
B=os.path.join(OUTPUT_DIR_T1_2,"pic_result")
C=os.path.join(OUTPUT_DIR_T2_2,"pic_result")
save_file_name="batchavg_selfkl_constrain"
for fold in range(3):
    for i,mode in enumerate(mode_box):
        combine(A,B,C,fold,mode,save_root, save_file_name)
        
A=os.path.join(OUTPUT_DIR_T1C_3,"pic_result")
B=os.path.join(OUTPUT_DIR_T1_3,"pic_result")
C=os.path.join(OUTPUT_DIR_T2_3,"pic_result")
save_file_name="mix_selfkl_or_constrain"
for fold in range(3):
    for i,mode in enumerate(mode_box):
        combine(A,B,C,fold,mode,save_root, save_file_name)
"""
