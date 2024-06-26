import os
import numpy as np

def get_sid_box(file_path):
    sid_box=[]
    count=0


    sub_fns = sorted(os.listdir(file_path))


    for i, f in enumerate(sub_fns):
            #if i>63:break
            fn = f.split("_")
            fn = fn[0].split(".")
            #get label: 0<sid<61: label =0 | 61<sid: label=1
            try:
                    sid=int(fn[0])

            except:
                print("it's the_select_file")
                continue
            count+=1
                    
            # no more new data
            if sid>122:
                continue
            sid_box.append(sid)
            # no more new data
    return sid_box, count

def check_sids(T1C_path,T1_path,T2_path):
    T1C_box,count_t1c=get_sid_box(T1C_path)
    T1_box , count_t1=get_sid_box(T1_path)
    T2_box, count_t2=get_sid_box(T2_path)
    print("T1C: ",count_t1c, T1C_box)
    print("T1 : ",count_t1, T1_box)
    print("T2: ",count_t2, T2_box)   
    
    t1c_t1=set(T1C_box).intersection(set(T1_box))
    t1c_t2=set(T1C_box).intersection(set(T2_box))
    t2_t1=set(T1C_box).intersection(set(T2_box))
    t1c_t2_t1=set(t1c_t2).intersection(set(T1_box))
    print("t1c_t1: ",len(t1c_t1)," ", t1c_t1)
    print("t1c_t2: ",len(t1c_t2)," ",t1c_t2)
    print("t2_t1 : ",len(t2_t1)," ",t2_t1)
    print("t1c_t2_t1: ",len(t1c_t2_t1)," ",t1c_t2_t1)
    
    return t1c_t2_t1
if __name__ =="__main__":
    T1C_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C"
    T1_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1"
    T2_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T2"
    #check_sids(T1C_path=T1C_path,T1_path=T1_path,T2_path=T2_path)
    test_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T2"
    print(get_sid_box(test_path))