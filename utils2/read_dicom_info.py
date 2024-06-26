import pydicom
import os
from sklearn.model_selection import StratifiedKFold
import numpy as np
def read_all(path):
    print("path: ",path)
    dcm = pydicom.read_file(path)
    print("dcm[0x0008,0x0070].value: ",dcm[0x0008,0x0070].value)
    print("dcm[0x0008,0x0070]",dcm[0x0008,0x0070])
    js = "".join(dcm[0x0008,0x0070].value)
    print("js: ",js)
    return js

def count_equipts(all_library,sids):
    new_book={}
    speci={58: 'GE MEDICAL SYSTEMS', 114: 'SIEMENS', 116: 'GE MEDICAL SYSTEMS'}
    for i, sid in enumerate(sids):
        try:
            equip = all_library[sid]
        except:
            equip = speci[sid]
        nb = new_book.get(equip,0)
        nb+=1
        new_book[equip] = nb
    return new_book

# 58 114 116
def get_specific_equip():
    pathbox = ["/opt/chenxingru/Pineal_region/Pineal_MRI/058/20210312001989/7/",
            "/opt/chenxingru/Pineal_region/Pineal_MRI/114/20201104002732/8/",
           "/opt/chenxingru/Pineal_region/Pineal_MRI/116/20201030002833/8/" ]
    spe_dict = {}
    for i, path in enumerate(pathbox):
        fns = os.listdir(path)
        js=None
        print("i: ",i)
        for j, f in enumerate(fns):
            vicepath = os.path.join(path,f)
            js = read_all(vicepath)
            if js!=None:
                spe_dict[i]=js
                break

    print(spe_dict)

def read_single():
    pathlist=["/opt/chenxingru/Pineal_region/Pineal_MRI/022/20201029001975/7/14_13.dcm", #SIEMENS
              "/opt/chenxingru/Pineal_region/Pineal_MRI/056/20201010003135/8/19_18.dcm",    #GE MEDICAL SYSTEMS
              "/opt/chenxingru/Pineal_region/Pineal_MRI/058/20210312001989/6/21_20.dcm",    #GE MEDICAL SYSTEMS
              "/opt/chenxingru/Pineal_region/Pineal_MRI/114/20201104002732/9/14_13.dcm",    #SIEMENS
              "/opt/chenxingru/Pineal_region/Pineal_MRI/116/20201030002833/7/13_6.dcm"]     #GE MEDICAL SYSTEMS
    for i, vicepath in enumerate(pathlist):
        js = read_all(vicepath)
        print("js: ",js+"\n")
    assert 0==1

if __name__ =="__main__":
    #get_specific_equip()
    #read_single()
    count = {}
    path="/home/chenxr/new_tumor/"
    path="/home/chenxr/new_tumor/0_mature/"
    path= "/home/chenxr/new_tumor/1_pineal/"
    path="/home/chenxr/new_tumor/2_germinoma/"

    path = "/opt/chenxingru/Pineal_region/Pineal_MRI/"
    recordf = open("/opt/chenxingru/Pineal_region/after_12_08/after_12_08/equipment_record.txt","w")
    fn = sorted(os.listdir(path))
    two_kinds={"0":{},"1":{}}
    three_kinds = {"0":{},"1":{},"2":{}}
    sum_two=0
    sum_three=0
    all_library={}
    for i, f in enumerate(fn):
        js= None
        viceroot = os.path.join(path, f)
        
        if f.endswith(".data"):
            continue
        sid = int(f)
        if sid>122: continue
        if sid<27 or (sid>127 and sid<131): 
            two_step_box=two_kinds.get("0",{})
            three_step_box=three_kinds.get("0",{})
        elif sid<61 or (sid>122 and sid<128): 
            two_step_box=two_kinds.get("0",{})
            three_step_box=three_kinds.get("1",{})
        else:
            two_step_box=two_kinds.get("1",{})
            three_step_box=three_kinds.get("2",{})
        vicepath=viceroot
        while(os.path.isdir(vicepath)):
            vice_fn = sorted(os.listdir(vicepath))
            for i, vice_f in enumerate(vice_fn):
                
                if vice_f.endswith(".data"):
                    continue
                vicepath =  os.path.join(vicepath, vice_f)
                if not os.path.isdir(vicepath):
                    try:
                        print("sid: ",sid)
                        js = read_all(vicepath)
                        all_library[sid] = js
                        recordf.write(str(sid)+","+js+"\n")
                        two_step_num = two_step_box.get(js,0)
                        two_step_num+=1
                        three_step_num = three_step_box.get(js,0)
                        three_step_num+=1
                        two_step_box[js]=two_step_num
                        three_step_box[js]=three_step_num
                        sum_two+=1
                        sum_three+=1
                        if sid>122: continue
                        if sid<27 or (sid>127 and sid<131): 
                            two_kinds["0"]=two_step_box
                            three_kinds["0"] = three_step_box
                        elif sid<61 or (sid>122 and sid<128): 
                            two_kinds["0"]=two_step_box
                            three_kinds["1"] = three_step_box
                        else:
                            two_kinds["1"]=two_step_box
                            three_kinds["2"] = three_step_box     
                       #56,58,22,116,114
                    except:
                        print("Error: ",sid)
                        #vicepath=viceroot
                        continue
                    try:
                        num = count[js]
                        num+=1
                        count[js]=num
                    except:
                        count[js]=1
                    break
                if  os.path.isdir(vicepath) :
                    break
            if not js == None:
                break
    print("count: ",count)
    recordf.close()
    print("three_kinds: ",three_kinds)
    print("two_kinds: ",two_kinds)
    print("all_library: ",all_library)
    
    total_comman_total_file=[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]
    test_comman_total_file=[3, 4, 8, 16, 20, 25, 26, 38, 45, 47, 64, 65, 66, 75, 93, 94, 98, 99, 101, 111, 120]
    y_box = [0 if i<61 else 1 for i in total_comman_total_file] 
    k=3
    all_train_box_sid={}
    all_valid_box_sid={}
    splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(total_comman_total_file)),y_box)):
        train_sids = [total_comman_total_file[i] for i in train_idx]
        valid_sids = [total_comman_total_file[i] for i in val_idx]
        count_train = count_equipts(all_library,train_sids)
        count_valid = count_equipts(all_library,valid_sids)
        count_test = count_equipts(all_library,test_comman_total_file)
        train_box_sid={}
        valid_box_sid={}
        for i, sid in enumerate(train_sids):
            if sid<27:
               nb =  train_box_sid.get(0,0)
               nb+=1
               train_box_sid[0]=nb
            elif sid<61:
               nb =  train_box_sid.get(1,0)
               nb+=1
               train_box_sid[1]=nb
            else:
                nb =  train_box_sid.get(2,0)
                nb+=1
                train_box_sid[2]=nb
        for i, sid in enumerate(valid_sids):
            if sid<27:
               nb =  valid_box_sid.get(0,0)
               nb+=1
               valid_box_sid[0]=nb
            elif sid<61:
               nb =  valid_box_sid.get(1,0)
               nb+=1
               valid_box_sid[1]=nb
            else:
                nb =  valid_box_sid.get(2,0)
                nb+=1
                valid_box_sid[2]=nb          
        all_train_box_sid[fold]=train_box_sid
        all_valid_box_sid[fold]=valid_box_sid
         
        print("\n==== Fold: ",fold)
        print("count_train: ",count_train)
        print("count_valid: ",count_valid)
        print("count_test: ",count_test)
        print("all_train_box_sid: ",all_train_box_sid)
        print("all_valid_box_sid: ",all_valid_box_sid)
#{'SIEMENS': 11, 'GE_MEDICAL_SYSTEMS': 8, 'Philips_Medical_Systems': 5, 'Philips': 3, 'U_I_H': 1}
#{'GE_MEDICAL_SYSTEMS': 2, 'SIEMENS': 2, 'Philips': 1}
# {'SIEMENS': 2, 'GE_MEDICAL_SYSTEMS': 1}

#count:  {'GE_MEDICAL_SYSTEMS': 11, 'SIEMENS': 15, 'Philips': 4,  'Philips_Medical_Systems': 5,'U_I_H': 1}
#count:  {'GE_MEDICAL_SYSTEMS': 43, 'SIEMENS': 61, 'Philips': 10, 'Philips_Medical_Systems': 3}
  
#count:  {'GE_MEDICAL_SYSTEMS': 11, 'SIEMENS': 15, 'Philips': 4,  'Philips_Medical_Systems': 5,'UIH': 1}
#count:  {'GE_MEDICAL_SYSTEMS': 43, 'SIEMENS': 61, 'Philips': 10, 'Philips_Medical_Systems': 3}
"""
GE_MEDICAL_SYSTEMS': 43
'SIEMENS': 61
'Philips': 10
'Philips_Medical_Systems': 3

GE_MEDICAL_SYSTEMS': 11
'SIEMENS': 15
'Philips': 4
'Philips_Medical_Systems': 5
'UIH': 1
"""

              