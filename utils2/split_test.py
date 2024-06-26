import sys
sys.path.append("/opt/chenxingru/Pineal_region/after_12_08/after_12_08")
from sklearn.model_selection import train_test_split
from load_data_2 import DIY_Folder
import numpy as np

def split_test(X,y):
    
    X_train, X_test, y_train, y_test= train_test_split( X, y, test_size=0.2, random_state=50, stratify=y)
    for i, (img,sid, slabel,_) in enumerate(X_train):
        print("[train]: ",i,sid)
    for i, (img,sid, slabel,_) in enumerate(X_test):
        print("[test]: ",i,sid)
        
if __name__ =="__main__":
    """
    total_data_Path="/opt/chenxingru/Pineal_region/1123_T1_axAtra_originNii_data/select_best_ax_no103/"
    data=DIY_Folder(total_data_Path,transform_dict=None)
    X,y = data.prepro_aug(data_idx = [i for i in range(len(data))],aug=False)
    X_train, X_test=split_test(X,y)
    """
    
    
    CM=[[21,0,0],
       [1,9,0],
       [2,0,8]]
    acc=np.sum(np.diag(CM)/np.sum(CM))
    print("acc: ",acc)
    