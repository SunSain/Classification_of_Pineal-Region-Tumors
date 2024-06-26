import os
import cv2
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
def binary(cmps):
    for i in range(cmps.shape[0]):
        for j in range(cmps.shape[1]):
            for k in range(cmps.shape[2]):
                if cmps[i][j][k]<=0:
                     cmps[i][j][k] = 0
                else:
                    cmps[i][j][k] = 1
    return cmps

def array2image(array, origin_image, new_spacing=None):
    rec_image = sitk.GetImageFromArray(array)
    print("origin_image.GetDirection() : ",origin_image.GetDirection())
    #rec_image.SetDirection(origin_image.GetDirection())
    if new_spacing is not None:
        rec_image.SetSpacing(new_spacing)
    else:
        rec_image.SetSpacing(origin_image.GetSpacing())
    rec_image.SetOrigin(origin_image.GetOrigin())

    return rec_image

def build_box(root,f, saveroot):
    path =  os.path.join(root,f)
    img = sitk.ReadImage(path)
    imag_array = sitk.GetArrayFromImage(img).astype(np.uint8)
    cmps = np.array(imag_array)
    cmps = binary(cmps)
    new_mask = np.zeros(cmps.shape)

<<<<<<< HEAD
    center_x = new_mask.shape[1]/2
    center_y = new_mask.shape[2]/2
=======
    center_x = new_mask.shape[2]/2
    center_y = new_mask.shape[1]/2
    out_x_right = new_mask.shape[2]/2+ new_mask.shape[2]/4
    out_x_left = new_mask.shape[2]/4
    out_y_down = new_mask.shape[1]/2+ new_mask.shape[1]/4
    out_y_up = new_mask.shape[1]/4    
    
>>>>>>> 3a4a4f2 (20240625-code)
    print("new_mask.shape: ",new_mask.shape)
    print("center_x: ",center_x," ;center_y: ",center_y)
    
    for i in range(cmps.shape[0]):
<<<<<<< HEAD
        
=======
        if i==5:
            print("============================ it's 5!================================")# 272,212,5
        else:
            print("================")
>>>>>>> 3a4a4f2 (20240625-code)
        image = cmps[i]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
        print('num_labels: ', num_labels)
        labels[labels>0] = 255
        labels = labels.astype(np.uint8)
        #将一维灰度图像扩展到三维
        labels= np.expand_dims(labels,axis=2).repeat(3,axis=2).astype(np.uint8)
        min_distance = center_x**2+ center_y**2
        st_1=0
        st_2=0
        st_3=0
        st_0=0
        print("init_dis: ",min_distance)
        for st in stats[1:]:
<<<<<<< HEAD
=======

            if st[0]>center_y and st[0]+st[2]/2-out_y_down >0:
                continue
            if st[0]<center_y and out_y_up-(st[0]-st[2]/2) >0:
                continue
>>>>>>> 3a4a4f2 (20240625-code)
            distance = (st[1]+st[3]/2-center_x)**2+(st[0]+st[2]/2-center_y)**2 
            print("st[0], st[1]), (st[0]+st[2], st[1]+st[3]: ",st[0], st[1], st[0]+st[2], st[1]+st[3])
            print("dis: ",distance)
            if distance <= min_distance:
                st_0,st_1,st_2,st_3=st[0],st[1],st[2],st[3]
                print("st_0,st_1,st_2,st_3: ",st_0,st_1,st_2,st_3)
<<<<<<< HEAD
            cv2.rectangle(labels, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), (0, 255, 0), 3)
        new_mask[i,st_1:st_1+st_3+2,st_0:st_0+st_2+2] = 1

    new_nii = array2image(new_mask,img)
    fileName = os.path.join(saveroot,f)
    sitk.WriteImage(new_nii,fileName )
=======
            cv2.rectangle(labels,(st[1], st[0]+st[2]), ( st[1]+st[3],st[0]), (0, 255, 0), 3)
        new_mask[i,st_1:st_1+st_3,st_0:st_0+st_2] = 1
        print("st_1:st_1+st_3,st_0:st_0+st_2: ",st_1,st_1+st_3,st_0,st_0+st_2)

    new_nii = array2image(new_mask,img)
    fileName = os.path.join(saveroot,f)
    sitk.WriteImage(new_nii,fileName)
>>>>>>> 3a4a4f2 (20240625-code)
    
def combine_box(path_a,path_b, saveroot,f):
    
    img_a = sitk.ReadImage(path_a)
    imag_array_a = sitk.GetArrayFromImage(img_a).astype(np.uint8)
    cmps_a = np.array(imag_array_a)
    cmps_a = binary(cmps_a)

    img_b = sitk.ReadImage(path_b)
    imag_array_b = sitk.GetArrayFromImage(img_b).astype(np.uint8)
    cmps_b = np.array(imag_array_b)
    cmps_b = binary(cmps_b)
    assert (cmps_a.shape == cmps_b.shape)
    new_mask = np.zeros(cmps_b.shape)
    
    cmps = cmps_a+cmps_b
    cmps = np.clip(cmps,0,1)
    print("cmps_a: ",cmps_a)
    print("cmps_b: ",cmps_b)
    print("cmps: ",cmps)
        
    for i in range(cmps.shape[0]):
        image = cmps[i]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
        print('num_labels: ', num_labels)
        labels[labels>0] = 255
        labels = labels.astype(np.uint8)
        #将一维灰度图像扩展到三维
        labels= np.expand_dims(labels,axis=2).repeat(3,axis=2).astype(np.uint8)
        for st in stats[1:]:
            cv2.rectangle(labels, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), (0, 255, 0), 3)
            new_mask[i,st[1]:st[1]+st[3]+2,st[0]:st[0]+st[2]+2] = 1
<<<<<<< HEAD
    new_nii = array2image(new_mask,img_a)
    fileName = os.path.join(saveroot,f)
    sitk.WriteImage(new_nii,fileName )
=======
            
    new_nii = array2image(new_mask,img_a)
    fileName = os.path.join(saveroot,f)
    sitk.WriteImage(new_nii,fileName)
>>>>>>> 3a4a4f2 (20240625-code)
    

if __name__ == '__main__':
    #root_b = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/test_inferTs/"
    #root_a= "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/test_box/inferTs/"
    saveroot = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/test_mask2mask_boundingbox/"
    
    #root_b="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/inferTs/"
    #root_a="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/train_box/inferTs_skullStripped/"

<<<<<<< HEAD
    root="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/train_mask_boundingbox/"
    saveroot="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/train_boundingbox/"
=======
    root="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/test_boundingbox/"
    saveroot="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/test_boundingbox_2/"
>>>>>>> 3a4a4f2 (20240625-code)
    
    #root="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/test_mask_boundingbox/"
    #saveroot="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/test_boundingbox/"
    
    paths={} #{sid:{"path_a":, "path_b":}}
    
    
    if not os.path.exists(saveroot):
        os.mkdir(saveroot)
    sub_fns_a = sorted(os.listdir(root))
    
    for i, f in enumerate(sub_fns_a):
        #build_box(root, f, saveroot)
        fn = f.split("_")  
        try:
                    sid=int(fn[0])
        except:
                print("it's the_select_file")
                continue
<<<<<<< HEAD
=======
        
>>>>>>> 3a4a4f2 (20240625-code)
        path_a = os.path.join(root,f)
        path_pair={}
        path_pair["path_a"]= path_a
        paths[sid]=path_pair
        build_box(root,f, saveroot)

    """
    sub_fns_b = sorted(os.listdir(root_b))
    
    for i, f in enumerate(sub_fns_b):
        #build_box(root, f, saveroot)
        fn = f.split("_")  
        try:
                    sid=int(fn[0])
        except:
                print("it's the_select_file")
                continue
        path_b = os.path.join(root_b,f)
        path_pair=paths[sid]
        path_pair["path_b"]= path_b
        paths[sid]=path_pair
    
    
    for i, f in enumerate(sub_fns_a):
        #build_box(root, f, saveroot)
        fn = f.split("_")  
        try:
                    sid=int(fn[0])
        except:
                print("it's the_select_file")
                continue
        path_pair=paths[sid]
        path_a =path_pair["path_a"]
        path_b =path_pair["path_b"]
        combine_box(path_a, path_b, saveroot, f)
        """