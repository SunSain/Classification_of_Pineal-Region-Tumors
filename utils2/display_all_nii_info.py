#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:10:59 2021

@author: chenxingru
"""
from nibabel.viewers import OrthoSlicer3D as osd
import nibabel as nib
import matplotlib.pyplot as plt
import pydicom
import os
import SimpleITK as sitk

import sys
print("sys_path: ",sys.path)


def read_nii_file(file_path):
    img=nib.load(file_path)
    w,h,q = img.dataobj.shape
    osd(img.dataobj).show()
    
def read_dicom_file(file_path):
<<<<<<< HEAD
    dcm = pydicom.read_file(file_path)
    plt.imshow(dcm.pixel_array)
=======
    #dcm = pydicom.read_file(file_path)
    ds = pydicom.dcmread(file_path)
 
    # 尝试获取采集矩阵信息
    try:
        pixel_spacing = ds.PixelSpacing  # 像素间隔
        slice_thickness = ds.SliceThickness  # 切片厚度
        rows = ds.Rows  # 行数
        columns = ds.Columns  # 列数
        print(f"PixelSpacing: {pixel_spacing}")
        print(f"SliceThickness: {slice_thickness}")
        print(f"Rows: {rows}")
        print(f"Columns: {columns}")
        width = ds.Rows
        height = ds.Columns
        
        # 获取FOV的物理尺寸（以毫米为单位）
        width_mm = ds.PixelSpacing[0] * (width - 1)
        height_mm = ds.PixelSpacing[1] * (height - 1)
        
        # 如果有ICS_Windth和ICS_Height标签，则可以直接获取
        try:
            width_mm_ics = ds.ImageOrientationPatient[0] * width * ds.PixelSpacing[0]
            height_mm_ics = ds.ImageOrientationPatient[1] * height * ds.PixelSpacing[1]
        except AttributeError:
            print("ICS tags not found.")
        
        print(f"FOV Width (pixels): {width}")
        print(f"FOV Height (pixels): {height}")
        print(f"FOV Width (mm): {width_mm:.2f}")
        print(f"FOV Height (mm): {height_mm:.2f}")
        circle_data = ds.pixel_array
        print(f"circle_data: {circle_data}")
        print(f"Manufacturer: {ds.Manufacturer}")
        print(f"Model Name: {ds.ManufacturerModelName}")
    except AttributeError as e:
        print(f"The DICOM file does not contain the information you are looking for: {e}")
    #plt.imshow(dcm.pixel_array)
>>>>>>> 3a4a4f2 (20240625-code)

def get_img_info(vice_path,f):
    img=sitk.ReadImage(vice_path)
    d, s, space = img.GetDepth(),img.GetSize(),img.GetSpacing()
    print(img.GetSize())
    print(img.GetSpacing())
    words = '\t'.join((vice_path,str(img.GetDepth()),str(img.GetSize()),str(img.GetSpacing()),'\n'))
    f.writelines(words)
    return d,s,space

def get_single_file_info(filepath):
    img=sitk.ReadImage(filepath)
    print(img.GetDepth())
    print(img.GetSize())
    print(img.GetSpacing())

if __name__ == '__main__':
    
<<<<<<< HEAD
=======
    file_path="/opt/chenxingru/Pineal_region/Pineal_MRI/114/20201104002732/6/7_6.dcm"
    read_dicom_file(file_path)
>>>>>>> 3a4a4f2 (20240625-code)
    
    """
    filepath="/opt/chenxingru/Pineal_region/0509_fresh_all_data/select_every_kind/T1W/001/1_C0206C92D6804D2188CA5E93CAD3B83_sag_t1_flair_Turn.nii.gz"
    filepath ="/opt/chenxingru/opt/Data/new4/select_best_sag/081_5_1D52/081_5_1D52_081_5_1D52ED6F5B41444986DC29F0FAC697AF_oax_t1_flair_Turn.nii.gz-lin.nii.gz"
    filepath="/opt/chenxingru/opt/Data/new4/select_best_sag/081_5_1D52/081_5_1D52_081_5_1D52ED6F5B41444986DC29F0FAC697AF_oax_t1_flair_Turn.nii.gz"
    
    skullstriped_t1c_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/imagesTs_no78_9/121_6_0000.nii.gz"
    original_t1c_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/masked_bounding/121_6.nii.gz"
    masked_t1c_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/inferTs_no78_9/121_6.nii.gz"
    boundingbox_t1c_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/masked_bounding/121_6.nii.gz"
    get_single_file_info(original_t1c_path)
    get_single_file_info(skullstriped_t1c_path)
    get_single_file_info(masked_t1c_path)
    get_single_file_info(boundingbox_t1c_path)
<<<<<<< HEAD
    """
=======

>>>>>>> 3a4a4f2 (20240625-code)

    print("================================================================================================")

    statics = {}
    #txtName="/opt/chenxingru/Pineal_region/T1W_registrated_SAG_best_file/"+"Non_registered_best_T1W_Sag_nii_Info.txt"
    txtName="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/trained_result/nnUNet/3d_fullres/Task009_myT1c/nnUNetTrainerV2__nnUNetPlansv2.1/gt_niftis/check_nii_infor.txt"
 
    f = open(txtName, "a+", encoding='UTF-8')
    f.writelines('\t'.join(('Name','Depth','Size','Spacing','\n')))

    i=0
    #total_path="/opt/chenxingru/Pineal_region/0509_fresh_all_data/select_every_kind/T1W_detailed_select/SAG/selected_best_sag/"
    total_path="/opt/chenxingru/Pineal_region/0509_fresh_all_data/select_every_k/opt/chenxingru/Pineal_region/10_14_T2_regis_AXandTra_best_f_Test/ind/T2_detailed_select_0528/AX_and_TRA/"
    total_path="/opt/chenxingru/Pineal_region/0509_fresh_all_data/select_every_kind/T1W_detailed_select_0528/AX_and_TRA/"
    
    total_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/trained_result/nnUNet/3d_fullres/Task009_myT1c/nnUNetTrainerV2__nnUNetPlansv2.1/gt_niftis/"
    total_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/trained_result/nnUNet/3d_fullres/Task009_myT1c/nnUNetTrainerV2__nnUNetPlansv2.1/cv_niftis_raw/"
    total_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_cropped_data/Task009_myT1c/gt_segmentations/"
    total_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/labelsTr/"
    total_file = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/imagesTr/"
    total_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/trained_result/nnUNet/3d_fullres/Task009_myT1c/nnUNetTrainerV2__nnUNetPlansv2.1/gt_niftis/"
    total_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/trained_result/nnUNet/3d_fullres/Task009_myT1c/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/"
    total_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_processed/Task009_myT1c/gt_segmentations/"
    total_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_cropped_data/Task009_myT1c/gt_segmentations/"
    total_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/trained_result/nnUNet/3d_fullres/Task010_BTumour_T1C/nnUNetTrainerV2__nnUNetPlansv2.1/gt_niftis/"
    total_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/trained_result/nnUNet/3d_fullres/Task010_BTumour_T1C/nnUNetTrainerV2__nnUNetPlansv2.1/fold_2/validation_raw/"
    total_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task010_BTumour_T1C/imagesTr/"
<<<<<<< HEAD
    for every_file in os.listdir(total_path):
        i+=1
=======
    total_path = "/home/chenxr/new_NII/T1C_ax_tra/"
    total_path="/opt/chenxingru/Pineal_region/0509_fresh_all_data/select_every_kind/T2_detailed_select_0528/AX_and_TRA/"
    total_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T2/"
    
    #96: {'640_640_24': {'num': 22}, '512_512_24': {'num': 10}, '324_384_24': {'num': 25}, '512_512_23': {'num': 35}, '512_640_24': {'num': 1}, '384_384_24': {'num': 1}, '448_448_24': {'num': 1}, '640_640_23': {'num': 1}}
    #23:  {'512_512_23': {'num': 11}, '512_512_24': {'num': 1}, '324_384_24': {'num': 4}, '640_640_24': {'num': 7}}
    for every_file in os.listdir(total_path):
        
>>>>>>> 3a4a4f2 (20240625-code)
        j=0
        
        vice_path=os.path.join(total_path,every_file)

        print("vice_path: ",vice_path)
        if os.path.isdir(vice_path):
            continue
        if not every_file.endswith(".nii.gz"):
            continue
        MRI_path=vice_path
<<<<<<< HEAD
=======
        sid = int(every_file.split("_")[0])
        print("sid: ",sid)
        if sid>122:
            continue
        i+=1
>>>>>>> 3a4a4f2 (20240625-code)
        #read_nii_file(MRI_path)
        print(every_file)
        depth, size, space = get_img_info(vice_path,f)
        size_string = "_".join(map(str,size))
<<<<<<< HEAD
        print("tostring: ",size_string)
        print("statics.keys(): ",statics.keys())
        if  not size_string in statics.keys():
            statics[size_string] = 1
        else:
            statics[size_string] =statics[size_string]+1
=======
        if  not size_string in statics.keys():
            statics[size_string]={}
            statics[size_string]['num']= 1
            box =[every_file]
            #statics[size_string]['name']=box
        else:
            statics[size_string]['num'] =statics[size_string]['num']+1
            #box = statics[size_string]['name']
            #box.append(every_file)
            #statics[size_string]['name']=box
>>>>>>> 3a4a4f2 (20240625-code)

    print("total: ",i)
    print("statics: ",statics)
    f.close()
<<<<<<< HEAD
    
=======
    """
>>>>>>> 3a4a4f2 (20240625-code)
    """
    file_path='/home/chenxingru/workspace/GCT_classfication/Data/GCT_image_Dicom/NGGCT/072_zhongjingsheng/M664027___20170518084522___9___t1_tir_tra_P3-t1_tir_tra_P3.nii.gz'
    read_nii_file(file_path)
    file_path='/home/chenxingru/workspace/GCT_classfication/Data/GCT_image_Dicom/NGGCT/072_zhongjingsheng/M664027___20170518084522___1___localizer-localizer_i00003.nii.gz'
    read_nii_file(file_path)
    file_path='/home/chenxingru/workspace/GCT_classfication/Data/GCT_image_Dicom/NGGCT/072_zhongjingsheng/M664027___20170518084522___1___localizer-localizer_i00001.nii.gz'
    read_nii_file(file_path)    
    
    file_path='/home/chenxingru/workspace/GCT_classfication/Data/GCT_image_Dicom/pure germinoma/48_lvsicheng/20150511002770/1_72CDCB9E9130410AAC1775FEB10BC85/1.3.12.2.1107.5.2.36.40717.2015051122263663018512599.dcm'
    read_dicom_file(file_path)
    
    file_path='/home/chenxingru/workspace/GCT_classfication/Data/GCT_image_Dicom/pure germinoma/48_lvsicheng/20150511002770/1_72CDCB9E9130410AAC1775FEB10BC85/1.3.12.2.1107.5.2.36.40717.2015051122264060626612603.dcm'
    read_dicom_file(file_path)
    
    #file_path='/home/chenxingru/workspace/GCT_classfication/Data/GCT_image_Dicom/pure germinoma/48_lvsicheng/20150511002770/1_72CDCB9E9130410AAC1775FEB10BC85/1.3.12.2.1107.5.2.36.40717.2015051122264457629212607.dcm'
    #read_dicom_file(file_path)
    
    
    file_path='/home/chenxingru/workspace/GCT_classfication/Data/GCT_image_Dicom/pure germinoma/48_lvsicheng/20150511002770/9_A6993C7ED81149F6B948CCF95DE030A6/1.3.12.2.1107.5.2.36.40717.2015051122382990770014552.dcm'
    read_dicom_file(file_path)
    file_path='/home/chenxingru/workspace/GCT_classfication/Data/GCT_image_Dicom/pure germinoma/48_lvsicheng/20150511002770/9_A6993C7ED81149F6B948CCF95DE030A6/1.3.12.2.1107.5.2.36.40717.2015051122374619490814491.dcm'
    read_dicom_file(file_path)
    
    
    #dicom_dir='/home/chenxingru/workspace/GCT_classfication/Data/GCT_image_Dicom/pure germinoma/48_lvsicheng/20150511002770/9_A6993C7ED81149F6B948CCF95DE030A6'
    #output_nii_dir='/home/liuziyang/workspace/workspace/GCT_classfication/Data/GCT_image_Dicom/test.nii'

    #read_nii_file(output_nii_dir)
    
    file_path="/home/liuziyang/workspace/workspace/GCT_classfication/Data/GCT_image_Dicom/procnii_2/02_liuguoqing/3/MR2nii/4/T1.nii.gz"
    read_nii_file(file_path)
    
    file_path="/home/liuziyang/workspace/workspace/GCT_classfication/Data/GCT_image_Dicom/procnii/001_liutaoning_T1.nii.gz"
    read_nii_file(file_path)
    
    
    file_path="/home/liuziyang/workspace/workspace/GCT_classfication/Data/GCT_image_Dicom/pure_germinoma/40_chenhaiyang/20171123001912/2_7061810A9268452191ACB12D74A35FCC/1.3.12.2.1107.5.1.4.53621.30000017112123302226500009417.dcm"
    read_dicom_file(file_path)

    file_path="/home/liuziyang/workspace/workspace/GCT_classfication/Data/GCT_image_Dicom/pure_germinoma/40_chenhaiyang/20171123001912/2_7061810A9268452191ACB12D74A35FCC/1.3.12.2.1107.5.1.4.53621.30000017112123302226500009425.dcm"
    read_dicom_file(file_path)
    
    file_path="/home/liuziyang/workspace/workspace/GCT_classfication/Data/GCT_image_Dicom/pure_germinoma/40_chenhaiyang/20171123001912/1_BBB3B893BC7D4EBD8D76679A1B2EC0D3/1.3.12.2.1107.5.1.4.53621.30000017112123302226500009411.dcm"
    read_dicom_file(file_path)
    
    file_path="/home/liuziyang/workspace/workspace/GCT_classfication/Data/GCT_image_Dicom/procnii_2/40_chenhaiyang/1_BBB3B893BC7D4EBD8D76679A1B2EC0D3_topogram__1.0__t20s_Turn.nii.gz"
    read_nii_file(file_path)
    """