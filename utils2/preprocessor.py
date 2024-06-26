"""You need to modify this file to fit your data organizations.
"""
import sys

import numpy as np
import SimpleITK as sitk
from collections import OrderedDict

from scipy.ndimage.interpolation import map_coordinates
#from batchgenerators.augmentations.utils import resize_segmentation
from skimage.transform import resize
import os
import pandas as pd
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

from utils2.config import  opt

from utils2.brain_extractor import BrainExtractor

def get_new_shape(itk_img, cropped_img, target_spacing):
    orig_spacing = itk_img.GetSpacing()
    cropped_shape = cropped_img.shape
    ratio_orig_target = (np.array(orig_spacing) / np.array(target_spacing)).astype(np.float)
    calibrate_ratio_order = [ratio_orig_target[2], ratio_orig_target[0], ratio_orig_target[1]]
    new_shape = np.round((np.array(calibrate_ratio_order) * cropped_shape)).astype(int)
    return new_shape


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


def resample_data_or_seg(data, new_shape, order=0, is_seg=False,axis=[0],do_separate_z=False, order_z=0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param axis:
    :param order: interpolation modes, order 0 for image, order 3 for segmentations
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == len(data.shape) - 1

    resize_fn = resize
    
    if is_seg:
        #resize_fn = resize_segmentation
        resize_fn = resize
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    #print("shape: ",shape)
    new_shape = np.array(new_shape)

    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            #print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
                #print("new_shape_2d: ",new_shape_2d)
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
                #print("new_shape_2d: ",new_shape_2d)
            else:
                new_shape_2d = new_shape[:-1]
                #print("new_shape_2d: ",new_shape_2d)

            reshaped_final_data = []
            for c in range(data.shape[0]):
                #print("c: ",c)
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    
                    if axis == 0:
                        #print("shape_size: ",data[c,slice_id].shape)
                        reshaped_data.append(
                            resize_fn(data[c,slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    elif axis == 1:
                        reshaped_data.append(
                            resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    else:
                        reshaped_data.append(
                            resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                #print("slice_id: ",slice_id)
                reshaped_data = np.stack(reshaped_data, axis)
                
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,
                                                                   mode='nearest').astype(dtype_data))
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped.astype(dtype_data))
                else:
                    #print("INTO else")
                    reshaped_final_data.append(reshaped_data.astype(dtype_data))
            #print("reshaped_final_data: ",len(reshaped_final_data),len(reshaped_final_data[0]),len(reshaped_final_data[0][0]))
            reshaped_final_data = np.vstack(reshaped_final_data)
            #print("reshaped_final_data: ",len(reshaped_final_data),len(reshaped_final_data[0]),len(reshaped_final_data[0][0]))
            

            #plt.imshow(reshaped_final_data[12])
            #plt.show()
        else:
            #print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                res= resize_fn(data[c], new_shape, order, **kwargs)[None].astype(dtype_data)
                #print("res: ",len(res))
                reshaped.append(res)
                #print("reshaped",len(reshaped))
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return


def get_image_slicer_to_crop(nonzero_mask):
    outside_value = 0
    #print("nonzero_mask__2: ",len(nonzero_mask))
    mask_voxel_coords = np.where(nonzero_mask != outside_value)
    #print("mask_voxel_coords: ",mask_voxel_coords)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    #print("resizer: ",resizer)
    return resizer


def crop_image(itk_img):
    img = sitk.GetArrayFromImage(itk_img).astype(np.float)
    threshold = np.percentile(img, 50)
    brain_extractor = BrainExtractor()
    nonzero_mask = brain_extractor.get_brain_mask(itk_img, th=threshold)
    #print("nonzero_mask__0: ",nonzero_mask.GetSize())
    nonzero_mask = sitk.GetArrayFromImage(nonzero_mask)
    #print("nonzero_mask__1: ",len(nonzero_mask))
    resizer = get_image_slicer_to_crop(nonzero_mask)
    cropped_img = img[resizer]
    cropped_nonzero_mask = nonzero_mask[resizer]
    #print("cropped_nonzero_mask:",cropped_nonzero_mask)
    test_mask = cropped_nonzero_mask>0
    #print("test_mask: ",test_mask)
    nonzero = np.count_nonzero(cropped_nonzero_mask)
    #print("nonzeros: ",nonzero)
    return cropped_img, cropped_nonzero_mask


class Preprocessor(object):
    def __init__(self, target_spacing):
        self.target_spacing = target_spacing

    def resample_patient(self, itk_img, cropped_img, cropped_mask):
        new_shape = get_new_shape(itk_img, cropped_img, self.target_spacing)
<<<<<<< HEAD
        #print("new_shape: ",new_shape)
=======
        print("new_shape: ",new_shape)
>>>>>>> 3a4a4f2 (20240625-code)
        cropped_img = np.expand_dims(cropped_img, axis=0)
        cropped_mask = np.expand_dims(cropped_mask, axis=0)
        resampled_img = resample_data_or_seg(cropped_img, new_shape, is_seg=False, axis=[0],
                                             do_separate_z=True, order_z=0)
        resampled_mask = resample_data_or_seg(cropped_mask, new_shape, is_seg=True, axis=[0],
                                              do_separate_z=True, order_z=0)
        return array2image(resampled_img, itk_img, self.target_spacing), array2image(resampled_mask, itk_img,
                                                                                     self.target_spacing)

    def run(self, img_path):
        itk_img = sitk.ReadImage(img_path)
        itk_img = sitk.DICOMOrient(itk_img, 'LPS')
        print("origin_img.data: ")
        print(itk_img.GetSize())
<<<<<<< HEAD
=======
        print("_Spacing: ",itk_img.GetSpacing())
>>>>>>> 3a4a4f2 (20240625-code)
        print(itk_img.GetSpacing())
        #print("itk_img.data: ")
        #print(itk_img.GetSize())
        #print(itk_img.GetSpacing())
        cropped_img, cropped_nonzero_mask = crop_image(itk_img)
        #print("cropped_img.data: ")
        #print(len(cropped_img))
        resampled_img, resampled_mask = self.resample_patient(itk_img, cropped_img, cropped_nonzero_mask)
        #save_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/test/test3.nii.gz"
        #sitk.WriteImage(resampled_img,save_path)
        return resampled_img, resampled_mask

if __name__ =="__main__":
    data_path=opt.data_path
    test_data_path = opt.testdata_path
    preprocessor = Preprocessor(target_spacing=[0.45,0.45,6.5])
    sub_fns = sorted(os.listdir(data_path))
    for i, f in enumerate(sub_fns):
            fn = f.split("_")
            #get label: 0<sid<61: label =0 | 61<sid: label=1
            try:
                    sid=int(fn[0])
            except:
                print("it's the_select_file")
            if sid<61: 
                slabel=0
            else: 
                slabel =1
            #print("sid: ",sid," ;slabel: ",slabel)

            smale = 0
            sub_path = os.path.join(data_path, f)

            img_data, mask_data = preprocessor.run(sub_path)
            #print("img_data: ",img_data)
            #print("img_mask: ",mask_data)
            print("img.data: ")
            print(img_data.GetSize())
            print(img_data.GetSpacing())
            
            print("mask_data.data: ")
            print(mask_data.GetSize())
            print(mask_data.GetSpacing())
     
