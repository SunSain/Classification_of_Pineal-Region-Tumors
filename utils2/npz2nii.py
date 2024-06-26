import numpy as np
import SimpleITK as sitk
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

if __name__=="__main__":
    npz_path="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_processed/Task010_BTumour_T1C/nnUNetData_plans_v2.1_stage0/BRATS_001.npz"
    origin_img_path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task010_BTumour_T1C/imagesTr/BRATS_001_0000.nii.gz"
    ac = np.load(npz_path)
    print("ac: ",ac)
