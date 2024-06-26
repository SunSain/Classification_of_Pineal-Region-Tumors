import argparse
<<<<<<< HEAD

parser = argparse.ArgumentParser(description='Binary-Brain_T1_1208_tumor classification')
# =========== save path ================ #

=======
import sys
import os
import numpy as np
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

free_gpu_id=0
parser = argparse.ArgumentParser(description='Binary-Brain_T1_1208_tumor classification')
# =========== save path ================ #

parser.add_argument('--free_gpu_id' ,default=free_gpu_id          ,type=int,   help="The number of free GPU")

>>>>>>> 3a4a4f2 (20240625-code)
"""
#=↓=twi============================================================================================================#
parser.add_argument('--model_name'     ,default='best_model.pth.tar'    ,type=str, help="Checkpoint file name")
parser.add_argument('--output_dir'     ,default='/opt/chenxingru/Pineal_region/for_binary_classi/model_result/1113/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--pic_output_dir'     ,default='/opt/chenxingru/Pineal_region/for_binary_classi/pic_result/1113/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--ps'     ,default='_1113_2_batch=16_newdata_binary_used_tio_augmen_new-cross-ComposedAug_'              ,type=str, help="affix needed to be informed in saved_file_name")


parser.add_argument('--x_pretrain_path'     ,default='/opt/chenxingru/Pineal_region/for_binary_classi/model_result/1027SEC_KaimingInit/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--x_ps'     ,default='_1027SEC_KaimingInit_batch=8_newdata_binary_used_tio_augmen_new-cross-ComposedAug_'              ,type=str, help="affix needed to be informed in saved_file_name")

parser.add_argument('--y_pretrain_path'     ,default='/opt/chenxingru/Pineal_region/for_binary_classi/T2_model_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--y_ps'     ,default='_1110_T2_batch=16_ACC_newdata_binary_ Tencent_Init_used_tio_augmen_new-cross-ComposedAug_'              ,type=str, help="affix needed to be informed in saved_file_name")
parser.add_argument('--T1_data_path'       ,default='/opt/chenxingru/Pineal_region/9_24_regis_AXandTra_best_f_NoTest/'         ,type=str, help=" data path ")
parser.add_argument('--T1_testdata_path'   ,default="/opt/chenxingru/Pineal_region/9_24_regis_AXandTra_best_f_Test/"         ,type=str, help=" data path ")
parser.add_argument('--T2_data_path'       ,default='/opt/chenxingru/Pineal_region/10_14_T2_regis_AXandTra_best_f_NoTest/'         ,type=str, help=" data path ")
parser.add_argument('--T2_testdata_path'   ,default="/opt/chenxingru/Pineal_region/10_14_T2_regis_AXandTra_best_f_Test/"         ,type=str, help=" data path ")
parser.add_argument('--sum_write_Mark'     ,default='/1113batch16/second/'              ,type=str, help="affix needed to be informed in saved_file_name")
#=↑=twi============================================================================================================#
"""

#=↓=T1========================================================================#
#parser.add_argument('--data_path'       ,default='/opt/chenxingru/Pineal_region/11_23_No_regis_T1_best_Notest_No103/'         ,type=str, help=" data path ")
#parser.add_argument('--testdata_path'   ,default="/opt/chenxingru/Pineal_region/11_23_No_regis_T1_best_TEST/"         ,type=str, help=" data path ")
parser.add_argument('--infor_path'   ,default="/opt/chenxingru/Pineal_region/pineal_csv.txt"         ,type=str, help=" data path ")

<<<<<<< HEAD
parser.add_argument('--data_path'       ,default='/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_No_regis_T1+C_best_Notest/'         ,type=str, help=" data path ")
parser.add_argument('--testdata_path'   ,default="/opt/chenxingru/Pineal_region/after_12_08/230305_data/12_03_No_regis_T1+C_TEST_No22/"        ,type=str, help=" data path ")

parser.add_argument('--skullstriped_path',   default="/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_skullstriped_No_regis_T1+C_best_Notest/", type =str)
#parser.add_argument('--root_mask_radiomics',   default="/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_seg_No_regis_T1+C_best_Notest", type =str)
=======
#parser.add_argument('--data_path'       ,default='/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_No_regis_T1+C_best_Notest/'         ,type=str, help=" data path ")
#parser.add_argument('--testdata_path'   ,default="/opt/chenxingru/Pineal_region/after_12_08/230305_data/12_03_No_regis_T1+C_TEST_No22/"        ,type=str, help=" data path ")

#parser.add_argument('--vice_testdata_path'   ,default="/home/chenxr/new_NII/T1C_ax_tra/"        ,type=str, help=" data path ")
#parser.add_argument('--test_root_bbx_path' ,        default= "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/test_boundingbox/", type=str)
#parser.add_argument('--root_bbx_path',   default="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/train_boundingbox/", type =str)


parser.add_argument('--skullstriped_path',   default="/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_skullstriped_No_regis_T1+C_best_Notest/", type =str)
parser.add_argument('--root_mask_radiomics',   default="/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_seg_No_regis_T1+C_best_Notest", type =str)
>>>>>>> 3a4a4f2 (20240625-code)
parser.add_argument('--test_root_mask_radiomics',   default="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/test_box/inferTs/", type =str)

#parser.add_argument('--test_root_bbx_path' ,        default= "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/test_box/masked_bounding/", type=str)

#parser.add_argument('--test_root_bbx_path' ,        default= "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/test_mask_boundingbox/", type=str)
<<<<<<< HEAD
parser.add_argument('--test_root_bbx_path' ,        default= "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/test_boundingbox/", type=str)
#/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/test_boundingbox/
#parser.add_argument('--root_mask_radiomics',   default="/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_seg_No_regis_T1+C_best_Notest/", type =str)
#parser.add_argument('--root_mask_radiomics',   default="/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_bbox_No_regis_T1+C_best_Notest/", type =str)
parser.add_argument('--root_mask_radiomics',   default="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/train_boundingbox/", type =str)

#parser.add_argument('--root_bbx_path' ,default= "/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_bbox_No_regis_T1+C_best_Notest/", type=str)
parser.add_argument('--root_bbx_path' ,default= "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/train_mask_boundingbox/", type=str)
=======

#/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/test_boundingbox/
#parser.add_argument('--root_mask_radiomics',   default="/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_seg_No_regis_T1+C_best_Notest/", type =str)
#parser.add_argument('--root_mask_radiomics',   default="/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_bbox_No_regis_T1+C_best_Notest/", type =str)
#parser.add_argument('--root_mask_radiomics',   default="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/train_boundingbox/", type =str)

#parser.add_argument('--root_bbx_path' ,default= "/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_bbox_No_regis_T1+C_best_Notest/", type=str)
#parser.add_argument('--root_bbx_path' ,default= "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/train_mask_boundingbox/", type=str)
>>>>>>> 3a4a4f2 (20240625-code)

parser.add_argument('--use_MGDA' ,default=True          ,type=bool,   help="use MGDA or not")
#parser.add_argument('--data_path'       ,default='/content/data_local/11_23_No_regis_T1_best_Notest_No103/'         ,type=str, help=" data path ")
#parser.add_argument('--testdata_path'   ,default="/content/data_local/11_23_No_regis_T1_best_TEST/"         ,type=str, help=" data path ")
#parser.add_argument('--output_dir'     ,default='/content/data_local/model_result/only_Composed/1207/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--pic_output_dir'     ,default='/content/data_local/pic_result/only_Composed/1207/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")

parser.add_argument('--ps'     ,default='_0401_T1_'              ,type=str, help="affix needed to be informed in saved_file_name")

parser.add_argument('--save_nii_path'     ,default='/opt/chenxingru/Pineal_region/after_12_08/after_12_08/0401_T1c_none/'              ,type=str, help=" save transfromed data for checking legality")
parser.add_argument('--sum_write_Mark'     ,default='/0401RM4/'              ,type=str, help="affix needed to be informed in saved_file_name")

#parser.add_argument('--model_name'     ,default='best_model.pth.tar'    ,type=str, help="Checkpoint file name")
#=↑=T1========================================================================#


#=↓==T2===========================================================================#
#parser.add_argument('--output_dir'     ,default='/opt/chenxingru/Pineal_region/for_binary_classi/T2_model_result/1110/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--pic_output_dir'     ,default='/opt/chenxingru/Pineal_region/for_binary_classi/T2_pic_result/1110/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--data_path'       ,default='/opt/chenxingru/Pineal_region/10_14_T2_regis_AXandTra_best_f_NoTest/'         ,type=str, help=" data path ")
#parser.add_argument('--testdata_path'   ,default="/opt/chenxingru/Pineal_region/10_14_T2_regis_AXandTra_best_f_Test/"         ,type=str, help=" data path ")

#parser.add_argument('--ps'     ,default='_1110_T2_batch=16_ACC_newdata_binary_ Tencent_Init_used_tio_augmen_new-cross-ComposedAug_'              ,type=str, help="affix needed to be informed in saved_file_name")
#parser.add_argument('--sum_write_Mark'     ,default='/1110batch16/second/'              ,type=str, help="affix needed to be informed in saved_file_name")

#parser.add_argument('--data_path'       ,default='/opt/chenxingru/Pineal_region/1014_T2_NotRegisted_TraAx/'         ,type=str, help=" data path ")
#parser.add_argument('--testdata_path'   ,default="/opt/chenxingru/Pineal_region/T2_Noregistered_test/"         ,type=str, help=" data path ")
#=↑==T2===========================================================================#

#parser.add_argument('--data_path'       ,default='/content/data_local/12_03_No_regis_T2_best_Notest'         ,type=str, help=" data path ")
#parser.add_argument('--testdata_path'   ,default="/content/data_local/12_03_No_regis_T2_best_TEST"         ,type=str, help=" data path ")

#parser.add_argument('--model_name'     ,default='best_model.pth.tar'    ,type=str, help="Checkpoint file name")
#parser.add_argument('--output_dir'     ,default='/content/data_local/model_result/T2/1208_RandomMask/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--pic_output_dir'     ,default='/content/data_local/pic_result/T2/1208_RandomMask/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--ps'     ,default='_1208_T2_batch=16__binary_used_RandomMask_new-cross_'              ,type=str, help="affix needed to be informed in saved_file_name")

#parser.add_argument('--save_nii_path'     ,default='/content/data_local/1208_T2_test_crop_NII_data_NotBrainCenter/'              ,type=str, help=" save transfromed data for checking legality")


#parser.add_argument('--sum_write_Mark'     ,default='/1208T2batch16/'              ,type=str, help=" save transfromed data for checking legality")



#=↓=+C========================================================================#
#parser.add_argument('--data_path'       ,default='/content/data_local/12_03_No_regis_T1+C_best_Notest'         ,type=str, help=" data path ")
#parser.add_argument('--testdata_path'   ,default="/content/data_local/12_03_No_regis_T1+C_best_TEST_No22"         ,type=str, help=" data path ")

#parser.add_argument('--model_name'     ,default='best_model.pth.tar'    ,type=str, help="Checkpoint file name")
#parser.add_argument('--output_dir'     ,default='/content/data_local/model_result/T1+C/1208_RandomMask/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--pic_output_dir'     ,default='/content/data_local/pic_result/T1+C/1208_RandomMask/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--ps'     ,default='_1208_+C_batch=16__binary_used_RandomMask_new-cross_'              ,type=str, help="affix needed to be informed in saved_file_name")

#parser.add_argument('--save_nii_path'     ,default='/content/data_local/1208_+C_test_crop_NII_data_NotBrainCenter/'              ,type=str, help=" save transfromed data for checking legality")


#parser.add_argument('--sum_write_Mark'     ,default='/1208+Cbatch16/'              ,type=str, help=" save transfromed data for checking legality")

#parser.add_argument('--model_name'     ,default='best_model.pth.tar'    ,type=str, help="Checkpoint file name")
#=↑=+C========================================================================#
<<<<<<< HEAD

=======
#tensorboard --logdir=/home/chenxr/Pineal_region/after_12_08/Results/old_T1/Two/new_appendix_batchavg_constrain_composed_ResNet18/model_result/
>>>>>>> 3a4a4f2 (20240625-code)
#tensorboard --logdir=/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/MGDA/none_composed_sampler_constrained_SelfKL_1090_ResNet24/model_result/loss_fold_
#tensorboard --logdir=/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_Weighted_SelfKL_1090_421_ResNet24/model_result/
#tensorboard --logdir=/opt/chenxingru/Pineal_region/after_12_08/Results/0401_T1C/MGDA/Med_Composed_selfKL_noL1/model_result/loss_fold_1/
#tensorboard --logdir=/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/MGDA/none_masked_constrained_train_CE/model_result/loss_fold_0/
#tensorboard --logdir=/opt/chenxingru/Pineal_region/after_12_08/Results/0401_T1C/MGDA/Med_Composed_selfKL_noL1/model_result/loss_fold_0/
#tensorboard --logdir=/opt/chenxingru/Pineal_region/after_12_08/Results/0401_T1C/MGDA/none_selfKL_noL1/model_result/loss_fold_0/
#tensorboard --logdir=/opt/chenxingru/Pineal_region/after_12_08/Results/0220_T1_result/0222_ResNet10_None_L0L1L2L3_5663/model_result/
#tensorboard --logdir=/opt/chenxingru/Pineal_region/after_12_08/Results/0220_T1_result/0221_ResNet10_None_L0L1L2L3_5111/model_result/
#tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/model_result/1113/
#tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/T2_model_result/1110/
#tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/model_result/1110/
#tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/T2_model_result/11120/second/
# tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/T2_model_result/11120/
# tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/model_result/1107/

<<<<<<< HEAD
=======
#tensorboard --logdir=/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/Two_Three/multiclass/no_MGDA/none_sampler_BatchAvg_ResNet18/model_result/

>>>>>>> 3a4a4f2 (20240625-code)
# tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/model_result/1103/

# tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/model_result/1027SEC_KaimingInit/
# tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/model_result/
# tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/T2_model_result/
<<<<<<< HEAD
=======
#tensorboard --logdir=/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/Two/new_data/none_sampler_constrained_batchavg_SelfKL_Composed_ResNet18_4/model_result/
>>>>>>> 3a4a4f2 (20240625-code)
#parser.add_argument('--ps'     ,default='_0908_used_tio_augmen_And_after_bet_'              ,type=str, help="affix needed to be informed in saved_file_name")
#parser.add_argument('--ps'     ,default='_0912_used_tio_augmen_backto0824Aug_'              ,type=str, help="affix needed to be informed in saved_file_name")
#parser.add_argument('--ps'     ,default='_0912_used_tio_augmen_'              ,type=str, help="affix needed to be informed in saved_file_name")
#parser.add_argument('--ps'     ,default='_0924_newdata_binary_used_tio_augmen_k-cross-ComposedAug_'              ,type=str, help="affix needed to be informed in saved_file_name")
#parser.add_argument('--ps'     ,default='_0917_binary_used_tio_augmen_k-cross-ComposedAug_'              ,type=str, help="affix needed to be informed in saved_file_name")

#parser.add_argument('--ps'     ,default='_0917_binary_used_tio_augm_for_vg11_2_differAdaptive_2_'              ,type=str, help="affix needed to be informed in saved_file_name")
#parser.add_argument('--ps'     ,default='_1002_newdata_binary_ Tencent_Init_used_tio_augmen_k-cross-ComposedAug_'              ,type=str, help="affix needed to be informed in saved_file_name")
#parser.add_argument('--ps'     ,default='_1004_newdata_binary_ Tencent_Init_used_tio_augmen_k-cross-ComposedAug_'              ,type=str, help="affix needed to be informed in saved_file_name")

#parser.add_argument('--ps'     ,default='_1005_newdata_binary_ Tencent_Init_used_tio_augmen_k-cross-ComposedAug_'              ,type=str, help="affix needed to be informed in saved_file_name")


#parser.add_argument('--ps'     ,default='_1010_batch=8_newdata_binary_ Tencent_Init_used_tio_augmen_new-cross-ComposedAug_'              ,type=str, help="affix needed to be informed in saved_file_name")

#parser.add_argument('--ps'     ,default='_0824_used_tio_augmen_'              ,type=str, help="affix needed to be informed in saved_file_name")

#parser.add_argument('--data_path'   ,default='/opt/chenxingru/Pineal_region/0812_T1W_AXTRA_augmentation_cache/'         ,type=str, help=" data path ")

#parser.add_argument('--pic_output_dir'     ,default='/opt/chenxingru/Pineal_region/for_binary_classi/pic_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")


#parser.add_argument('--testdata_path'   ,default='/opt/chenxingru/Pineal_region/5_28_T1W_registrated_AXandTRA_best_file_TEST/'         ,type=str, help=" data path ")




#parser.add_argument('--data_path'   ,default='/opt/chenxingru/Pineal_region/5_28_T1W_registrated_AXandTRA_best_file_NoTest/'         ,type=str, help=" data path ")
#parser.add_argument('--excel_path'     ,default='/opt/chenxingru/Pineal_region/lables/Training.xls',type=str, help="Excel file path ")
#parser.add_argument('--first_stage_net',default='/opt/chenxingru/Pineal_region/model/best.pth.tar'  ,type=str, help="When training the second stage network, appoint the trained first stage network checkpoint file path is needed ")
parser.add_argument('--npz_name'       ,default='test.npz'              ,type=str, help="After inference the trained model in test set, a npz file will be saved in assigned path. So the npz name need to be appointed. ")
parser.add_argument('--plot_name'      ,default='test.png'              ,type=str, help="After inference the trained model in test set, a scatter plot will be saved in assigned path. So the plot name need to be appointed. ")


parser.add_argument('--tencent_pth_rootdir',default='/opt/chenxingru/Pineal_region/tencent_mednet/'  ,type=str, help="When training the second stage network, appoint the trained first stage network checkpoint file path is needed ")

#=========== hyperparameter ================ #
parser.add_argument('--random_seed' ,default=42          ,type=int,   help="random_seed in spliting train, valid and test")
parser.add_argument('--num_classes' ,default=2          ,type=int,   help="The number of types of pictures")


parser.add_argument('--model_name'     ,default='best_model.pth.tar'    ,type=str, help="Checkpoint file name")
<<<<<<< HEAD
parser.add_argument('--output_dir'     ,default='/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_sampler_SelfKL_composed_1090_ResNet18/model_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--pic_output_dir' ,default='/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_sampler_SelfKL_composed_1090_ResNet18/pic_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")

parser.add_argument('--usethird'        ,default=False          ,type=bool,   help="if  want to use thirdimg, then True")

parser.add_argument('--use_radiomics' ,default=False          ,type=bool,   help="use nomalized Radiomics or not")

parser.add_argument('--usesecond'        ,default=False          ,type=bool,   help="if  want to use secondimg, then True")
parser.add_argument('--sec_pair_orig'        ,default=False          ,type=bool,   help="bbx_pair_orig_or_not")
parser.add_argument('--noSameRM'        ,default=True          ,type=bool,   help="True if keep secondimg complete")

parser.add_argument('--multiclass'        ,default=True          ,type=bool,   help="True if with 3class not 2class")

parser.add_argument('--model'       ,default='ResNet18'   ,type=str,   help="Deep learning model to do brain age estimation")
parser.add_argument('--batch_size'  ,default=4     ,type=int,   help="Batch size during training process")
=======


#parser.add_argument('--t1_path'       ,default='/opt/chenxingru/Pineal_region/11_23_No_regis_T1_best_Notest_No103/'         ,type=str, help=" data path ")
#parser.add_argument('--t2_path'       ,default='/opt/chenxingru/Pineal_region/1014_T2_NotRegisted_TraAx/'         ,type=str, help=" data path ")
#parser.add_argument('--t1c_path'       ,default='/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_No_regis_T1+C_best_Notest/'         ,type=str, help=" data path ")

#parser.add_argument('--t1_path'       ,default='/home/chenxr/new_nii_data/all_T1'         ,type=str, help=" data path ")
#parser.add_argument('--t2_path'       ,default='/home/chenxr/new_nii_data/all_T2'         ,type=str, help=" data path ")
#parser.add_argument('--t1c_path'       ,default='/home/chenxr/new_nii_data/all_T1C'         ,type=str, help=" data path ")



#parser.add_argument('--t1_test_path'       ,default='/opt/chenxingru/Pineal_region/11_23_No_regis_T1_best_TEST/'         ,type=str, help=" data path ")
#parser.add_argument('--t1c_test_path'       ,default='/opt/chenxingru/Pineal_region/after_12_08/230305_data/12_03_No_regis_T1+C_TEST_No22/'         ,type=str, help=" data path ")
#parser.add_argument('--t2_test_path'       ,default='/opt/chenxingru/Pineal_region/after_12_08/230305_data/12_03_No_regis_T2_best_TEST'         ,type=str, help=" data path ")

parser.add_argument('--data_path'       ,default='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C'         ,type=str, help=" data path ")
parser.add_argument('--testdata_path'   ,default="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C"         ,type=str, help=" data path ")
parser.add_argument('--vice_testdata_path'   ,default=""        ,type=str, help=" data path ")
parser.add_argument('--test_root_bbx_path' ,        default= "", type=str)
parser.add_argument('--root_bbx_path',   default="", type =str)
parser.add_argument('--output_dir'     ,default='/home/chenxr/Pineal_region/after_12_08/Results/tmp_test/Two/T1C_SingleBatchAvg_selfKL_Composed_RM_composed_ResNet18/model_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--pic_output_dir' ,default='/home/chenxr/Pineal_region/after_12_08/Results/tmp_test/Two/T1C_SingleBatchAvg_selfKL_Composed_RM_composed_ResNet18/pic_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--best_pretrained_model_path' ,default="/home/chenxr/Pineal_region/after_12_08/Results/united/old_T1/Two/none_composed_ResNet18/model_result/"              ,type=str, help="best baseline model")


"""
parser.add_argument('--data_path'       ,default='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C'         ,type=str, help=" data path ")
parser.add_argument('--testdata_path'   ,default="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C"        ,type=str, help=" data path ")
parser.add_argument('--vice_testdata_path'   ,default="/home/chenxr/new_NII/T1C_ax_tra/"        ,type=str, help=" data path ")
parser.add_argument('--test_root_bbx_path' ,        default= "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task13_old_T1C/seg_all_results/test/", type=str)
parser.add_argument('--root_bbx_path',              default="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task13_old_T1C/seg_all_results/train/", type =str)

#parser.add_argument('--test_root_bbx_path' ,        default= "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task012_all_T1C/for_classifying/test/", type=str)
#parser.add_argument('--root_bbx_path',   default="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task012_all_T1C/for_classifying/train/", type =str)

parser.add_argument('--output_dir'     ,default='/home/chenxr/Pineal_region/after_12_08/Results/united/old_T1C/Two/new_none_composed_ResNet18/model_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--pic_output_dir' ,default='/home/chenxr/Pineal_region/after_12_08/Results/united/old_T1C/Two/new_none_composed_ResNet18/pic_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--best_pretrained_model_path' ,default=""              ,type=str, help="best baseline model")
"""
"""
parser.add_argument('--data_path'       ,default='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T2'         ,type=str, help=" data path ")
parser.add_argument('--testdata_path'   ,default="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T2"         ,type=str, help=" data path ")
parser.add_argument('--vice_testdata_path'   ,default=""        ,type=str, help=" data path ")
parser.add_argument('--test_root_bbx_path' ,        default= "", type=str)
parser.add_argument('--root_bbx_path',   default="", type =str)
parser.add_argument('--output_dir'     ,default='/home/chenxr/Pineal_region/after_12_08/Results/united/old_T2/Two/none_composed_ResNet18/model_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--pic_output_dir' ,default='/home/chenxr/Pineal_region/after_12_08/Results/united/old_T2/Two/none_composed_ResNet18/pic_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--best_pretrained_model_path' ,default="/home/chenxr/Pineal_region/after_12_08/Results/united/old_T2/Two/none_composed_ResNet18/model_result/"              ,type=str, help="best baseline model")

"""
"""
parser.add_argument('--data_path'       ,default='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C'         ,type=str, help=" data path ")
parser.add_argument('--testdata_path'   ,default="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C"        ,type=str, help=" data path ")
parser.add_argument('--vice_testdata_path'   ,default=""        ,type=str, help=" data path ")
parser.add_argument('--test_root_bbx_path' ,        default= "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task13_old_T1C/seg_all_results/test/", type=str)
parser.add_argument('--root_bbx_path',              default="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task13_old_T1C/seg_all_results/train/", type =str)

parser.add_argument('--output_dir'     ,default='/home/chenxr/Pineal_region/after_12_08/Results/united/old_all/Two/new_pretrained_batchavg_constrain_composed_ResNet18/model_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--pic_output_dir' ,default='/home/chenxr/Pineal_region/after_12_08/Results/united/old_all/Two/new_pretrained_batchavg_constrain_composed_ResNet18/pic_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--best_pretrained_model_path' ,default="/home/chenxr/Pineal_region/after_12_08/Results/old_all/Two/new_composed_ResNet18/model_result/"              ,type=str, help="best baseline model")

#parser.add_argument('--best_pretrained_model_path' ,default="/home/chenxr/Pineal_region/after_12_08/Results/united/old_all/Two/none_composed_CE_None_ResNet18/model_result/ResNet18_CE_0401_T1__total_best__best_model_pth.tar"              ,type=str, help="best baseline model")
"""


parser.add_argument('--multi_m'        ,default=False          ,type=bool,   help="use multi-modality or single-modality")
parser.add_argument('--use_radiomics' ,default=False          ,type=bool,   help="use nomalized Radiomics or not")


parser.add_argument('--usethird'        ,default=False          ,type=bool,   help="if  want to use thirdimg, then True")

parser.add_argument('--usesecond'        ,default=False          ,type=bool,   help="if  want to use secondimg, then True")
parser.add_argument('--sec_pair_orig'        ,default=False          ,type=bool,   help="bbx_pair_orig_or_not")

parser.add_argument('--t1_path'       ,default='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1'         ,type=str, help=" data path ")
parser.add_argument('--t2_path'       ,default='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T2'         ,type=str, help=" data path ")
parser.add_argument('--t1c_path'       ,default='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C'         ,type=str, help=" data path ")
parser.add_argument('--t1_test_path'       ,default='/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1'         ,type=str, help=" data path ")
parser.add_argument('--t1c_test_path'       ,default='/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C'         ,type=str, help=" data path ")
parser.add_argument('--t2_test_path'       ,default='/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T2'         ,type=str, help=" data path ")

parser.add_argument('--noSameRM'        ,default=True          ,type=bool,   help="True if keep secondimg complete")

parser.add_argument('--multiclass'        ,default=False          ,type=bool,   help="True if with 3class not 2class")

parser.add_argument('--model'       ,default='ResNet18'   ,type=str,   help="Deep learning model to do brain age estimation")
parser.add_argument('--batch_size'  ,default=4     ,type=int,   help="Batch size during training process")
parser.add_argument('--Two_Three' ,default=True          ,type=bool,   help="train 2,3 classes together")

parser.add_argument('--patience' ,default=250          ,type=int,   help="for the earlystopping")
>>>>>>> 3a4a4f2 (20240625-code)

parser.add_argument('--lambda_0' ,default=1          ,type=int,   help="The weight of L0")
parser.add_argument('--lambda_1' ,default=0         ,type=int,   help="The weight of L1")
parser.add_argument('--lambda_2' ,default=9          ,type=int,   help="The weight of L2")
parser.add_argument('--lambda_3' ,default=0          ,type=int,   help="The weight of L3")
parser.add_argument('--CE_or_KL' ,default=True          ,type=bool,   help="using CE mode or KL mode")
<<<<<<< HEAD
parser.add_argument('--aug_form'       ,default='Composed'   ,type=str,   help="type of augmentation, such as Composed/Random_Mask/Composed_RM")

parser.add_argument('--train_form'       ,default='None'   ,type=str,   help="normal training or masked_constrained_train or(maybe) advchain_training")
parser.add_argument('--constrain_lambd'    ,default=0.5           ,type=float,   help="constrained_loss for losses of CE(selfCE) in raw/y and masked/y (no masked/raw)")

parser.add_argument('--continue_train_fold'  ,default=0     ,type=int,   help="which fold to train( for the first model, default=0,others: to cover the unexpected stop during last training)")



parser.add_argument('--lossfunc'       ,default='SelfKL'   ,type=str,   help="loss function, such as FLoss, BatchAvg")
parser.add_argument('--loss_weight'       ,default=[2,1,1]   ,type=list,   help="weight of loss of each imbalanced class")
parser.add_argument('--FL_gamma'    ,default=2           ,type=float,   help="gamma for Focal_Loss")

=======
parser.add_argument('--aug_form'       ,default='Composed_RM'   ,type=str,   help="type of augmentation, such as Composed/Random_Mask/Composed_RM")

parser.add_argument('--train_form'       ,default='single_batchavg_train'   ,type=str,   help="normal training or masked_constrained_train or(maybe) advchain_training")
parser.add_argument('--constrain_lambd'    ,default=0.5           ,type=float,   help="constrained_loss for losses of CE(selfCE) in raw/y and masked/y (no masked/raw)")

parser.add_argument('--continue_train_fold'  ,default=2     ,type=int,   help="which fold to train( for the first model, default=0,others: to cover the unexpected stop during last training)")



parser.add_argument('--lossfunc'       ,default='SingleBatchAvg_selfKL'   ,type=str,   help="loss function, such as FLoss, BatchAvg")
parser.add_argument('--loss_weight'       ,default=[2,1,1]   ,type=list,   help="weight of loss of each imbalanced class")
parser.add_argument('--FL_gamma'    ,default=2           ,type=float,   help="gamma for Focal_Loss")

parser.add_argument('--kfold'    ,default=3           ,type=float,   help="gamma for Focal_Loss")

parser.add_argument('--batchavg_lamda'    ,default=0.8           ,type=float,   help="gamma for batchavg_lamda")

parser.add_argument('--single_batchavg_lamda'    ,default=1           ,type=float,   help="gamma for single_batchavg_lamda")
parser.add_argument('--selfKL_lamda'    ,default=2           ,type=float,   help="gamma for single_batchavg_lamda")

>>>>>>> 3a4a4f2 (20240625-code)


parser.add_argument('--warm_up_iter' ,default=20           ,type=int,   help="warm up epochs")
parser.add_argument('--lr_step'      ,default=10           ,type=int,   help="warm up epochs")
<<<<<<< HEAD
parser.add_argument('--max_lr'      ,default=0.1           ,type=int,   help="top limit of  learing rate")
parser.add_argument('--min_lr'      ,default=1e-5           ,type=int,   help="down limit of  learing rate")
=======
parser.add_argument('--max_lr'      ,default=0.1           ,type=float,   help="top limit of  learing rate")
parser.add_argument('--min_lr'      ,default=1e-5           ,type=float,   help="down limit of  learing rate")
>>>>>>> 3a4a4f2 (20240625-code)


parser.add_argument('--num_workers' ,default=8           ,type=int,   help="The number of worker for dataloader")
parser.add_argument('--epochs'      ,default=300       ,type=int,   help="Total training epochs")
parser.add_argument('--lr'          ,default=1e-4        ,type=float, help="Initial learning rate")
parser.add_argument('--print_freq'  ,default=10           ,type=int,   help="Training log print interval")
parser.add_argument('--weight_decay',default=5e-4        ,type=float, help="L2 weight decay ")
parser.add_argument('--dis_range'   ,default=5           ,type=int,   help="Discritize step when training the second stage network")
parser.add_argument('--adv_w'   ,default=0.2           ,type=float,   help="weight of advchain_loss")


# =========== loss function ================ #
args = parser.parse_args()
opt = args 