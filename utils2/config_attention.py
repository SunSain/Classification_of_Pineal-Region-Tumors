import argparse
import sys
import os
import numpy as np
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

free_gpu_id=1
parser = argparse.ArgumentParser(description='Binary-Brain_T1_1208_tumor classification')
# =========== save path ================ #

parser.add_argument('--free_gpu_id' ,default=free_gpu_id          ,type=int,   help="The number of free GPU")

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

parser.add_argument('--infor_path'   ,default="/opt/chenxingru/Pineal_region/pineal_csv.txt"         ,type=str, help=" data path ")


parser.add_argument('--skullstriped_path',   default="/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_skullstriped_No_regis_T1+C_best_Notest/", type =str)
parser.add_argument('--test_root_mask_radiomics',   default="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/test_box/inferTs/", type =str)

parser.add_argument('--root_mask_radiomics',   default="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/combined_boxes/train_boundingbox/", type =str)


parser.add_argument('--use_MGDA' ,default=True          ,type=bool,   help="use MGDA or not")

parser.add_argument('--ps'     ,default='_0401_T1_'              ,type=str, help="affix needed to be informed in saved_file_name")

parser.add_argument('--save_nii_path'     ,default='/opt/chenxingru/Pineal_region/after_12_08/after_12_08/0401_T1c_none/'              ,type=str, help=" save transfromed data for checking legality")
parser.add_argument('--sum_write_Mark'     ,default='/0401RM4/'              ,type=str, help="affix needed to be informed in saved_file_name")
parser.add_argument('--npz_name'       ,default='test.npz'              ,type=str, help="After inference the trained model in test set, a npz file will be saved in assigned path. So the npz name need to be appointed. ")
parser.add_argument('--plot_name'      ,default='test.png'              ,type=str, help="After inference the trained model in test set, a scatter plot will be saved in assigned path. So the plot name need to be appointed. ")


parser.add_argument('--tencent_pth_rootdir',default='/opt/chenxingru/Pineal_region/tencent_mednet/'  ,type=str, help="When training the second stage network, appoint the trained first stage network checkpoint file path is needed ")

#=========== hyperparameter ================ #
parser.add_argument('--random_seed' ,default=42          ,type=int,   help="random_seed in spliting train, valid and test")
parser.add_argument('--num_classes' ,default=2          ,type=int,   help="The number of types of pictures")


parser.add_argument('--model_name'     ,default='best_model.pth.tar'    ,type=str, help="Checkpoint file name")


parser.add_argument('--data_path'       ,default='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C'         ,type=str, help=" data path ")
parser.add_argument('--testdata_path'   ,default="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C"         ,type=str, help=" data path ")
parser.add_argument('--vice_testdata_path'   ,default=""        ,type=str, help=" data path ")
parser.add_argument('--test_root_bbx_path' ,        default= "", type=str)
parser.add_argument('--root_bbx_path',   default="", type =str)

#parser.add_argument('--output_dir'     ,default='/home/chenxr/Pineal_region/after_12_08/Results/new_arch_New_attention_3_cnn+attention/Two/three_modality_best_attention_SingelBatchavg/model_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--pic_output_dir' ,default='/home/chenxr/Pineal_region/after_12_08/Results/new_arch_New_attention_3_cnn+attention/Two/three_modality_best_attention_SingelBatchavg/pic_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--atten_model',default='Transformer',type=str,help="Transformer or Concatenation_mlp")
parser.add_argument('--new_arch' ,default=False          ,type=bool,   help="use attention or not")

#parser.add_argument('--output_dir'     ,default='/home/chenxr/Pineal_region/after_12_08/Results/New_concatenation_mlp/Two/three_modality_SelfKL/model_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--pic_output_dir' ,default='/home/chenxr/Pineal_region/after_12_08/Results/New_concatenation_mlp/Two/three_modality_SelfKL/pic_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--atten_model',default='Concateation_MLP',type=str,help="Transformer , Concateation_MLP or SelfAttention_MLP_Transformer")

#parser.add_argument('--output_dir'     ,default='/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/SelfAttention_concatenation_mlp/Two/three_modality_composed_SingleBatchavg/model_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--pic_output_dir' ,default='/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/SelfAttention_concatenation_mlp/Two/three_modality_composed_SingleBatchavg/pic_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--atten_model',default='SelfAttention_MLP_Transformer',type=str,help="Transformer , Concateation_MLP or SelfAttention_MLP_Transformer")
#parser.add_argument('--two_stage' ,default=False          ,type=bool,   help="use attention or not")




#parser.add_argument('--output_dir'     ,default='/home/chenxr/Pineal_region/after_12_08/Results/SelfAttention_CrossAttention_mlp_2Stage_training/Two/three_modality_SingleBatchavg/model_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--pic_output_dir' ,default='/home/chenxr/Pineal_region/after_12_08/Results/SelfAttention_CrossAttention_mlp_2Stage_training/Two/three_modality_SingleBatchavg/pic_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--atten_model',default='SelfAttention_CrossAttention_Transformer',type=str,help="Transformer , Concateation_MLP , SelfAttention_MLP_Transformer or SelfAttention_CrossAttention_Transformer")

#parser.add_argument('--output_dir'     ,default='/home/chenxr/Pineal_region/after_12_08/Results/Res_CrossAttention_mlp/Two/three_modality_Batchavg/model_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--pic_output_dir' ,default='/home/chenxr/Pineal_region/after_12_08/Results/Res_CrossAttention_mlp/Two/three_modality_Batchavg/pic_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--atten_model',default='CrossAttention_ResTransformer',type=str,help="Transformer , Concateation_MLP , SelfAttention_MLP_Transformer , SelfAttention_CrossAttention_Transformer or CrossAttention_ResTransformer")



#parser.add_argument('--output_dir'     ,default='/home/chenxr/Pineal_region/after_12_08/Results/Clinical_pure_binaryRegression_5Layer/Two/three_modality/model_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--pic_output_dir' ,default='/home/chenxr/Pineal_region/after_12_08/Results/Clinical_pure_binaryRegression_5Layer/Two/three_modality/pic_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--atten_model',default='Binary_LogisticRegression',type=str,help="Transformer , Concateation_MLP , SelfAttention_MLP_Transformer or SelfAttention_CrossAttention_Transformer")


#parser.add_argument('--output_dir'     ,default='/home/chenxr/Pineal_region/after_12_08/Results/clinical_logistic/Two/all/three_modality/age_new/model_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--pic_output_dir' ,default='/home/chenxr/Pineal_region/after_12_08/Results/clinical_logistic/Two/all/three_modality/age_new/pic_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--atten_model',default='true_logistic_regression',type=str,help="Transformer , Concateation_MLP , SelfAttention_MLP_Transformer or SelfAttention_CrossAttention_Transformer")
#parser.add_argument('--clinical_type',default='gender',type=str,help="Transformer , Concateation_MLP , SelfAttention_MLP_Transformer or SelfAttention_CrossAttention_Transformer")




parser.add_argument('--output_dir'     ,default='/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/arch1_111_BatchAvg/model_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--pic_output_dir' ,default='/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/arch1_111_BatchAvg/pic_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--atten_model',default='MixAttention',type=str,help="Transformer or Concatenation_mlp")

parser.add_argument('--w1'   ,default=0.5          ,type=float,   help="weight of advchain_loss")
parser.add_argument('--w2'   ,default=0.5          ,type=float,   help="weight of advchain_loss")
parser.add_argument('--w3'   ,default=0.5         ,type=float,   help="weight of advchain_loss")
parser.add_argument('--use_clinical',   default=True, type =bool)



parser.add_argument('--multi_m'        ,default=True          ,type=bool,   help="use multi-modality or single-modality")
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

parser.add_argument('--patience' ,default=80          ,type=int,   help="for the earlystopping")

parser.add_argument('--lambda_0' ,default=1          ,type=int,   help="The weight of L0")
parser.add_argument('--lambda_1' ,default=0         ,type=int,   help="The weight of L1")
parser.add_argument('--lambda_2' ,default=9          ,type=int,   help="The weight of L2")
parser.add_argument('--lambda_3' ,default=0          ,type=int,   help="The weight of L3")
parser.add_argument('--CE_or_KL' ,default=True          ,type=bool,   help="using CE mode or KL mode")
parser.add_argument('--aug_form'       ,default='Composed_RM'   ,type=str,   help="type of augmentation, such as Composed/Random_Mask/Composed_RM")

parser.add_argument('--train_form'       ,default='None'   ,type=str,   help="normal training or masked_constrained_train or(maybe) advchain_training, single_batchavg_train")
parser.add_argument('--constrain_lambd'    ,default=0.5           ,type=float,   help="constrained_loss for losses of CE(selfCE) in raw/y and masked/y (no masked/raw)")

parser.add_argument('--continue_train_fold'  ,default=0     ,type=int,   help="which fold to train( for the first model, default=0,others: to cover the unexpected stop during last training)")



parser.add_argument('--lossfunc'       ,default='BatchAvg'   ,type=str,   help="loss function, such as FLoss, BatchAvg,SingleBatchAvg")
parser.add_argument('--loss_weight'       ,default=[2,1,1]   ,type=list,   help="weight of loss of each imbalanced class")
parser.add_argument('--FL_gamma'    ,default=2           ,type=float,   help="gamma for Focal_Loss")




parser.add_argument('--warm_up_iter' ,default=20           ,type=int,   help="warm up epochs")
parser.add_argument('--lr_step'      ,default=10           ,type=int,   help="warm up epochs")
parser.add_argument('--max_lr'      ,default=0.1           ,type=float,   help="top limit of  learing rate")
parser.add_argument('--min_lr'      ,default=1e-5           ,type=float,   help="down limit of  learing rate")


parser.add_argument('--num_workers' ,default=8           ,type=int,   help="The number of worker for dataloader")
parser.add_argument('--epochs'      ,default=200       ,type=int,   help="Total training epochs")
parser.add_argument('--lr'          ,default=1e-4        ,type=float, help="Initial learning rate")
parser.add_argument('--print_freq'  ,default=10           ,type=int,   help="Training log print interval")
parser.add_argument('--weight_decay',default=5e-4        ,type=float, help="L2 weight decay ")
parser.add_argument('--dis_range'   ,default=5           ,type=int,   help="Discritize step when training the second stage network")
parser.add_argument('--adv_w'   ,default=0.2           ,type=float,   help="weight of advchain_loss")


# =========== loss function ================ #
args = parser.parse_args()
opt = args 