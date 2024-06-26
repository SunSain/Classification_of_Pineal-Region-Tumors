import argparse

parser = argparse.ArgumentParser(description='Binary-Brain_T1_1208_tumor classification')
# =========== save path ================ #

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
parser.add_argument('--data_path'       ,default='/opt/chenxingru/Pineal_region/11_23_No_regis_T1_best_Notest_No103/'         ,type=str, help=" data path ")
parser.add_argument('--testdata_path'   ,default="/opt/chenxingru/Pineal_region/11_23_No_regis_T1_best_TEST/"         ,type=str, help=" data path ")

parser.add_argument('--model_name'     ,default='best_model.pth.tar'    ,type=str, help="Checkpoint file name")
parser.add_argument('--output_dir'     ,default='/opt/chenxingru/Pineal_region/after_12_08/Results/20230215/T1_result/for_advchain/0216_None_CE/model_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--pic_output_dir'  ,default='/opt/chenxingru/Pineal_region/after_12_08/Results/20230215/T1_result/for_advchain/0216_None_CE/pic_result/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")

#parser.add_argument('--data_path'       ,default='/content/data_local/11_23_No_regis_T1_best_Notest_No103/'         ,type=str, help=" data path ")
#parser.add_argument('--testdata_path'   ,default="/content/data_local/11_23_No_regis_T1_best_TEST/"         ,type=str, help=" data path ")
#parser.add_argument('--output_dir'     ,default='/content/data_local/model_result/only_Composed/1207/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
#parser.add_argument('--pic_output_dir'     ,default='/content/data_local/pic_result/only_Composed/1207/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")

parser.add_argument('--ps'     ,default='_0216_advchain_T1_'              ,type=str, help="affix needed to be informed in saved_file_name")

parser.add_argument('--save_nii_path'     ,default='/opt/chenxingru/Pineal_region/after_12_08/after_12_08/for_advchain/0216_L0L1_52_T1/'              ,type=str, help=" save transfromed data for checking legality")
parser.add_argument('--sum_write_Mark'     ,default='/0216RM8/'              ,type=str, help="affix needed to be informed in saved_file_name")

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



#tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/model_result/1113/
#tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/T2_model_result/1110/
#tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/model_result/1110/
#tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/T2_model_result/11120/second/
# tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/T2_model_result/11120/
# tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/model_result/1107/

# tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/model_result/1103/

# tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/model_result/1027SEC_KaimingInit/
# tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/model_result/
# tensorboard --logdir=/opt/chenxingru/Pineal_region/for_binary_classi/T2_model_result/
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

parser.add_argument('--model'       ,default='ResNet10'   ,type=str,   help="Deep learning model to do brain age estimation")
parser.add_argument('--batch_size'  ,default=2     ,type=int,   help="Batch size during training process")

parser.add_argument('--lambda_0' ,default=5          ,type=int,   help="The weight of L0")
parser.add_argument('--lambda_1' ,default=2         ,type=int,   help="The weight of L1")
parser.add_argument('--lambda_2' ,default=0          ,type=int,   help="The weight of L2")
parser.add_argument('--lambda_3' ,default=0          ,type=int,   help="The weight of L3")
parser.add_argument('--CE_or_KL' ,default=True          ,type=bool,   help="using CE mode or KL mode")


parser.add_argument('--aug_form'       ,default='None'   ,type=str,   help="type of augmentation")

#parser.add_argument('--aug_form'       ,default='Composed'   ,type=str,   help="type of augmentation")
#parser.add_argument('--aug_form'       ,default='All'   ,type=str,   help="type of augmentation")



parser.add_argument('--lossfunc'       ,default='CE'   ,type=str,   help="loss function")
parser.add_argument('--loss_weight'       ,default=[3,2,1]   ,type=list,   help="weight of loss of each imbalanced class")

parser.add_argument('--constrain_lambd'    ,default=10           ,type=int,   help="constrained loss for raw/y and masked/y (no masked/raw)")

parser.add_argument('--warm_up_iter' ,default=20           ,type=int,   help="warm up epochs")
parser.add_argument('--lr_step'      ,default=10           ,type=int,   help="warm up epochs")
parser.add_argument('--max_lr'      ,default=0.1           ,type=int,   help="top limit of  learing rate")
parser.add_argument('--min_lr'      ,default=1e-5           ,type=int,   help="down limit of  learing rate")


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