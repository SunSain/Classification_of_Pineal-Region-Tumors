# Classification_of_Pineal-Region-Tumors
To detect the Germinoma in the Pineal region by multimodal MRIs and clinical information

[**On 1080Ti**]

DATA PATH:
  t1_train:/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1
  t2_train:/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T2
  t1c_train:/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C
  t1_test:/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1
  t1c_test:/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C
  t2_test:/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T2

Environment:
  conda activate chen_py37

To test Single-modality models:
  run test_new.py
  *Follow the function main() in the file

To test Multi-modality models:
  run cnn_attention_test.py
  *Follow the function main() in the file

The path to pretrained models and their hyperparam files:

  Best T1C_model: /home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_17/model_result/
  Best T1_model: /home/chenxr/Pineal_region/after_12_08/Results/SelfKL/Two/T1C_mask1_composed_RM_1090/model_result/
  Best T2_model: /home/chenxr/Pineal_region/after_12_08/Results/SelfKL/Two/T2_mask1_composed_RM_1090/model_result/
  Best multi_model: /home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_111_CE/model_result/
  *To test more models with different settings( augmentation methods, clinical infor, etc) , please check the dirs list in function comman_drawing() and function special_drawing() in test_new.py file or the file: utils2/static_data.py 
