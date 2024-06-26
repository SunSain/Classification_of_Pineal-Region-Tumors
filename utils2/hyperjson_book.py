

import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import SimpleITK as sitk
from scipy.ndimage import rotate
import nibabel as nib
import numpy as np
import os


class HyperJson_book(torch.utils.data.Dataset):
    def __init__(self, hyf):
        self.output_dir =hyf["output_dir"]
        self.data_path =hyf["data_path"]
        self.testdata_path=hyf["testdata_path"]
        self.num_workers=hyf["num_workers"]
        self.multi_m=hyf["multi_m"]
        self.train_form=hyf["train_form"]
        self.t1_path=hyf["t1_path"]
        self.t2_path=hyf["t2_path"]
        self.t1c_path=hyf["t1c_path"]
        self.root_bbx_path=hyf["root_bbx_path"]
        self.use_radiomics=hyf["use_radiomics"]
        self.sec_pair_orig=hyf["sec_pair_orig"]
        self.multiclass=hyf["multiclass"]
        self.usesecond=hyf["usesecond"]
        self.noSameRM=hyf["noSameRM"]
        self.usethird=hyf["usethird"]
        self.batch_size=hyf["batch_size"]
        self.epochs=hyf["epochs"]
        self.usesecond=hyf["usesecond"]
        self.num_classes=hyf["num_classes"]
        self.loss_weight=hyf["loss_weight"]
        self.lambda_0=hyf["lambda_0"]
        self.lambda_1=hyf["lambda_1"]
        self.lambda_2=hyf["lambda_2"]
        self.FL_gamma=hyf["FL_gamma"]
        self.CE_or_KL=hyf["CE_or_KL"]
        self.lossfunc=hyf["lossfunc"]
        self.kfold=hyf["kfold"]
        self.random_seed=hyf["random_seed"]
        self.model=hyf["model"]
        self.test_root_bbx_path = hyf["test_root_bbx_path"]
        self.aug_form = hyf["aug_form"]
        self.constrain_lambd=hyf["constrain_lambd"]
        self.print_freq=hyf["print_freq"]