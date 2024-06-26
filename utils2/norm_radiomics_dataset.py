
import numpy as np
import torch
class Unite_norm_radiomics():
    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test
        self.unit_mean = None
        self.unit_std = None
        self.get_unit_mean()
    
    def norm(self):
        train, valid, test  = self.unit_dataset(self.train),self.unit_dataset(self.valid),self.unit_dataset(self.test)
        return train, valid, test

    def get_unit_radiomics_mean(self):
        return self.unit_mean, self.unit_std
    
    def get_unit_mean(self):
        radiomics_val_box=[]
        for i, (img,sid,target, saffix, sradiomics) in enumerate(self.train):
            print("train_sradiomics: ",sradiomics)
            radiomics_val_box.append(sradiomics)
            #radiomics_val_box.append(sradiomics)
        radiomics_val_box = torch.stack(radiomics_val_box, 0)
        print("radiomics_val_box: ",radiomics_val_box)
        self.unit_mean = torch.mean(radiomics_val_box, axis=0)
        self.unit_std = torch.std(radiomics_val_box, axis=0)
            
    def unit_dataset(self,box):
        for i, (img,sid,target, saffix, sradiomics) in enumerate(box):
            sradiomics = (sradiomics-self.unit_mean)/self.unit_std
            box[i] = (img,sid,target, saffix, sradiomics)
        return box

            
            
           