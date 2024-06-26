
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
from utils2.config import opt
from sklearn.model_selection import train_test_split
import torchio as tio
from utils2.preprocessor import Preprocessor,array2image
from utils2.cube_mask import cube_mask,ratio_cube_mask
from utils2.diy_single_radiomics import DiySingleRadiomics

def read_table(path):
    return(pd.read_excel(path).values) # default to first sheet

def get_ND_bounding_box(label, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = label.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(label)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max


def preprocess_sigle_img(image, threshold=0):
    "standardize voxels with value > threshold"
    image = image.astype(np.float32)
    print("a tra: ",image[:,:,50])
    idx_min,idx_max=get_ND_bounding_box(image)
    mean = np.mean(image)
    std = np.sqrt(image)

    image-=mean
    #image/=std
    return image



class Sub_DiyFolder(torch.utils.data.Dataset):
    def __init__(self, dataset,fileset,aug, aug_form,train_form="", use_radiomics=False,submaskimgs=[]
                 ,use_secondimg = False,noSameRM=False,thirdimg = []
                 ,usethird = False,radiomics_feature=None,multi_modality=False):
        #self.table_refer = read_table(excel_path)
        self.dataset=dataset
        self.files=fileset
        self.aug = aug
        self.aug_form = aug_form
<<<<<<< HEAD
        self.saveroot = opt.save_nii_path
=======
>>>>>>> 3a4a4f2 (20240625-code)
        self.train_form = train_form
        self.radio_mean = None
        self.radio_std = None
        self.use_radiomics = use_radiomics
        self.submaskimgs = submaskimgs
        self.radiomics_feature =radiomics_feature
        self.use_secondimg = use_secondimg
        self.noSameRM = noSameRM
        self.thirdimg = thirdimg
        self.usethird = usethird
        self.multi_modality=multi_modality
<<<<<<< HEAD

        
        if not os.path.exists(self.saveroot):
            os.mkdir(self.saveroot)

=======
        self.saveroot="/home/chenxr/augmented/"
        
>>>>>>> 3a4a4f2 (20240625-code)
    def input_unit_radiomics_mean(self,mean, std):
        self.radio_mean = mean
        self.radio_std = std

    def __len__(self):
        return len(self.dataset)

    def get_self_radiomics_mean(self):
        self.radio_mean

    def __add__(self,data):
        self.dataset.append(data)
        

    def get_all_radiomics(self): # canceled
        for i,(img, sid, slabel,affix,paths) in enumerate(self.dataset):
            imgpath = paths['imgpath']
            maskpath = paths['maskpath']
            features = self.get_radiomics(imgpath,maskpath, i)
            self.radiomics_feature.append(features)
            
    def calc_own_radiomics_mean(self):
        if self.radio_mean !=None:
            return self.radio_mean, self.radio_std
        if not self.use_radiomics:
            return None, None
        
        allfeatures = []
        for i,features in enumerate(self.radiomics_feature):
            allfeatures.append(features)
        allfeatures = torch.stack(allfeatures, 0)
        #print("radiomics_val_box: ",allfeatures)
        self.radio_mean = torch.mean(allfeatures, axis=0)
        self.radio_std = torch.std(allfeatures, axis=0)
        self.normalize_all_features()
        return self.radio_mean, self.radio_std

    def normalize_all_features(self):
        for i,idx in enumerate(self.radiomics_feature):
            features = self.radiomics_feature[i]
            features = self.normalize_radiomics(features)
            self.radiomics_feature[i] = features
        
         
    def inject_other_mean(self,mean, std):
        self.radio_mean = mean
        self.radio_std = std
        self.normalize_all_features()
        
           
    def normalize_radiomics(self,features):
        if self.radio_mean == None:
            print("Haven't set radiomics' mean!")
            self.calc_own_radiomics_mean()
        features = (features-self.radio_mean)/self.radio_std
        return features
            
        
    def get_radiomics(self, imgpath, maskpath, id):
        print("begin to get radiomics,id___",id)
        try:
            
            radio_feature=self.radiomics_feature[id]
        except:
            radiomicsbox = DiySingleRadiomics(imgpath, maskpath)
            radio_feature = radiomicsbox.get_features()
            radio_feature = torch.tensor(radio_feature,dtype=torch.float32)
            if self.radio_mean!=None:
                radio_feature = (radio_feature-self.radio_mean)/self.radio_std       
            else:
                print("Haven't Got Radiomics mean/std!")
            self.radiomics_feature[id]=radio_feature
        return radio_feature
        
<<<<<<< HEAD
        
        
        
    def masked_train_form_augmentation(self, origin_img,id,paths,submask,thirdimg): # masked_training img could also get transformed
        print("transform: Random_Mask_Aug ")
        img,secondimg,thirdimg = self.augment(origin_img, submask,thirdimg,id)
        if not self.use_secondimg:
            secondimg = img
            
        print("img.shape: ",img.shape)
        print("secondimg.shape:",secondimg.shape)
        instance_img = torch.from_numpy(np.append(img,secondimg,axis=0)).type(torch.FloatTensor)
=======
    def masked_train_form_augmentation(self, origin_img,id,paths,submask,thirdimg,attach_thing): # masked_training img could also get transformed
        print("transform: Random_Mask_Aug ")
        org_img,img,secondimg,thirdimg,attachment = self.augment(origin_img, submask,thirdimg,id,attach_thing)
        if not self.use_secondimg:
            secondimg = img
        print("img.shape: ",img.shape)
        print("secondimg.shape:",secondimg.shape)
        
        instance_img = np.append(img,secondimg,axis=0)
        
        if self.multi_modality:

            (t1_img,t2_img,auged_t1_img,auged_t2_img) = attachment
            #img = np.append(img,t1_img,axis=0)
            #img = np.append(img,t2_img,axis=0)
            t1_img = torch.from_numpy(t1_img).type(torch.FloatTensor)
            t2_img = torch.from_numpy(t2_img).type(torch.FloatTensor)
            auged_t1_img = torch.from_numpy(auged_t1_img).type(torch.FloatTensor)
            auged_t2_img = torch.from_numpy(auged_t2_img).type(torch.FloatTensor)
            instance_img = np.append(instance_img,t1_img,axis=0)
            instance_img = np.append(instance_img,t2_img,axis=0)
            
        instance_img = torch.from_numpy(instance_img).type(torch.FloatTensor)
        
>>>>>>> 3a4a4f2 (20240625-code)
        
        #instance_img={0:torch.from_numpy(img), 1:torch.from_numpy(secondimg),2:self.noSameRM}


        mask_instance_img = tio.Lambda(ratio_cube_mask)(instance_img).numpy()
        
        auged_data,auged_second_data = mask_instance_img[0],mask_instance_img[1]
        
<<<<<<< HEAD
        check_img = array2image(auged_data,origin_img)
        save_path = os.path.join("/opt/chenxingru/cube_100/",self.files[id])
        sitk.WriteImage(check_img,save_path)
=======
        #check_img = array2image(auged_data,origin_img)
        #save_path = os.path.join("/opt/chenxingru/cube_100/",self.files[id])
        #sitk.WriteImage(check_img,save_path)
>>>>>>> 3a4a4f2 (20240625-code)
        
        auged_data = np.expand_dims(auged_data, axis=0)
        
        auged_second_data = np.expand_dims(auged_second_data, axis=0)   
            
        if self.use_secondimg:
            img = np.append(img, secondimg,axis=0)
            if not self.noSameRM:
                auged_data = np.append(auged_data, auged_second_data,axis=0)
            else:
                auged_data = np.append(auged_data, secondimg,axis=0)
        if self.usethird:
            img = np.append(img, thirdimg,axis=0)
            auged_data = np.append(auged_data, thirdimg,axis=0)
<<<<<<< HEAD
        
        img = torch.from_numpy(img).type(torch.FloatTensor)
        auged_data = torch.from_numpy(auged_data).type(torch.FloatTensor)

        #print('after img.size: ',img.shape)
        #print('after augedimg.size: ',auged_data.shape)
        #img = sitk.GetArrayFromImage(origin_img).astype(np.float32)

        #print("img_2: ",img.shape)


        #print("auged_data: ",auged_data.shape)
                
        origin_data = img
        #origin_data = np.append(origin_data,submaskimg,axis=0)
        
        #1
        #auged_data,auged_mask_data = tio.Lambda(cube_mask)(img).type(torch.FloatTensor)

        #auged_data = np.append(auged_data,auged_mask_data,axis=0)
=======
            

        
        img = torch.from_numpy(img).type(torch.FloatTensor)
        auged_data = torch.from_numpy(auged_data).type(torch.FloatTensor)           
        origin_data = img
        
        if self.multi_modality:
            #origin_data = np.append(origin_data,t1_img,axis=0)
            #origin_data = np.append(origin_data,t2_img,axis=0)
            
            print('multi_M img.size: ',origin_data.shape)
            
            origin_data_t1 = t1_img
            origin_data_t2 = t2_img
            
            auged_data_t1, auged_data_t2 = mask_instance_img[2],mask_instance_img[3]
            auged_data_t1 = np.expand_dims(auged_data_t1, axis=0)
            auged_data_t2 = np.expand_dims(auged_data_t2, axis=0)
            
            auged_data = np.append(auged_data,auged_data_t1,axis=0)
            auged_data = np.append(auged_data,auged_data_t2,axis=0)
            
            auged_data_t1 = torch.from_numpy(auged_data_t1).type(torch.FloatTensor) 
            auged_data_t2 = torch.from_numpy(auged_data_t2).type(torch.FloatTensor) 
        else:
            origin_data_t1 ,origin_data_t2,auged_data_t1,auged_data_t2 = None,None,None,None
>>>>>>> 3a4a4f2 (20240625-code)

        print("begin")
        if self.use_radiomics:
            if self.radiomics_feature!=None:
                print("id___",id)
                sradiomics = self.radiomics_feature[id]
            else:
                sradiomics = self.get_radiomics(paths['imgpath'],paths['maskpath'],id)
        else:
            print("no radiom")
            sradiomics=torch.tensor([0.])
            self.radiomics_feature.append(sradiomics)
        print("it's back!")
<<<<<<< HEAD
        return origin_data,auged_data,sradiomics

    def normal_train_form(self, origin_img, id,paths,submaskimg,thirdimg):
        img,secondimg,thirdimg= self.augment(origin_img,submaskimg, thirdimg,id)
=======
        
        return origin_data,auged_data,sradiomics,(origin_data_t1,auged_data_t1,origin_data_t2,auged_data_t2)

    def normal_train_form(self, origin_img, id,paths,submaskimg,thirdimg,attach_thing):
        org_img,img,secondimg,thirdimg, attachment= self.augment(origin_img,submaskimg, thirdimg,id,attach_thing)
>>>>>>> 3a4a4f2 (20240625-code)
        if self.use_secondimg:
            img = np.append(img, secondimg,axis=0)
            print("img.shape: ",img.shape)
        if self.usethird:
            img = np.append(img, thirdimg,axis=0)
        print('after img.size: ',img.shape)
<<<<<<< HEAD
        
        img = torch.from_numpy(img).type(torch.FloatTensor)
=======
        if self.multi_modality:
            (t1_img,t2_img,auged_t1_img,auged_t2_img) = attachment
            #img = np.append(img,t1_img,axis=0)
            #img = np.append(img,t2_img,axis=0)
            t1_img = torch.from_numpy(t1_img).type(torch.FloatTensor)
            t2_img = torch.from_numpy(t2_img).type(torch.FloatTensor)
            auged_t1_img = torch.from_numpy(auged_t1_img).type(torch.FloatTensor)
            auged_t2_img = torch.from_numpy(auged_t2_img).type(torch.FloatTensor)
            print('multi_M img.size: ',img.shape)
        else:
            t1_img,t2_img,auged_t1_img,auged_t2_img = None, None,None, None
            
        img = torch.from_numpy(img).type(torch.FloatTensor)
        org_img = torch.from_numpy(org_img).type(torch.FloatTensor)
>>>>>>> 3a4a4f2 (20240625-code)
        
        if self.use_radiomics:
            if self.radiomics_feature!=None:
                sradiomics = self.radiomics_feature[id]
            else:
                sradiomics = self.get_radiomics(paths['imgpath'],paths['maskpath'],id)
        else:
            sradiomics=torch.tensor([0.])
            self.radiomics_feature.append(sradiomics)
<<<<<<< HEAD
        return img, sradiomics
    
    #数据增强(aug=True). 如果aug =False则只是从原来的 NII_img拿出img.get_data(),作为输入模型的数据
    def augment(self, origin_img,submaskimg,thirdsubimg,id): # return array,[1,23,400,400] ,[1,23,400,400]
        
=======
        return org_img,img, sradiomics, (t1_img,t2_img,auged_t1_img,auged_t2_img)
    
    #数据增强(aug=True). 如果aug =False则只是从原来的 NII_img拿出img.get_data(),作为输入模型的数据
    def augment(self, origin_img,submaskimg,thirdsubimg,id, attachment=None): # return array,[1,23,400,400] ,[1,23,400,400]
        # ori_img is must be T1C
>>>>>>> 3a4a4f2 (20240625-code)
        maskimg = None
        third_maskimg=None
        composed_RM = False
        tradition_token = self.aug_form
        img = origin_img
        secondimg = submaskimg
        thirdimg = thirdsubimg
<<<<<<< HEAD
        
        if self.aug_form == "Composed_RM":
=======
        t1_img, t2_img,img_t1,img_t2 = None, None,None,None
        orig_t1, orig_t2,ori_t1c = None,None,None
        if self.multi_modality:
            (t1_img, t2_img ) = attachment
            
        if self.aug_form == "Composed_RM" or self.aug_form == "composed_RM":
>>>>>>> 3a4a4f2 (20240625-code)
            composed_RM=True
            tradition_token = "Composed"
        if self.train_form == "masked_constrained_train":
            composed_RM = False # just leave composed, RM will be made afterwhile        
        if self.aug== False or self.aug_form=="" or self.aug_form=="None" or self.aug_form=="none":
            img = sitk.GetArrayFromImage(origin_img).astype(np.float32)
            img = np.expand_dims(img, axis=0)
<<<<<<< HEAD
            img = np.ascontiguousarray(img, dtype= np.float32)    
=======
            img = np.ascontiguousarray(img, dtype= np.float32)   
            if  attachment!=None:
                img_t1 = sitk.GetArrayFromImage(t1_img).astype(np.float32)
                img_t1 = np.expand_dims(img_t1, axis=0)
                img_t1 = np.ascontiguousarray(img_t1, dtype= np.float32)
                orig_t1 = img_t1 
                
                img_t2 = sitk.GetArrayFromImage(t2_img).astype(np.float32)
                img_t2 = np.expand_dims(img_t2, axis=0)
                img_t2 = np.ascontiguousarray(img_t2, dtype= np.float32) 
                orig_t2 = img_t2
>>>>>>> 3a4a4f2 (20240625-code)
            #print('img: ',img)
            #print('img.size: ',img.shape)
            if self.use_secondimg:
                maskimg = sitk.GetArrayFromImage(submaskimg).astype(np.float32)
                maskimg = np.expand_dims(maskimg, axis=0)
                maskimg = np.ascontiguousarray(maskimg, dtype= np.float32)
            if self.usethird:
                third_maskimg = sitk.GetArrayFromImage(thirdimg).astype(np.float32)
                third_maskimg = np.expand_dims(third_maskimg, axis=0)
                third_maskimg = np.ascontiguousarray(third_maskimg, dtype= np.float32)
                #img = np.append(img, maskimg,axis=0)
            #img.append(maskimg[0])
            #print('after img.size: ',img.shape)
            #img = torch.from_numpy(img).type(torch.FloatTensor)
            
            
<<<<<<< HEAD
            return img, maskimg,third_maskimg
=======
            return img, img,maskimg,third_maskimg ,(img_t1,img_t2,img_t1,img_t2)#attachment
>>>>>>> 3a4a4f2 (20240625-code)
        
        if not self.aug_form == "Random_Mask": #not zero, not just RM, then must be traditional aug or aug_RM
            transform_dict={
                tio.RandomAffine(degrees=15),
                tio.RandomFlip(flip_probability=0.5, axes=('LR'))
                }
            trans_dict = {
                    'Affine':tio.RandomAffine(degrees=15),
                    'Flip':tio.RandomFlip(flip_probability=0.5, axes=('LR')),
                    'Composed':tio.Compose(transform_dict)
                    }

            transform = trans_dict[tradition_token]
            print("transform: ",transform)
<<<<<<< HEAD

=======
            org_img = sitk.GetArrayFromImage(origin_img).astype(np.float32)
            org_img = np.expand_dims(org_img, axis=0)
            org_img = np.ascontiguousarray(org_img, dtype= np.float32)   
            
>>>>>>> 3a4a4f2 (20240625-code)
            image = transform(img) 
            img = sitk.GetArrayFromImage(image).astype(np.float32)
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img, dtype= np.float32)
            #img = torch.from_numpy(img).type(torch.FloatTensor)
            auged_img = img 
<<<<<<< HEAD
=======
            
            if  attachment!=None:
                org_img_t1 = sitk.GetArrayFromImage(t1_img).astype(np.float32)
                org_img_t1 = np.expand_dims(org_img_t1, axis=0)
                org_img_t1 = np.ascontiguousarray(org_img_t1, dtype= np.float32)
                orig_t1 = org_img_t1
                org_img_t2 = sitk.GetArrayFromImage(t2_img).astype(np.float32)
                org_img_t2 = np.expand_dims(org_img_t2, axis=0)
                org_img_t2 = np.ascontiguousarray(org_img_t2, dtype= np.float32)
                orig_t2 = org_img_t2              
                
                img_t1 = transform(t1_img)
                img_t1 = sitk.GetArrayFromImage(img_t1).astype(np.float32)
                img_t1 = np.expand_dims(img_t1, axis=0)
                img_t1 = np.ascontiguousarray(img_t1, dtype= np.float32) 
                
                img_t2 = transform(t2_img)
                img_t2 = sitk.GetArrayFromImage(img_t2).astype(np.float32)
                img_t2 = np.expand_dims(img_t2, axis=0)
                img_t2 = np.ascontiguousarray(img_t2, dtype= np.float32) 
                auged_img_t1,auged_img_t2 = img_t1,img_t2
            else:
                auged_img_t1,auged_img_t2  = None, None
            
>>>>>>> 3a4a4f2 (20240625-code)
            if self.use_secondimg:
                maskimg = transform(secondimg)
                maskimg = sitk.GetArrayFromImage(maskimg).astype(np.float32)
                maskimg = np.expand_dims(maskimg, axis=0)
                maskimg = np.ascontiguousarray(maskimg, dtype= np.float32)
            if self.usethird:
                third_maskimg = transform(thirdimg)
                third_maskimg = sitk.GetArrayFromImage(third_maskimg).astype(np.float32)
                third_maskimg = np.expand_dims(third_maskimg, axis=0)
                third_maskimg = np.ascontiguousarray(third_maskimg, dtype= np.float32)
                #maskimg = torch.from_numpy(maskimg).type(torch.FloatTensor)
<<<<<<< HEAD
            

            #auged_img = np.append(img,submaskimage,axis=0)
            """
            img = array2image(auged_img[0],origin_img)
            save_path = os.path.join(self.saveroot,self.files[id])
            sitk.WriteImage(img,save_path)
            """
            if not composed_RM:
                return auged_img,maskimg,third_maskimg
            else:
                img = array2image(auged_img[0],origin_img)
                secondimg = array2image(maskimg[0],submaskimg)
                thirdimg = array2image(third_maskimg[0],thirdimg)
=======
            #auged_img = np.append(img,submaskimage,axis=0)
            img = array2image(auged_img[0],origin_img)
            save_path = os.path.join(self.saveroot,self.files[id])
            sitk.WriteImage(img,save_path)
            
            if not composed_RM:
                return org_img,auged_img,maskimg,third_maskimg,(orig_t1,orig_t2,auged_img_t1,auged_img_t2)
            else:
                img = array2image(auged_img[0],origin_img)
                if self.use_secondimg:
                    secondimg = array2image(maskimg[0],submaskimg)
                if self.usethird:
                    thirdimg = array2image(third_maskimg[0],thirdimg)
                if  attachment!=None:
                    img_t1 = array2image(auged_img_t1[0],t1_img)
                    img_t2 = array2image(auged_img_t2[0],t2_img)
                else:
                    img_t1,img_t2 =img, img
>>>>>>> 3a4a4f2 (20240625-code)
                
        

        if self.aug_form == "Random_Mask" or composed_RM == True : # constrained_loss_part don't need RM, only RM or aug_RM(and normal_train)
            print("transform: Random_Mask_Aug ")
            img = sitk.GetArrayFromImage(img).astype(np.float32)
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img, dtype= np.float32)
<<<<<<< HEAD
            #img = torch.from_numpy(img).type(torch.FloatTensor)
=======
            org_img = img
            #img = torch.from_numpy(img).type(torch.FloatTensor)
            if  attachment!=None:
                img_t1 = sitk.GetArrayFromImage(img_t1).astype(np.float32)
                img_t1 = np.expand_dims(img_t1, axis=0)
                img_t1 = np.ascontiguousarray(img_t1, dtype= np.float32) 
                
                img_t2 = sitk.GetArrayFromImage(img_t2).astype(np.float32)
                img_t2 = np.expand_dims(img_t2, axis=0)
                img_t2 = np.ascontiguousarray(img_t2, dtype= np.float32) 
                orig_t1 = img_t1 
                orig_t2 = img_t2 
                
>>>>>>> 3a4a4f2 (20240625-code)
            if self.use_secondimg:
                maskimg = sitk.GetArrayFromImage(secondimg).astype(np.float32)
                maskimg = np.expand_dims(maskimg, axis=0)
                maskimg = np.ascontiguousarray(maskimg, dtype= np.float32)
            else:
                maskimg = img
            if self.usethird:
                third_maskimg = sitk.GetArrayFromImage(thirdimg).astype(np.float32)
                third_maskimg = np.expand_dims(third_maskimg, axis=0)
                third_maskimg = np.ascontiguousarray(third_maskimg, dtype= np.float32)
            else:
                third_maskimg = img
                
            instance_img = np.append(img,maskimg,axis=0)
<<<<<<< HEAD
=======
            if  attachment!=None:
                instance_img = np.append(instance_img,img_t1,axis=0)
                instance_img = np.append(instance_img,img_t2,axis=0)
                
>>>>>>> 3a4a4f2 (20240625-code)
            print("instance_img.shape: ",instance_img.shape)
            instance_img = torch.from_numpy(instance_img).type(torch.FloatTensor)
            #instance_img={"img":torch.from_numpy(img), "secondimg":torch.from_numpy(maskimg),"noSameRM":self.noSameRM}
            #instance_img={0:torch.from_numpy(img), 1:torch.from_numpy(secondimg),2:self.noSameRM}
            #print("instance_img: ",instance_img)
            mask_instance_img = tio.Lambda(ratio_cube_mask)(instance_img).numpy()
<<<<<<< HEAD
            auged_img,auged_maskimg = mask_instance_img[0],mask_instance_img[1]
            auged_img = np.expand_dims(auged_img, axis=0)
            auged_maskimg = np.expand_dims(auged_maskimg, axis=0)   
          
            #1
            #auged_img,auged_mask_img = tio.Lambda(cube_mask)(img).type(torch.FloatTensor)
            #auged_img = np.append(auged_img,auged_mask_img,axis=0)           
            """
            # just for record affix aug_im into files
            img = array2image(auged_img[0],origin_img)
            save_path = os.path.join(self.saveroot,self.files[id])
            sitk.WriteImage(img,save_path)
           
            if self.aug_form == "Random_Mask":
                return auged_img,auged_maskimg
            else:
                print("auged_img.shape: ",auged_img.shape)
                img = array2image(auged_img[0],origin_img)
                secondimg = array2image(auged_maskimg[0],submaskimg)
            """
            if not self.noSameRM:
                return auged_img,auged_maskimg,third_maskimg
            else:
                return auged_img,maskimg,third_maskimg
        
        """
        

        transform_dict={
            tio.RandomAffine(degrees=15),
            tio.RandomFlip(flip_probability=0.5, axes=('LR'))
            }
        trans_dict = {
                'Affine':tio.RandomAffine(degrees=15),
                'Flip':tio.RandomFlip(flip_probability=0.5, axes=('LR')),
                'Composed':tio.Compose(transform_dict)
                }

        transform = trans_dict[tradition_token]
        print("transform: ",transform)

        image = transform(img) 
        img = sitk.GetArrayFromImage(image).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img, dtype= np.float32)
        #img = torch.from_numpy(img).type(torch.FloatTensor)
        auged_img = img 
        if self.use_secondimg:
            maskimg = transform(secondimg)
            maskimg = sitk.GetArrayFromImage(maskimg).astype(np.float32)
            maskimg = np.expand_dims(maskimg, axis=0)
            maskimg = np.ascontiguousarray(maskimg, dtype= np.float32)
            #maskimg = torch.from_numpy(maskimg).type(torch.FloatTensor)
        

        #auged_img = np.append(img,submaskimage,axis=0)
       
        #img = array2image(auged_img[0],origin_img)
        #save_path = os.path.join(self.saveroot,self.files[id])
        #sitk.WriteImage(img,save_path)
       

        return auged_img,maskimg
        """
=======
            auged_img,auged_maskimg = img[0],mask_instance_img[1]
            auged_img = np.expand_dims(auged_img, axis=0)
            auged_maskimg = np.expand_dims(auged_maskimg, axis=0)   
            if  attachment!=None:
                auged_img_t1,auged_img_t2 = mask_instance_img[2],mask_instance_img[3]
                auged_img_t1 = np.expand_dims(auged_img_t1, axis=0)
                auged_img_t2 = np.expand_dims(auged_img_t2, axis=0)  

            else:
                auged_img_t1,auged_img_t2 = None, None
            if not self.noSameRM:
                return auged_img,auged_maskimg,secondimg,third_maskimg,(orig_t1,orig_t2,auged_img_t1,auged_img_t2)
            else:
                return auged_img,maskimg,secondimg,third_maskimg,(orig_t1,orig_t2,auged_img_t1,auged_img_t2)
>>>>>>> 3a4a4f2 (20240625-code)

    def __getitem__(self,index):

        if not self.multi_modality:
            (img, sid, slabel,affix,paths) = self.dataset[index] #paths: {'imgpath':, 'maskpath': , 'bbx_path':,...},file_path, seg_file_path, bbx_path...
<<<<<<< HEAD
        else:
            (T1C_img, sid, slabel, inforbox,paths,T1_img,T2_img) =   self.dataset[index]  
=======
            attach_thing = None
        else:
            (t1_img,t2_img,t1c_img, sid, slabel, affix,paths) =   self.dataset[index]  
            attach_thing = (t1_img,t2_img)
>>>>>>> 3a4a4f2 (20240625-code)
        #print("1item: ",img, sid, slabel,affix,paths)
        
        submaskimg = self.submaskimgs[index]
        #print("2item: ",img, sid, slabel,affix,paths)
        thirdimg = self.thirdimg[index]
        saffix = torch.tensor(affix,dtype=torch.float32)
        
        #print("paths: ",paths)  
        
        if not self.multi_modality:
            if self.train_form == "masked_constrained_train":
<<<<<<< HEAD
                data, auged_data,sradiomics = self.masked_train_form_augmentation(origin_img=img, id=index,paths=paths,submask=submaskimg,thirdimg = thirdimg)

                return (data,auged_data,sid,slabel,saffix,sradiomics)
            else:
                auged_data,sradiomics = self.normal_train_form(img,index,paths,submaskimg, thirdimg = thirdimg)
                return (auged_data, sid, slabel, saffix,sradiomics)
        else:
            if self.train_form == "masked_constrained_train":
                T1C_data, T1C_auged_data,sradiomics = self.masked_train_form_augmentation(origin_img=T1C_img, id=index,paths=paths,submask=submaskimg,thirdimg = thirdimg)
                T1_data, T1_auged_data,sradiomics = self.masked_train_form_augmentation(origin_img=T1_img, id=index,paths=paths,submask=submaskimg,thirdimg = thirdimg)
                T2_data, T2_auged_data,sradiomics = self.masked_train_form_augmentation(origin_img=T2_img, id=index,paths=paths,submask=submaskimg,thirdimg = thirdimg)
                return (T1_data,T1_auged_data,T2_data,T2_auged_data,T1C_data,T1C_auged_data,sid,slabel,saffix,sradiomics)
            else:
                T1C_auged_data,sradiomics = self.normal_train_form(T1C_img,index,paths,submaskimg, thirdimg = thirdimg)
                T1_auged_data,sradiomics = self.normal_train_form(T1_img,index,paths,submaskimg, thirdimg = thirdimg)
                T2_auged_data,sradiomics = self.normal_train_form(T2_img,index,paths,submaskimg, thirdimg = thirdimg)
                return (T1_auged_data,T2_auged_data,T1C_auged_data, sid, slabel, saffix,sradiomics)
=======
                data, auged_data,sradiomics,item = self.masked_train_form_augmentation(origin_img=img, id=index,paths=paths,submask=submaskimg,thirdimg = thirdimg,attach_thing = attach_thing)

                return (data,auged_data,sid,slabel,saffix,sradiomics)
            else:
                data,auged_data,sradiomics,item  = self.normal_train_form(img,index,paths,submaskimg, thirdimg = thirdimg,attach_thing = attach_thing)
                return (data,auged_data, sid, slabel, saffix,sradiomics)
        else:
            if self.train_form == "masked_constrained_train":
                T1C_data, T1C_auged_data,sradiomics,item= self.masked_train_form_augmentation(origin_img=t1c_img, id=index,paths=paths,submask=submaskimg,thirdimg = thirdimg,attach_thing = attach_thing)
                (T1_data,T2_data,T1_auged_data,T2_auged_data)  = item
                #T1_data, T1_auged_data,sradiomics = self.masked_train_form_augmentation(origin_img=t1_img, id=index,paths=paths,submask=submaskimg,thirdimg = thirdimg)
                #T2_data, T2_auged_data,sradiomics = self.masked_train_form_augmentation(origin_img=t2_img, id=index,paths=paths,submask=submaskimg,thirdimg = thirdimg)
                return (T1C_data,T1C_auged_data,T1_data,T1_auged_data,T2_data,T2_auged_data,sid,slabel,saffix,sradiomics)
            else:
                T1C_data,T1C_auged_data,sradiomics,item = self.normal_train_form(t1c_img,index,paths,submaskimg, thirdimg = thirdimg,attach_thing = attach_thing)
                (T1_data,T2_data,T1_auged_data,T2_auged_data) = item
                #T1_auged_data,sradiomics = self.normal_train_form(t1_img,index,paths,submaskimg, thirdimg = thirdimg)
                #T2_auged_data,sradiomics = self.normal_train_form(t2_img,index,paths,submaskimg, thirdimg = thirdimg)
                return (T1C_data,T1C_auged_data,T1_data,T1_auged_data,T2_data,T2_auged_data,sid,slabel,saffix,sradiomics)
>>>>>>> 3a4a4f2 (20240625-code)
            
    def gety(self):
        return self.y



<<<<<<< HEAD



=======
def augment( origin_img,aug_form,saveroot,filename,aug=False): # return array,[1,23,400,400] ,[1,23,400,400]
        # ori_img is must be T1C

        composed_RM = False
        tradition_token = aug_form
        img = origin_img

        
        if self.aug== False or self.aug_form=="" or self.aug_form=="None" or self.aug_form=="none":
            img = sitk.GetArrayFromImage(origin_img).astype(np.float32)
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img, dtype= np.float32)   
            
            
            return img, img,maskimg,third_maskimg ,(img,img,img,img)#attachment
        
        if not aug_form == "Random_Mask": #not zero, not just RM, then must be traditional aug or aug_RM
            transform_dict={
                tio.RandomAffine(degrees=15),
                tio.RandomFlip(flip_probability=0.5, axes=('LR'))
                }
            trans_dict = {
                    'Affine':tio.RandomAffine(degrees=15),
                    'Flip':tio.RandomFlip(flip_probability=0.5, axes=('LR')),
                    'Composed':tio.Compose(transform_dict)
                    }
            transform = trans_dict[tradition_token]
            print("transform: ",transform)
            org_img = sitk.GetArrayFromImage(origin_img).astype(np.float32)
            org_img = np.expand_dims(org_img, axis=0)
            org_img = np.ascontiguousarray(org_img, dtype= np.float32)   
            
            image = transform(img) 
            img = sitk.GetArrayFromImage(image).astype(np.float32)
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img, dtype= np.float32)
            #img = torch.from_numpy(img).type(torch.FloatTensor)
            auged_img = img 
            
            img = array2image(auged_img[0],origin_img)
            save_path = os.path.join(saveroot,filename)
            sitk.WriteImage(img,save_path)
            
            if not composed_RM:
                return 
            else:
                img = array2image(auged_img[0],origin_img)
                if use_secondimg:
                    secondimg = array2image(maskimg[0],submaskimg)
                if usethird:
                    thirdimg = array2image(third_maskimg[0],thirdimg)
                if  attachment!=None:
                    img_t1 = array2image(auged_img_t1[0],img_t1)
                    img_t2 = array2image(auged_img_t2[0],img_t2)
                else:
                    img_t1,img_t2 =img, img
                
        

        if aug_form == "Random_Mask" or composed_RM == True : # constrained_loss_part don't need RM, only RM or aug_RM(and normal_train)
            print("transform: Random_Mask_Aug ")
            img = sitk.GetArrayFromImage(img).astype(np.float32)
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img, dtype= np.float32)
            org_img = img
            #img = torch.from_numpy(img).type(torch.FloatTensor)

            maskimg=img
            instance_img = np.append(img,maskimg,axis=0)
                
            print("instance_img.shape: ",instance_img.shape)
            instance_img = torch.from_numpy(instance_img).type(torch.FloatTensor)
            #instance_img={"img":torch.from_numpy(img), "secondimg":torch.from_numpy(maskimg),"noSameRM":noSameRM}
            #instance_img={0:torch.from_numpy(img), 1:torch.from_numpy(secondimg),2:noSameRM}
            #print("instance_img: ",instance_img)
            mask_instance_img = tio.Lambda(ratio_cube_mask)(instance_img).numpy()
            auged_img,auged_maskimg = img[0],mask_instance_img[1]
            auged_img = np.expand_dims(auged_img, axis=0)
            auged_maskimg = np.expand_dims(auged_maskimg, axis=0)   
            
            img = array2image(auged_img[0],origin_img)
            save_path = os.path.join(saveroot,filename)
            sitk.WriteImage(img,save_path)
            
            return auged_img[0]



if __name__=="__main__":
    t1c_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C/034_7_4A2FBEA7174F4ED6BE896ADB8D202353_t1_tirm_tra_dark-fluid_Turn_0000.nii.gz"
    preprocessor = Preprocessor(target_spacing=[0.45,0.45,6.5])
    origin_img, brain_mask_img = preprocessor.run(img_path=t1c_path)
    origin_img=sitk.ReadImage(t1c_path)
    aug_form=""
    img, img,maskimg,third_maskimg ,(img,img,img,img)=augment( origin_img,aug_form,saveroot,filename)
    print("img.shape: ",img.shape)
    concatenated_tensor = torch.cat([img, img, img], dim=0)
    print("concatenated_tensor.shape: ",concatenated_tensor.shape)
    
    


    

>>>>>>> 3a4a4f2 (20240625-code)
