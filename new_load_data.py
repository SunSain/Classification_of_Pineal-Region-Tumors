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
<<<<<<< HEAD

=======
from utils2.preprocessor import Preprocessor
>>>>>>> 3a4a4f2 (20240625-code)

def nii_loader(path):
    img = nib.load(str(path))
    return img

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



class DIY_Folder(torch.utils.data.Dataset):
    def __init__(self, data_path, loader=nii_loader,transform_dict=None):
        self.root = data_path
        self.sub_fns = sorted(os.listdir(self.root))
        #self.table_refer = read_table(excel_path)
        self.loader = loader
        self.transform_dict = transform_dict
        self.dataset=[]
        self.stastic = [0,0,0]
        self.stastic2=[0,0,0]
        self.origi=[0,0,0]
        self.count=0
        self.y=[]
        self.pre_aug() 


<<<<<<< HEAD
=======

    def preprocess(self):
        preprocessor = Preprocessor()
        
>>>>>>> 3a4a4f2 (20240625-code)
    # 预处理(只是导入数据、归一化，没有数据增强)
    def pre_aug(self):
        for i, f in enumerate(self.sub_fns):
            fn = f.split("_")
            #get label: 0<sid<61: label =0 | 61<sid: label=1
            try:
                    sid=int(fn[0])
            except:
                print("it's the_select_file")
            self.count+=1
            if sid<61: 
                slabel=0
                self.y.append(0)
                self.stastic[1]+=1
            else: 
                slabel =1
                self.y.append(1)
                self.stastic[2]+=1
            #print("sid: ",sid," ;slabel: ",slabel)

            smale = 0
            sub_path = os.path.join(self.root, f)

            img = self.loader(sub_path)
            #——————————crop and Normalization————————
<<<<<<< HEAD
            print("img.shape: ",img.get_data().shape)
            tiocrop=tio.transforms.CropOrPad([75,85,75],  'linear')
            img=tiocrop(img)
=======
            print("sid: ",sid,"img.shape: ",img.get_data().shape)
            #tiocrop=tio.transforms.CropOrPad([75,85,75],  'linear')
            #img=tiocrop(img)
>>>>>>> 3a4a4f2 (20240625-code)
            Ztransform= tio.ZNormalization()
            img = Ztransform(img)
            #——————————————————
            original_data = (img, sid, slabel, 0)
            self.__add__(original_data)
        

    def __len__(self):
        return len(self.dataset)


    def __add__(self,other):
        self.dataset.append(other)
<<<<<<< HEAD
=======
        
    def _pad_or_crop_to_img_size(self, image, img_size, mode="constant"):
        """Image cropping to img_size
        """
        rank = len(img_size)
        
        # Create placeholders for the new shape
        from_indices = [[0, image.shape[dim]] for dim in range(rank)]  # [ [0, 0], [0, 1], [0, 2] ]
        to_padding = [[0, 0] for dim in range(rank)]
        
        slicer = [slice(None)] * rank
        
        # for each dimensions find whether it is supposed to be cropped or padded
        for i in range(rank):
            if image.shape[i] <= img_size[i]:
                to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
                to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
            else:
                slice_start = np.random.randint(0, image.shape[i] - img_size[i] + 1)
                from_indices[i][0] = slice_start
                from_indices[i][1] = from_indices[i][0] + img_size[i]
            
            # Create slicer object to crop or leach each dimension
            slicer[i] = slice(from_indices[i][0], from_indices[i][1])
        
        padded_img = np.pad(image[tuple(slicer)], to_padding, mode=mode, constant_values=0)
        #padded_seg = np.pad(seg[tuple(slicer)], to_padding, mode=mode, constant_values=0)
        
        return {
            "padded_img": padded_img
        }
>>>>>>> 3a4a4f2 (20240625-code)

    #数据增强(aug=True)， 如果aug =False则只是从原来的 NII_img拿出img.get_data(),作为输入模型的数据
    def prepro_aug(self, data_idx=[], aug=True, transform_dict=None):
        subdata=[]
        for i, idx in enumerate(data_idx):
            (img,sid,slabel,_) = self.dataset[idx]
            a1 = (img.get_data(), sid, slabel, 1)
            subdata.append(a1)
        suby=[self.y[i] for i in data_idx] #只是统计img的label

        if aug== False or transform_dict ==None:
            return subdata
        #三次数据增强
        for j, idx in enumerate(data_idx):
            (img,sid,slabel,_) = self.dataset[idx]

            transform = tio.Compose(transform_dict)
            image = transform(img)
            a1 = (image.get_data(), sid, slabel, 1)
            subdata.append(a1)
            suby.append(slabel) 

            transform2 = tio.Compose(transform_dict)
            image = transform2(img)
            a2 = (image.get_data(), sid, slabel, 2)        
            subdata.append(a2)
            suby.append(slabel)

            transform3 = tio.Compose(transform_dict)
            image = transform3(img)
            a3 = (image.get_data(), sid, slabel, 3)        
            subdata.append(a3)
            suby.append(slabel)
        return subdata

    def __getitem__(self,index):
        data=self.dataset[index]
<<<<<<< HEAD
=======
        
        padded_dict = self._pad_or_crop_to_img_size(data, self.img_size)
        
        data = padded_dict["padded_img"]
        
>>>>>>> 3a4a4f2 (20240625-code)
        return data

    def gety(self):
        return self.y





<<<<<<< HEAD
=======
"""


class DIY_Folder(torch.utils.data.Dataset):
    def __init__(self, data_path, loader=nii_loader,transform_dict=None):
        self.root = data_path
        self.sub_fns = sorted(os.listdir(self.root))
        #self.table_refer = read_table(excel_path)
        self.loader = loader
        self.transform_dict = transform_dict
        self.dataset=[]
        self.stastic = [0,0,0]
        self.stastic2=[0,0,0]
        self.origi=[0,0,0]
        self.trans1={
        tio.RandomElasticDeformation(),
        tio.RandomNoise()
        
        }
        self.trans3={        
        tio.RandomAffine(),
        tio.RandomElasticDeformation()}

        self.trans2={
            tio.RandomFlip(),
            tio.RandomBlur()
        }
        self.trans4={
        tio.RandomElasticDeformation(),
        tio.RandomNoise(),
        tio.RandomAffine()
        }
        self.trans5={
            tio.RandomElasticDeformation(),
            tio.RandomFlip(),
            tio.RandomBlur()
        }

        self.pre_aug() 



    def pre_aug(self):
        for i, f in enumerate(self.sub_fns):
            fn = f.split("_")
            #print("fn: ",fn)
            try:
                    sid=int(fn[0])
            except:
                print("it's the_select_file")
            
            if sid<27: 
                slabel=0
                self.stastic[0]+=1
                self.stastic2[0]+=1
                self.origi[0]+=1
            elif sid<61: 
                slabel=1
                self.stastic[1]+=1
                self.stastic2[1]+=1
                self.origi[1]+=1
            else: 
                slabel =2
                self.stastic[2]+=1
                self.stastic2[2]+=1
                self.origi[2]+=1
            #print("sid: ",sid," ;slabel: ",slabel)

            smale = f[2]
            sub_path = os.path.join(self.root, f)
            img = self.loader(sub_path)
            #——————————Normalization————————
            Ztransform= tio.ZNormalization()
            img = Ztransform(img)
            #——————————————————
            original_data = (img.get_data(), sid, slabel, 0)
            self.__add__(original_data)
        
            print("lenth1: ",len(self.dataset))
            if self.transform_dict !=None:  
                self.aug(sid,slabel,img)
            
            #else:
                #self.__add__(original_data)  
            
        print("orginal_count: ",self.origi) 


    def __len__(self):
        return len(self.dataset)


    def __add__(self,other):
        self.dataset.append(other)

    def aug(self, sid,slabel, img):
        if sid<27:
            trans_1 = tio.transforms.OneOf(self.trans1)
            trans_2 = tio.transforms.OneOf(self.trans2)
            trans_3 = tio.transforms.OneOf(self.trans3)

            img1 =trans_1(img)
            img2 =trans_2(img)
            img3 =trans_3(img)  
            a1 = (img1.get_data(), sid, slabel, 1)              
            a2 = (img2.get_data(), sid, slabel, 2) 
            a3 = (img3.get_data(), sid, slabel, 3) 
            self.__add__(a1)
            self.__add__(a2)
            self.__add__(a3)
            self.stastic2[0]+=3
        elif sid<61:
            trans_1 = tio.transforms.OneOf(self.trans4)
            trans_2 = tio.transfKorms.OneOf(self.trans5)
            img1 =trans_1(img)
            img2 =trans_2(img)
            a1 = (img1.get_data(), sid, slabel, 1)              
            a2 = (img2.get_data(), sid, slabel, 2)
            self.__add__(a1)
            self.__add__(a2)
            self.stastic2[1]+=2
        else:
            transform = tio.transforms.OneOf(self.transform_dict)
            img = transform(img)
            a1 = (img.get_data(), sid, slabel, 1) 
            self.__add__(a1)
            self.stastic2[2]+=1



    def __getitem__(self,index):
        data=self.dataset[index]
        return data
"""

>>>>>>> 3a4a4f2 (20240625-code)

# 测试
if __name__=="__main__":
        

    excel_path="/opt/chenxingru/Pineal_region/Pineal_0410.xlsx/"
    data_path=opt.data_path
    test_data_path = opt.testdata_path


    transform_dict = {
        tio.RandomElasticDeformation(),
        tio.RandomNoise(),
        tio.RandomFlip(),
        tio.RandomBlur(),
        tio.RandomAffine(),
        tio.RandomMotion(),
        tio.RandomSwap()
        }



    data = DIY_Folder(data_path=data_path,transform_dict=transform_dict)
    print("data.form: ",data)
    train_data, valid_data = train_test_split(data,shuffle=True, test_size=0.2,random_state=42)
    test_data = DIY_Folder(data_path=test_data_path, transform_dict=None)

    print("=============================================================")
    id=[[0 for i in range(3)] for j in range(3)]
    count=0
    for i,(img, sid, slabel, smale) in enumerate(train_data):
        print("[Total_data]: i:%d  sid: %d label: %d ,smale: %d "%(i,sid, slabel,smale) )
        count+=1
        if slabel==0: id[0][0]+=1
        elif slabel==1: id[0][1]+=1

    for i,(img, sid, slabel, _) in enumerate(valid_data):
        print("[validate_data]: %d label: %d "%(sid, slabel) )
        count+=1
        if slabel==0: id[1][0]+=1
        elif slabel==1: id[1][1]+=1

    print("id: ",id)
                    
    print("total data : ",len(data))
    print("len(data.dataset): ",len(data.dataset))
    print("count data : ",count)



