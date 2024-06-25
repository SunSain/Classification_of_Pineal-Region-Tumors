
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
from utils2.cube_mask import cube_mask

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
        self.files=[]
        self.preprocessor = Preprocessor(target_spacing=[0.45,0.45,6.5])
        self.saveroot = opt.save_nii_path
        if not os.path.exists(self.saveroot):
            os.mkdir(self.saveroot)
        self.pre_aug() 


        
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
                slabel = 1
                self.y.append(1)
                self.stastic[2]+=1

            sub_path = os.path.join(self.root, f)
            self.files.append(f)
            img, brain_mask_img = self.preprocessor.run(img_path=sub_path)
            print("before_img.data: ",img.GetSize(),img.GetSpacing())

            img_data=sitk.GetArrayFromImage(img)
            mask_data = sitk.GetArrayFromImage(brain_mask_img)
            brain_mask = mask_data > 0
            brain_mean = img_data[brain_mask].mean()
            brain_std = img_data[brain_mask].std()
            img_data = (img_data - brain_mean) / brain_std
            
            print("img.mean: ",img_data.mean())
            print("img_brain.mean: ",brain_mean)
            img= array2image(img_data,img)
            print("new_img.data:  ",img.GetSize(),img.GetSpacing())

            croporpad = tio.CropOrPad(
            (400,400,23),
            mask_name='brain_mask',
            )
            img = croporpad(img)
            print("cropped_img.data:  ",img.GetSize(),img.GetSpacing())

            
            #save_path  = os.path.join(self.saveroot,f)
            #sitk.WriteImage(img,save_path)
            """
            #——————————crop and Normalization————————
            print("sid: ",sid,"img.shape: ",img.get_data().shape)
            #tiocrop=tio.transforms.CropOrPad([75,85,75],  'linear')
            #img=tiocrop(img)
            Ztransform= tio.ZNormalization()
            img = Ztransform(img)
            #——————————————————
            """
            
            original_data = (img, sid, slabel, 0)
            self.__add__(original_data)
        

    def __len__(self):
        return len(self.dataset)


    def __add__(self,other):
        self.dataset.append(other)
        
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

    #数据增强(aug=True). 如果aug =False则只是从原来的 NII_img拿出img.get_data(),作为输入模型的数据
    def prepro_aug(self, data_idx=[], aug=True, transform_dict=None, aug_form="Random_mask"):
        subdata=[]
        suby=[]
        
        for j, idx in enumerate(data_idx):
            (origin_img,sid,slabel,_) = self.dataset[idx]
            print("j, idx,sid, label: ",j, idx,sid,slabel)


            img = sitk.GetArrayFromImage(origin_img).astype(np.float32)
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img, dtype= np.float32)
            img = torch.from_numpy(img).type(torch.FloatTensor)

            masked_img = tio.Lambda(cube_mask)(img).type(torch.FloatTensor)


            print('img_sum',img.sum(),'img_min',img.min(),'img_max',img.max())
            print('masked_img_sum',masked_img.sum(),'masked_img_min',masked_img.min(),'masked_img_max',masked_img.max())


            a1 = (img, masked_img, sid, slabel, 1)
            subdata.append(a1)
            suby.append(slabel) 

            img= array2image(masked_img[0],origin_img)
            save_path  = os.path.join(self.saveroot,self.files[j])
            sitk.WriteImage(img,save_path)
        
            
        return subdata

    def __getitem__(self,index):
        """
        (img, sid, slabel,copyid)=self.dataset[index]
        
        padded_dict = self._pad_or_crop_to_img_size(img, self.img_size)
        
        img = padded_dict["padded_img"]
        """
        
        
        data = self.dataset[index]
        
        
        
        return data

    def gety(self):
        return self.y




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
    test_data = test_data.prepro_aug(data_idx=[i for i in range(len(test_data))], aug=False, transform_dict=None)


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



