
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import six
import numpy as np
import os

from radiomics import featureextractor,firstorder, glcm, imageoperations, shape, glrlm, glszm,gldm,ngtdm

#No normalization

class DiyRadiomics():
<<<<<<< HEAD
    def __init__(self,paths): #paths: {sid:{'imgpath': , 'maskpath': }, }
        self.paths =paths
=======
    def __init__(self,paths,savepath): #paths: {sid:{'imgpath': , 'maskpath': }, }
        self.paths =paths
        self.savepath=savepath
>>>>>>> 3a4a4f2 (20240625-code)
        self.settings = {}
        self.settings['binWidth'] = 25
        self.settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
        self.settings['interpolator'] = sitk.sitkBSpline
        self.settings['verbose'] = True
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**self.settings)
        self.features = {} #{sid:[], sid:[]}
        self.sids = [] #病人的id
        self.Allfeatures=[]
<<<<<<< HEAD
=======
        self.empty_box=[]
>>>>>>> 3a4a4f2 (20240625-code)
        self.getFeatures()
    
    def get_features(self):
        return self.features
<<<<<<< HEAD
=======
    
    def get_empty_box(self):
        return self.empty_box
>>>>>>> 3a4a4f2 (20240625-code)

    """
    def norm_features(self):
        if self.norm == True:
            if self.train_std==None:
                mean = np.mean(self.Allfeatures, axis=0)
                std = np.std(self.Allfeatures, axis=0)
            else:
                mean =  self.train_mean
                std = self.train_std
        self.Allfeatures = (self.Allfeatures-mean)/std
        for i, sid in enumerate(self.sids):
            features = {sid: self.Allfeatures[i]}
            self.features.append(features)
    """          

    def get_single_features(self,imgpath,maskpath):#这是抽取radiomics的主要部分
        i=0
        features =[]
        image = sitk.ReadImage(imgpath)
        mask = sitk.ReadImage(maskpath)
        try:
<<<<<<< HEAD
=======
            shapeFeature = shape.RadiomicsShape(image,mask, **self.settings )
            shapeFeature.enableAllFeatures()
            shapeFeature_result = shapeFeature.execute()
            for (key, val) in six.iteritems(shapeFeature_result):
                print('  ', key, ':', val)
                i+=1
                features.append(float(val))
            
>>>>>>> 3a4a4f2 (20240625-code)
            firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **self.settings)
            firstOrderFeatures.enableAllFeatures()
            firstorder_result = firstOrderFeatures.execute()
            glcmFeatures = glcm.RadiomicsGLCM(image, mask,**self.settings)
            glcmFeatures.enableAllFeatures()
            glcm_result = glcmFeatures.execute()

            glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask,**self.settings)
            glrlmFeatures.enableAllFeatures()
            glrlm_result = glrlmFeatures.execute()

            glszmFeatures = glszm.RadiomicsGLSZM(image, mask,**self.settings)
            glszmFeatures.enableAllFeatures()
            glszm_result = glszmFeatures.execute()               

            gldmFeatures = gldm.RadiomicsGLDM(image, mask,**self.settings)
            gldmFeatures.enableAllFeatures()
            gldm_result = gldmFeatures.execute() 
            
            ngtdmFeatures = ngtdm.RadiomicsNGTDM(image, mask,**self.settings)
            ngtdmFeatures.enableAllFeatures()
            ngtdm_result = ngtdmFeatures.execute()                
            
            for (key, val) in six.iteritems(firstorder_result):
                print('  ', key, ':', val)
                i+=1
                features.append(float(val))
            for (key, val) in six.iteritems(glcm_result):
                print('  ', key, ':', val)
                i+=1
                features.append(float(val))
            for (key, val) in six.iteritems(glrlm_result):
                print('  ', key, ':', val)
                i+=1
                features.append(float(val))
            for (key, val) in six.iteritems(glszm_result):
                print('  ', key, ':', val)
                i+=1
                features.append(float(val))
            for (key, val) in six.iteritems(gldm_result):
                print('  ', key, ':', val)
                i+=1
                features.append(float(val))
            for (key, val) in six.iteritems(ngtdm_result):
                print('  ', key, ':', val)
                i+=1
                features.append(float(val))
            """
            featureVector = self.extractor.execute(imgpath, maskpath)
            for featureName in featureVector.keys():
                i+=1
                features.append( featureVector[featureName])
            """
            print("feature_number: ",i," ;features: ",features)
        except:
                print('There is nothing in featureVector! file id: ',sid)
        return features
    
    def getFeatures(self): # No normalization
        empty_box = []
        self.enableFeatures()
        for i,sid in enumerate(self.paths):
            pair_path = self.paths[sid] #获取图像的路径
<<<<<<< HEAD
            
=======
            print("sid: ",sid)
>>>>>>> 3a4a4f2 (20240625-code)
            imgpath = pair_path['imgpath']
            maskpath = pair_path['maskpath']

            try:
                features = self.get_single_features(imgpath, maskpath)
<<<<<<< HEAD
=======

                with open(self.savepath+'/%d.txt'%sid, 'w') as file:
                    file.write(str(features) + '\n')

>>>>>>> 3a4a4f2 (20240625-code)
                self.Allfeatures.append(features)
                self.sids.append(sid)
                
            except:
                print('There is nothing in featureVector! file id: ',sid)
                empty_box.append(sid)
                continue
            
          
        print("self.features: ",self.features)
        for i, sid in enumerate(self.sids):
            #features = {sid: self.Allfeatures[i]}
            self.features[sid] = self.Allfeatures[i]
        print("self.features: ",self.features)
        print("empty box: ",empty_box)
<<<<<<< HEAD
=======
        self.empty_box=empty_box
        
>>>>>>> 3a4a4f2 (20240625-code)


    def enableFeatures(self): #没什么用，后续还是一个一个enable的
        self.extractor.disableAllFeatures()
        
    
if __name__=="__main__":
    img_root="/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/imagesTs0_noSkullStrip/"
    mask_root = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/inferTs_skullStripped/"

    radiomics_pair_path={}
    for i, f in enumerate(sorted(os.listdir(img_root))):
        fn = f.split("_")
        #get label: 0<sid<61: label =0 | 61<sid: label=1
        try:
                sid=int(fn[0])
        except:
            print("it's the_select_file")
            continue

        sub_path = os.path.join(img_root, f)
        pair_file = {'imgpath':sub_path}
        radiomics_pair_path[sid]=pair_file
        
    mask_root = mask_root
    instance_mask_file = sorted(os.listdir(mask_root))
    for i, f in enumerate(instance_mask_file):
        fn = f.split("_")
        try:
                sid=int(fn[0])
        except:
            print("it's the_select_file")
            continue
        sub_path = os.path.join(mask_root, f)
        pair_file = radiomics_pair_path[sid]
        pair_file['maskpath']=sub_path
        radiomics_pair_path[sid] = pair_file
    print('self.radiomics_pair_path: ',radiomics_pair_path)
        
    diybox = DiyRadiomics(radiomics_pair_path)
    