import SimpleITK as sitk
import radiomics
from radiomics import featureextractor,firstorder, glcm, imageoperations, shape, glrlm, glszm
import six

imageName= "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/imagesTs/001_4_0000.nii.gz"
maskName = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/masked_bounding/001_4.nii.gz" 
print('imageName, maskName', imageName, maskName)
if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
  print('Error getting testcase!')
  exit()

# Define settings for signature calculation
# These are currently set equal to the respective default values
settings = {}
settings['binWidth'] = 25
settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
settings['interpolator'] = sitk.sitkBSpline
settings['verbose'] = True

#Initialize feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
#By default, only original is enabled. Optionally enable some image types:
#extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})
# 所有类型
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('shape')
extractor.enableFeatureClassByName('firstorder')
"""
extractor.enableFeatureClassByName('glcm')
extractor.enableFeatureClassByName('glrlm')
extractor.enableFeatureClassByName('glszm')
extractor.enableFeatureClassByName('gldm')
extractor.enableFeatureClassByName('ngtdm')
# 指定使用LoG和Wavelet滤波器
extractor.enableImageTypeByName('LoG')
extractor.enableImageTypeByName('Wavelet')
"""


print("Calculating features")
featureVector = extractor.execute(imageName, maskName)

image = sitk.ReadImage(imageName)
mask = sitk.ReadImage(maskName)
firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **settings)
firstOrderFeatures.enableAllFeatures()

#Enable all features in firstorder
#extractor.enableFeatureClassByName('firstorder')
#Only enable mean and skewness in firstorder
print("firstoder")
print('Will calculate the following first order features: ')
for f in firstOrderFeatures.enabledFeatures.keys():
  print(f)
  print(getattr(firstOrderFeatures, 'get%sFeatureValue' % f).__doc__)

print('Calculating first order features...',)
result = firstOrderFeatures.execute()
print('done')

print('Calculated first order features: ')
for (key, val) in six.iteritems(result):
  print('  ', key, ':', val)
#extractor.enableFeaturesByName(firstorder=['Mean', 'Skewness'])

featureVector = extractor.execute(imageName, maskName)
i=0

for featureName in featureVector.keys():
  i+=1
  print("%s: %s" % (featureName, featureVector[featureName]))
print("i: ",i)





