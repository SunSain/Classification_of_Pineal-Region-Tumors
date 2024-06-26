
import SimpleITK as sitk
import sys

def main(args):
    if len(args) < 2:
        print(
            "Usage: N4BiasFieldCorrection inputImage "
            + "outputImage [shrinkFactor] [maskImage] [numberOfIterations] "
            + "[numberOfFittingLevels]"
        )
        sys.exit(1)
    print("args[1]: ",args[1])
    path="/home/chenxr/sub-OAS314/sub-OAS31472_sess-d0045_T1w.nii.gz-lin.nii.gz"
    inputImage = sitk.ReadImage(path, sitk.sitkFloat32)
    image = inputImage

    if len(args) > 4:
        maskImage = sitk.ReadImage(args[4], sitk.sitkUInt8)
    else:
        maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

    shrinkFactor = 1
    if len(args) > 3:
        shrinkFactor = int(args[3])
        if shrinkFactor > 1:
            image = sitk.Shrink(
                inputImage, [shrinkFactor] * inputImage.GetDimension()
            )
            maskImage = sitk.Shrink(
                maskImage, [shrinkFactor] * inputImage.GetDimension()
            )

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    numberFittingLevels = 4

    if len(args) > 6:
        numberFittingLevels = int(args[6])

    if len(args) > 5:
        corrector.SetMaximumNumberOfIterations(
            [int(args[5])] * numberFittingLevels
        )

    corrected_image = corrector.Execute(image, maskImage)

    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)

    corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)

    sitk.WriteImage(corrected_image_full_resolution, args[2])

    if shrinkFactor > 1:
        sitk.WriteImage(
            corrected_image, "Python-Example-N4BiasFieldCorrection-shrunk.nrrd"
        )

    return_images = {"input_image": inputImage,
                     "mask_image": maskImage,
                     "log_bias_field": log_bias_field,
                     "corrected_image": corrected_image}
    return return_images

if __name__=="__main__":
    image = "/home/chenxr/sub-OAS314/sub-OAS31472_sess-d0045_T1w.nii.gz-lin.nii.gz"
    maskimage=""
    args=[maskimage,image]
    output=main(args)
    print("output: ",output)

