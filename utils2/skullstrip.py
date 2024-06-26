from pyrobex.robex import robex
import os
import nibabel as nib

def strip_single(filepath, saveroot,f):
    img = nib.load(filepath)
    stripped, mask = robex(img)
    #another_img = nib.Nifti1Image(stripped, img.affine,img.header)
    savepath =  os.path.join(saveroot,f)
    nib.save(stripped,savepath)
    
    
if __name__=="__main__":
    root = "/home/chenxr/all_T1C_cp/all_T1C/"
    saveroot = "/home/chenxr/all_T1C_cp/all_T1C_skullstripped/"
    if not os.path.exists(saveroot):
        os.mkdir(saveroot)
    subfns = sorted(os.listdir(root))
    for i, f in enumerate(subfns):
        filepath = os.path.join(root,f)
        print("f:",f)
        strip_single(filepath, saveroot, f)
        print("strip 6666")

            
            
    