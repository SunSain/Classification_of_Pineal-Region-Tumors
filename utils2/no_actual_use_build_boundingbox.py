
import numpy as np
import SimpleITK as sitk

class build_boundingbox():
    def __init__(self, cpm):
        self.cpm = cpm
        self.boxes = []
        
    def build_new_box(self,ri, down, up,new_box):
        mark = 0
        (i,j,k) = ri
        ri_box, down_box,up_box =[],[],[]
        index_ri, index_down, index_up = 0,0,0
        if i>=self.cpm.shape[0] or j>=self.cpm.shape[1] or k>=self.cpm.shape[2]:
            print("ri out of bound")
        else :
            if self.cpm[i][j][k] > 0:
                print("(i,j,k): ",(i,j,k),self.cpm[i][j][k])
                mark=1
                ri_box.append(ri)
                ri_box, index_ri = self.build_new_box((i,j,k+1),(i,j+1,k),(i+1,j,k),new_box)
                self.cpm[i][j][k] = 0
            
        (i,j,k) = down
        if i>=self.cpm.shape[0] or j>=self.cpm.shape[1] or k>=self.cpm.shape[2]:
            print("down out of bound")
        else :
            if self.cpm[i][j][k] > 0:
                print("(i,j,k): ",(i,j,k),self.cpm[i][j][k])
                mark = 1
                down_box.append(down)
                down_box, index_down = self.build_new_box((i,j,k+1),(i,j+1,k),(i+1,j,k),new_box)
                self.cpm[i][j][k] = 0

        (i,j,k) = up
        if i>=self.cpm.shape[0] or j>=self.cpm.shape[1] or k>=self.cpm.shape[2]:
           print("up out of bound")
        else: 
            if self.cpm[i][j][k] > 0:
                print("(i,j,k): ",(i,j,k),self.cpm[i][j][k])
                mark = 1
                up_box.append(up)
                up_box, index_up = self.build_new_box((i,j,k+1),(i,j+1,k),(i+1,j,k),new_box)
                self.cpm[i][j][k] = 0
        print("ri, down, up :",ri,down, up)
        print("max(mark,index): ",max(mark,index_ri,index_down,index_up))
        new_box.extend(ri_box)
        new_box.extend(down_box)
        new_box.extend(up_box)
        new_box = list(set(new_box))
        return new_box, max(mark,index_ri,index_down,index_up)
            

    def find_area(self):
        print("cpm.shape: ",self.cpm.shape[0],self.cpm.shape[1],self.cpm.shape[2])
        mark = 1
        while(1):
            if mark == 0:
                break
            mark = 0
            for i in range(self.cpm.shape[0]):
                for j in range(self.cpm.shape[1]):
                    for k in range(self.cpm.shape[2]):
                        
                        if self.cpm[i][j][k] <= 0:
                            continue
                        else:
                            print("(i,j,k): ",(i,j,k),self.cpm[i][j][k])
                            mark = 1
                            new_box =[]
                            new_box.append((i,j,k))
                            new_box, submark = self.build_new_box((i,j,k+1),(i,j+1,k),(i+1,j,k),new_box)
                            self.cpm[i][j][k] = 0
                            if submark == 1:
                                self.boxes.append(new_box)
                            print("mark: ",mark)
                
        print("boxes: ",self.boxes)
        print("boxes.length: ",len(self.boxes))

if __name__ =="__main__":
    path = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/segmentation/DATASET/data_raw/nnUNet_raw_data/Task009_myT1c/inferTs/001_4.nii.gz"
    img = sitk.ReadImage(path)
    imag_array = sitk.GetArrayFromImage(img).astype(np.float32)
    cmp = np.array(imag_array)
    bb =  build_boundingbox(cmp)
    bb.find_area()
