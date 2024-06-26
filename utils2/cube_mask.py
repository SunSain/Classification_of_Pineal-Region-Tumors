<<<<<<< HEAD
import os
import torch,random
import nibabel as nib
import numpy as np
from random import randint, seed 

def cube_mask(im,number_cube_range=[100,120], cube_size=[20,50]): # im=[img, subimg], img is numpy.array,subimg may be None
    #instance_img={"img":torch.from_numpy(img), "secondimg":torch.from_numpy(maskimg),"noSameR":self.noSameRM}
    image = im[0]
    secondimg = im[1]
    image = np.expand_dims(image, axis=0)
    secondimg = np.expand_dims(secondimg, axis=0)
    iterations = random.randint(number_cube_range[0], number_cube_range[1])
    masked_img = image
    second_mask_img = secondimg
    print("image.shape: ",image.shape)
    mask = np.zeros(image.shape)
    #print("iterations: ",iterations)
    for i in range(iterations):
        random_center_x, random_center_y, random_center_z = randint(0, 22), randint(10, 390), randint(10, 390) #23 400 400
        random_width, random_height, random_deep = randint(2,8), randint(cube_size[0],cube_size[1]), randint(cube_size[0],cube_size[1])
        #print("random_center_x, random_center_y, random_center_z: ",random_center_x, random_center_y, random_center_z)
        #print("random_width, random_height, random_deep: ",random_width, random_height, random_deep)
        mask = cube(image.shape,(random_width, random_height, random_deep), (random_center_x, random_center_y, random_center_z))
        #print("mask0: ",np.sum(mask))
        mask = 1 - mask
        masked_img = masked_img * mask
        second_mask_img = second_mask_img*mask
    result = np.append(masked_img,second_mask_img,axis=0)
    return torch.from_numpy(result).type(torch.FloatTensor)

def ratio_cube_mask(im,number_cube_range=[100,120], cube_size=[20,50]): # im=[img, subimg], img is numpy.array,subimg may be None
    #instance_img={"img":torch.from_numpy(img), "secondimg":torch.from_numpy(maskimg),"noSameR":self.noSameRM}
    ratio=3
    image = im[0]
    secondimg = im[1]
    
    image = np.expand_dims(image, axis=0)
    secondimg = np.expand_dims(secondimg, axis=0)
    iterations = random.randint(number_cube_range[0], number_cube_range[1])
    masked_img = image
    second_mask_img = secondimg
    print("image.shape: ",image.shape)
    mask = np.zeros(image.shape)
    #print("iterations: ",iterations)
    for i in range(iterations):
        count_time=0
        while(count_time< ratio):
            random_center_x, random_center_y, random_center_z = randint(0, 22), randint(10, 390), randint(10, 390) #23 400 400
            if secondimg[0][random_center_x][random_center_y][random_center_z]==0 :break
            count_time+=1
        print("count_time: ",count_time, " ;secondimg[center]: ",secondimg[0][random_center_x][random_center_y][random_center_z])
        random_width, random_height, random_deep = randint(2,8), randint(cube_size[0],cube_size[1]), randint(cube_size[0],cube_size[1])
        #print("random_center_x, random_center_y, random_center_z: ",random_center_x, random_center_y, random_center_z)
        #print("random_width, random_height, random_deep: ",random_width, random_height, random_deep)
        mask = cube(image.shape,(random_width, random_height, random_deep), (random_center_x, random_center_y, random_center_z))
        #print("mask0: ",np.sum(mask))
        mask = 1 - mask
        masked_img = masked_img * mask
        second_mask_img = second_mask_img*mask
    result = np.append(masked_img,second_mask_img,axis=0)
    return torch.from_numpy(result).type(torch.FloatTensor)


def cube(img_shape, mask_shape, position):
    # print("running here")
    mask = np.zeros(img_shape)
    # print('mask shape', mask.shape)
    width, height, deep = mask_shape[0], mask_shape[1], mask_shape[2]
    center_x, center_y, center_z = position[0], position[1], position[2]
    # 实际上输入的图像是四维的，原始代码中生成mask是按照三维的来生成的，因此在赋值上出现了一些问题，导致最后输出的mask_img和输入的img一样
    mask[ :, center_x - width // 2 : center_x + width // 2
        , center_y - height // 2: center_y + height // 2
        , center_z - deep // 2 : center_z + deep // 2] = 1
    # print('mask.sum',mask.sum())
    return mask

# img = np.ones((2,91,109,91))
# print(img.sum(axis=1).sum(axis=1).sum(axis=1))
# masked_img = cube_mask(img)
# print(masked_img.sum(axis=1).sum(axis=1).sum(axis=1))
=======
import os
import torch,random
import nibabel as nib
import numpy as np
from random import randint, seed 

def cube_mask(im,number_cube_range=[100,120], cube_size=[20,50]): # im=[img, subimg], img is numpy.array,subimg may be None
    #instance_img={"img":torch.from_numpy(img), "secondimg":torch.from_numpy(maskimg),"noSameR":self.noSameRM}
    image = im[0]
    secondimg = im[1]
    image = np.expand_dims(image, axis=0)
    secondimg = np.expand_dims(secondimg, axis=0)
    iterations = random.randint(number_cube_range[0], number_cube_range[1])
    masked_img = image
    second_mask_img = secondimg
    print("image.shape: ",image.shape)
    mask = np.zeros(image.shape)
    #print("iterations: ",iterations)
    for i in range(iterations):
        random_center_x, random_center_y, random_center_z = randint(0, 22), randint(10, 390), randint(10, 390) #23 400 400
        random_width, random_height, random_deep = randint(2,8), randint(cube_size[0],cube_size[1]), randint(cube_size[0],cube_size[1])
        #print("random_center_x, random_center_y, random_center_z: ",random_center_x, random_center_y, random_center_z)
        #print("random_width, random_height, random_deep: ",random_width, random_height, random_deep)
        mask = cube(image.shape,(random_width, random_height, random_deep), (random_center_x, random_center_y, random_center_z))
        #print("mask0: ",np.sum(mask))
        mask = 1 - mask
        masked_img = masked_img * mask
        second_mask_img = second_mask_img*mask
    result = np.append(masked_img,second_mask_img,axis=0)
    return torch.from_numpy(result).type(torch.FloatTensor)

#def ratio_cube_mask(im,number_cube_range=[100,120], cube_size=[20,50]): # im=[img, subimg], img is numpy.array,subimg may be None
def ratio_cube_mask(im,number_cube_range=[1,10], cube_size=[20,50]): # im=[img, subimg], img is numpy.array,subimg may be None
#def ratio_cube_mask(im,number_cube_range=[20,50], cube_size=[20,50]): # im=[img, subimg], img is numpy.array,subimg may be None
#def ratio_cube_mask(im,number_cube_range=[70,100], cube_size=[20,50]): # im=[img, subimg], img is numpy.array,subimg may be None
    #instance_img={"img":torch.from_numpy(img), "secondimg":torch.from_numpy(maskimg),"noSameR":self.noSameRM}
    ratio=3
    image = im[0].numpy()
    #secondimg = im[1]
    
    image = np.expand_dims(image, axis=0)
    #secondimg = np.expand_dims(secondimg, axis=0)
    iterations = random.randint(number_cube_range[0], number_cube_range[1])
    #masked_img = image
   
    masked_im_all = im.numpy()
    #second_mask_img = secondimg
    
    print("image.shape: ",image.shape)
    
    mask = np.zeros(image.shape)
    #print("iterations: ",iterations)
    for i in range(iterations):
        count_time=0
        while(count_time< ratio): #ratio=0.25
            random_center_x, random_center_y, random_center_z = randint(0, 22), randint(10, 390), randint(10, 390) #23 400 400
            #if secondimg[0][random_center_x][random_center_y][random_center_z]==0 :break
            #if (random_center_x<=4 or random_center_x>=18) and (random_center_y<=100 or random_center_y>=300) and (random_center_z<=100 or random_center_z>=300):break
            count_time+=1
        #print("count_time: ",count_time, " ;secondimg[center]: ",secondimg[0][random_center_x][random_center_y][random_center_z])
        random_width, random_height, random_deep = randint(2,8), randint(cube_size[0],cube_size[1]), randint(cube_size[0],cube_size[1])
        #print("random_center_x, random_center_y, random_center_z: ",random_center_x, random_center_y, random_center_z)
        #print("random_width, random_height, random_deep: ",random_width, random_height, random_deep)
        mask = cube(image.shape,(random_width, random_height, random_deep), (random_center_x, random_center_y, random_center_z))
        print("mask_shape: ",mask.shape)
        mask = 1 - mask
        #masked_img = masked_img * mask
        #second_mask_img = second_mask_img*mask
        for j in range(im.shape[0]):
            masked_im_all[j] = masked_im_all[j]*mask[0]
            print("masked_im_all[j].shape: ",masked_im_all[j].shape)
    #result = np.append(masked_img,second_mask_img,axis=0)
    result = masked_im_all
    #return result
    return torch.from_numpy(result).type(torch.FloatTensor)

#valid_cp_
def valid_cp_ratio_cube_mask(im,number_cube_range=[1,10], cube_size=[20,50]): # im=[img, subimg], img is numpy.array,subimg may be None
    ratio=5
    image = im[0].numpy()
    #secondimg = im[1]
    
    image = np.expand_dims(image, axis=0)
    #secondimg = np.expand_dims(secondimg, axis=0)
    iterations = random.randint(number_cube_range[0], number_cube_range[1])
    #masked_img = image
   
    masked_im_all = im.numpy()
    #second_mask_img = secondimg
    
    print("image.shape: ",image.shape)
    
    mask = np.zeros(image.shape)
    #print("iterations: ",iterations)
    for i in range(iterations):
        count_time=0
        while(count_time< ratio): #ratio=0.25
            random_center_x, random_center_y, random_center_z = randint(0, 22), randint(10, 390), randint(10, 390) #23 400 400
            #if secondimg[0][random_center_x][random_center_y][random_center_z]==0 :break
            if (random_center_x<=4 or random_center_x>18) or ((random_center_y<=50 or random_center_y>=350) and (random_center_z<=50 or random_center_z>=350)):break
            #count_time+=1
        #print("count_time: ",count_time, " ;secondimg[center]: ",secondimg[0][random_center_x][random_center_y][random_center_z])
        random_width, random_height, random_deep = randint(1,2), randint(cube_size[0],cube_size[1]), randint(cube_size[0],cube_size[1])
        print("iterations: ",iterations)
        print("random_center_x, random_center_y, random_center_z: ",random_center_x, random_center_y, random_center_z)
        print("random_width, random_height, random_deep: ",random_width, random_height, random_deep)
        mask = cube(image.shape,(random_width, random_height, random_deep), (random_center_x, random_center_y, random_center_z))
        print("mask_shape: ",mask.shape)
        mask = 1 - mask
        #masked_img = masked_img * mask
        #second_mask_img = second_mask_img*mask
        for j in range(im.shape[0]):
            masked_im_all[j] = masked_im_all[j]*mask[0]
            print("masked_im_all[j].shape: ",masked_im_all[j].shape)
    #result = np.append(masked_img,second_mask_img,axis=0)
    result = masked_im_all
    #return result
    return torch.from_numpy(result).type(torch.FloatTensor)
def cp_ratio_cube_mask(im,number_cube_range=[3,10], cube_size=[20,50]): # im=[img, subimg], img is numpy.array,subimg may be None
    ratio=5
    image = im[0].numpy()
    #secondimg = im[1]
    
    image = np.expand_dims(image, axis=0)
    #secondimg = np.expand_dims(secondimg, axis=0)
    iterations = random.randint(number_cube_range[0], number_cube_range[1])
    #masked_img = image
   
    masked_im_all = im.numpy()
    #second_mask_img = secondimg
    
    print("image.shape: ",image.shape)
    
    mask = np.zeros(image.shape)
    #print("iterations: ",iterations)
    for i in range(iterations):
        count_time=0
        while(count_time< ratio): #ratio=0.25
            random_center_x, random_center_y, random_center_z = randint(0, 22), randint(10, 390), randint(10, 390) #23 400 400
            #if secondimg[0][random_center_x][random_center_y][random_center_z]==0 :break
            if (random_center_x<=4 or random_center_x>=18) and (random_center_y<=100 or random_center_y>=300) and (random_center_z<=100 or random_center_z>=300):break
            count_time+=1
        #print("count_time: ",count_time, " ;secondimg[center]: ",secondimg[0][random_center_x][random_center_y][random_center_z])
        random_width, random_height, random_deep = randint(2,8), randint(cube_size[0],cube_size[1]), randint(cube_size[0],cube_size[1])
        #print("random_center_x, random_center_y, random_center_z: ",random_center_x, random_center_y, random_center_z)
        #print("random_width, random_height, random_deep: ",random_width, random_height, random_deep)
        mask = cube(image.shape,(random_width, random_height, random_deep), (random_center_x, random_center_y, random_center_z))
        print("mask_shape: ",mask.shape)
        mask = 1 - mask
        #masked_img = masked_img * mask
        #second_mask_img = second_mask_img*mask
        for j in range(im.shape[0]):
            masked_im_all[j] = masked_im_all[j]*mask[0]
            print("masked_im_all[j].shape: ",masked_im_all[j].shape)
    #result = np.append(masked_img,second_mask_img,axis=0)
    result = masked_im_all
    #return result
    return torch.from_numpy(result).type(torch.FloatTensor)



def cube(img_shape, mask_shape, position):
    # print("running here")
    mask = np.zeros(img_shape)
    # print('mask shape', mask.shape)
    width, height, deep = mask_shape[0], mask_shape[1], mask_shape[2]
    center_x, center_y, center_z = position[0], position[1], position[2]
    # 实际上输入的图像是四维的，原始代码中生成mask是按照三维的来生成的，因此在赋值上出现了一些问题，导致最后输出的mask_img和输入的img一样
    mask[ :, center_x - width // 2 : center_x + width // 2
        , center_y - height // 2
        : center_y + height // 2
        , center_z - deep // 2 : center_z + deep // 2] = 1
    # print('mask.sum',mask.sum())
    return mask

# img = np.ones((2,91,109,91))
# print(img.sum(axis=1).sum(axis=1).sum(axis=1))
# masked_img = cube_mask(img)
# print(masked_img.sum(axis=1).sum(axis=1).sum(axis=1))
>>>>>>> 3a4a4f2 (20240625-code)
# # print(masked_img.min())