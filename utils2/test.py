import numpy as np

img=[[[0,1,2],
     [2,2,2]]]
maskimg=[[[0,0,0],
         [1,1,1]]]
img_t1 = [[[2,3,4],
          [3,3,3]]]
img_t2 = [[[6,5,4],
          [5,5,5]]]
instance_img = np.append(img,maskimg,axis=0)
print("instance_img ",instance_img)
instance_img = np.append(instance_img,img_t1,axis=0)
instance_img = np.append(instance_img,img_t2,axis=0)           
print("instance_img.shape: ",instance_img.shape)
print("instance_img ",instance_img)

a=[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 69, 70, 71, 73, 76, 77, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 95, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]

delete_a = [41, 42, 68, 72, 74, 78, 79, 103]# T1无，T2，T1C都有

delete_b= [3, 4, 8, 16, 20, 22, 
           25, 26, 32, 33, 38, 41, 42, 45,
           47, 56, 64, 65, 66, 68, 72, 74, 75, 78,
           79, 83, 92, 93, 94, 96, 98, 99, 101, 103, 111, 120]# T2有，T1/T1C无
#T1无：3,4,8,16,20,22,25,26,38,41,42,45,47,56,64,65,66,68,72,74,75,78,79,93,94,98,99,101,103,111,120
#T1C无：3,4,8,16,20,22,25,26,,38,45,47,50,53,56,64,65,66,75,83,89,92,93,94,96,98,99,101,111,120
#t1c:32,33,41?,65,83,
#t1:41,45_56_66_89_92_96_111伪影太大,
#T2无： 50,53,89
delete_c = [32, 33, 50, 53, 83, 89, 92, 96]

#无t1: 151_017_wangenzong', '135_019_yangziyue', '140_005_lijinpeng', '138_020_liaocandong', '136_027_zhangdehao', '156_018_guozexu
print("a.length: ",len(a))

"""
new:
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 127, 130, 131, 132, 134, 139, 141, 143, 144, 145, 146, 148, 149, 150, 152, 153]
delete: [41, 53, 89, 103, 147]# t1c有，其它没有

 [32, 33, 50, 53, 83, 89, 126, 128, 129, 133, 135, 136, 137, 138, 142, 147, 154, 155, 156, 157]#t1有，其它没有
 [32, 33, 41, 83, 103,    126, 128, 129, 133, 135, 136, 137, 138, 140, 142, 154, 155, 156, 157]#t2 有，其它没有
 无T1C: 128,129,133,136,137,140,142,154,155,157
 """
 #last: 
a =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 130, 131, 132, 134, 138, 139, 141, 143, 144, 145, 146, 148, 149, 150, 152, 153, 156]

print("ahj: ",len(a))

a=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39, 40,41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53,54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89,90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 130, 131, 132, 134, 138, 139, 141, 143, 144, 145, 146, 148, 149, 150, 152, 153, 156])
print("lema: ",len(a))
pos = [43,53,68,2,38,132,54,100,46,26,1,88,23,64,138,60,131,117,139,73,39,52,18,113,110,31,41,76,108,10]
print("a_5",a[sorted(pos)])
b=a[sorted(pos)]
print("b: ",b)
#needed = [ 11  19  34  42  44  56  64  76  79 112 114 117 121 143 153]
c=[3, 4, 8, 16, 20, 22, 25, 26, 38, 45, 47, 56, 64, 65, 66, 75, 93, 94, 98, 99, 101, 111, 120]
jx=[]
for i, id in enumerate(b):
     if id in c:
          jx.append(id)
print("jx: ",jx,len(jx))
[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127, 130, 131, 132, 134, 138, 139, 141, 143, 144, 145, 146, 148, 149, 150, 152, 153, 156]

#delete_box:  [41, 53, 89, 103, 147]