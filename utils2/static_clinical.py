import os
import re
import numpy as np
import shutil

def counting_gender(txtfile):
    """_summary_

    Args:
        统计临床信息
        age: 对于年龄为几个月的病人，月数/12 从而统一单位为“岁”
        gender: 男性, 女性
        
        使用sid区分肿瘤的类别,sid是病人文件的编号
            类别0: sid为 1-60, 123-130
            类别1(germinoma): sid为 61-122, 131-156
    """
    gender_0=[] #male
    gender_1=[]
    ages={}

    with open(txtfile) as f:
        lines = f.readlines()
        i=0
        for line in lines:
            if i==0: 
                i+=1
                continue
            else: i+=1
            reg = r"[\t,' ',\n]+"
            row = re.split(reg,line)
            print("row: ",row)
            if row==None or row[0]=='': continue
            print("row2: ",row)
            try:
                sid = int(row[0])
            except:
                print("it's not valid row: ",row)
                continue
            gender = row[3]
            age = int(re.findall(r'\d+', row[4])[0])
            unit = re.findall(r'[\u4e00-\u9fa5]', row[4])[0]
            print("gender: ",gender)
            if gender=='男':
                gender=0.
                gender_0.append(sid)
            else:
                gender=1.
                gender_1.append(sid)
            if unit == '月':
                age = age/12.0
            print("age: ",age)
            ages[sid]=age
    print("\n男:女: ",len(gender_0),':',len(gender_1))
    print("\nsid<=122 男:女: ",len([a for a in gender_0 if a<=122 ]),':',len([a for a in gender_1 if a<=122 ]))   
    print("\nbinary_label_0 男:女: ",len([a for a in gender_0 if a<61 ]),':',len([a for a in gender_1 if a<61 ]))   
    print("\nbinary_label_1 男:女: ",len([a for a in gender_0 if a<=122 and a>60 ]),':',len([a for a in gender_1 if a<=122 and a>60]))   
    print("\nAge:")
    #print("label_0 avg age: ",np.mean(np.array([ages[a] for a in ages.keys() if a<61])), max(min(ages[a]  for a in ages.keys() if a<61)))
    mean_age_label_0=np.mean(np.array([ages[a] for a in ages.keys() if a<61]))
    min_age_label_0=min(ages[a]  for a in ages.keys() if a<61)
    max_age_label_0=max(ages[a]  for a in ages.keys() if a<61)
    print("label_0 avg age: ",mean_age_label_0)
    print("label_0 +- age: ",mean_age_label_0-min_age_label_0,max_age_label_0-mean_age_label_0,max(mean_age_label_0-min_age_label_0,max_age_label_0-mean_age_label_0))
    
    
    
    mean_age_label_1=np.mean(np.array([ages[a] for a in ages.keys() if a<=122 and a>60]))
    min_age_label_1=min(ages[a]  for a in ages.keys() if a<=122 and a>60)
    max_age_label_1=max(ages[a]  for a in ages.keys() if a<=122 and a>60)
    print("label_1 avg age: ",mean_age_label_1)
    print("label_1 +- age: ",mean_age_label_1-min_age_label_1,max_age_label_1-mean_age_label_1,max(mean_age_label_1-min_age_label_1,max_age_label_1-mean_age_label_1))

    print("Total<123 avg age: ",np.mean(np.array([ages[a] for a in ages.keys() if a<=122])))
    mean_age_label_1=np.mean(np.array([ages[a] for a in ages.keys() if a<=122]))
    min_age_label_1=min(ages[a]  for a in ages.keys() if a<=122)
    max_age_label_1=max(ages[a]  for a in ages.keys() if a<=122)
    print("total avg age: ",mean_age_label_1)
    print("total +- age: ",mean_age_label_1-min_age_label_1,max_age_label_1-mean_age_label_1,max(mean_age_label_1-min_age_label_1,max_age_label_1-mean_age_label_1))

    #print("男sid: ",gender_0)
    #print("女sid: ",gender_1)


if __name__=="__main__":

    pineal_txt="/home/chenxr/pineal_data_statistic_final/pineal_csv.txt"
    counting_gender(pineal_txt)
