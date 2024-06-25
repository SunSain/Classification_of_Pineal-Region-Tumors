import autogluon
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from autogluon.tabular import TabularDataset, TabularPredictor
#train_data = TabularDataset("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/none_composed_0_train.csv")
#/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/
import pandas as pd
import numpy as np
from sklearn.utils import resample
import scipy.stats as stats
import csv

def add_headers(csv_f,new_csv_f,sorted_f,feature_number):#feature_number: 512 or 93

    file = pd.read_csv(csv_f)

    print("csv_f: ",csv_f)
    print("new_csv_f: ",new_csv_f)
    headerList = ['sid']
    if feature_number==512:
        colums=['features_'+str(i+1) for i in range(feature_number)]
    else:
        colums=['radiomics_features_'+str(i+1) for i in range(feature_number)]
    headerList.extend(colums)
    if feature_number==512:
        headerList.append('gender')
        headerList.append('age')
    headerList.append('class')
    print("headerList: ",headerList)
    # converting data frame to csv
    #file.to_csv(new_csv_f, header=headerList, index=False)
    #new_file = pd.read_csv(new_csv_f)
    
    
    with open(new_csv_f, 'w', newline='') as out_f:
        writer = csv.writer(out_f)

        with open(csv_f, newline='') as in_f:
            reader = csv.reader(in_f)

            # Read the first row
            first_row = next(reader)
            # Count the columns in first row; equivalent to your `for i in range(len(first_row)): ...`
            header = headerList

            # Write header and first row
            writer.writerow(header)
            writer.writerow(first_row)

            # Write rest of rows
            for row in reader:
                writer.writerow(row)
    #writer.close()
    new_file = pd.read_csv(new_csv_f)
    #print("new_file: ",new_file)
    new_file=new_file.sort_values(by=["sid"],ascending=[True], inplace=False)
    print("new_file: ",new_file)
    new_file.to_csv(sorted_f, index=False)



def add_radiomics_header():
    add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/all_test.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/new_all_test.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/final/sorted_all_test.csv",feature_number=107)
    add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/all_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/new_all_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/final/sorted_all_train.csv",feature_number=107)
    add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/fold_0_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/new_fold_0_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/final/sorted_fold_0_train.csv",feature_number=107)
    add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/fold_1_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/new_fold_1_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/final/sorted_fold_1_train.csv",feature_number=107)
    add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/fold_2_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/new_fold_2_train.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/final/sorted_fold_2_train.csv",feature_number=107)
    add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/fold_0_valid.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/new_fold_0_valid.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/final/sorted_fold_0_valid.csv",feature_number=107)
    add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/fold_1_valid.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/new_fold_1_valid.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/final/sorted_fold_1_valid.csv",feature_number=107)
    add_headers("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/fold_2_valid.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/new_fold_2_valid.csv","/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/final/sorted_fold_2_valid.csv",feature_number=107)

#add_radiomics_header()

def process(fake_path, final_path,limit=-1,inbox=[]): #delete sid, limit=27
    df = pd.read_csv(fake_path)
    box=[]
    if inbox==[]:
        first_column = df.columns[0]
        for row in range(df.shape[0]):
            if df.iat[row,0]<limit:
                box.append(row)

        df=df.drop(index=box)
        df = df.drop([first_column], axis=1)
        df.to_csv(final_path, index=False)
        return 
    
    origfile= open(fake_path, 'r')
    reader = csv.reader(origfile)
    headers = next(reader)
    
    data=[]
    data_box={}
    unimportant_path="/home/chenxr/test.csv"
    newfile=open(unimportant_path, 'w', newline='') 
    writer = csv.writer(newfile)
    writer.writerow(headers)
    #print("inbox: ",inbox)
    for row in reader:
            if int(row[0]) in inbox:
                #print("add in")
                data_box[int(row[0])]=row
                
    for i in inbox:
        data.append(data_box[i])
    writer.writerows(data)
    
    df = pd.read_csv(unimportant_path)
    first_column = df.columns[0]
    df = df.drop([first_column], axis=1)
    df.to_csv(final_path, index=False)

def process_test(fake_path,limit=-1): #delete sid, limit=27
    df = pd.read_csv(fake_path)
    box=[]
    
    origfile= open(fake_path, 'r')
    reader = csv.reader(origfile)
    headers = next(reader)
    
    data=[]
    data_box={}

    #print("inbox: ",inbox)
    #print("headers: ",headers)
    for row in reader:
            data_box[int(row[0])]=row
    return   headers,data_box

def build_test(headers,data_box,final_path,inbox):
    data=[]
    unimportant_path="/home/chenxr/test.csv"
    newfile=open(unimportant_path, 'w', newline='') 
    writer = csv.writer(newfile)
    writer.writerow(headers)
    print(len(inbox))
    for i in inbox:
        data.append(data_box[i])
    writer.writerows(data)
    newfile.close()
  
    df = pd.read_csv(unimportant_path)
    first_column = df.columns[0]
    df = df.drop([first_column], axis=1)
    df.to_csv(final_path, index=False)

fake_train_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_all_train.csv"
train_path ="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/instrance_sorted_all_train.csv"

fake_test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_all_test.csv"
test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/instance_sorted_all_test.csv"

fake_train_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics_batchavg_constrain/t1c_t1_t2_radiomics_fold_0_train.csv"
train_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics_batchavg_constrain/instance_t1c_t1_t2_radiomics_fold_0_train.csv"
fake_test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics_batchavg_constrain/t1c_t1_t2_radiomics_fold_0_test.csv"
test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics_batchavg_constrain/instance_t1c_t1_t2_radiomics_fold_0_test.csv"
fake_valid_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics_batchavg_constrain/t1c_t1_t2_radiomics_fold_0_valid.csv"
valid_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics_batchavg_constrain/instance_t1c_t1_t2_radiomics_fold_0_valid.csv"

save_path = '"/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics_batchavg_constrain_result_model'  # specifies folder to store trained models
##############


fake_train_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics/t1c_t1_t2_radiomics_fold_0_train.csv"
train_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics/instance_t1c_t1_t2_radiomics_fold_0_train.csv"
fake_test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics/t1c_t1_t2_radiomics_fold_0_test.csv"
test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics/instance_t1c_t1_t2_radiomics_fold_0_test.csv"

fake_valid_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics/t1c_t1_t2_radiomics_fold_0_valid.csv"
valid_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics/instance_t1c_t1_t2_radiomics_fold_0_valid.csv"

save_path = '"/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/none_t1_t1c_t2_radiomics'  # specifies folder to store trained models

fake_train_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics_batchavg_constrain/t1c_t1_t2_radiomics_fold_0_train.csv"
train_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics_batchavg_constrain/instance_t1c_t1_t2_radiomics_fold_0_train.csv"
fake_test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics_batchavg_constrain/t1c_t1_t2_radiomics_fold_0_test.csv"
test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics_batchavg_constrain/instance_t1c_t1_t2_radiomics_fold_0_test.csv"
fake_valid_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics_batchavg_constrain/t1c_t1_t2_radiomics_fold_0_valid.csv"
valid_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics_batchavg_constrain/instance_t1c_t1_t2_radiomics_fold_0_valid.csv"

save_path = '"/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics_batchavg_constrain_result_model'  # specifies folder to store trained models
##############
fake_train_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_fold_0_train.csv"
train_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/instance_sorted_fold_0_train.csv"
fake_test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_all_test.csv"
test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/instance_sorted_all_test.csv"

fake_valid_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_fold_0_valid.csv"
valid_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/instance_sorted_fold_0_valid.csv"

save_path = '"/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/onlyradiomics'  # specifies folder to store trained models

fake_train_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_all_train.csv"
train_path ="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/instrance_sorted_all_train.csv"

fake_test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_all_test.csv"
test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/instance_sorted_all_test.csv"

limit=27


def check_boostrap_test(fake_train_path,train_path,fake_test_path,test_path,fake_valid_path,valid_path):
    ...
    '''80%置信空间估计'''
    iter=1000
    sigma_acc = []
    sigma_auc=[]
    sigma_f1=[]
    test_comman_total_file=[3, 4, 8, 16, 20, 22, 25, 26, 38, 45, 47, 56, 64, 65, 66, 75, 93, 94, 98, 99, 101, 111, 120]
    #fake_train_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_all_train.csv"
    #train_path ="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/instrance_sorted_all_train.csv"
    #fake_test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/sorted_all_test.csv"
    #test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/Radiomics/final/instance_sorted_all_test.csv"

    process(fake_train_path,train_path,inbox=[])
    

    train_data = TabularDataset(train_path)
    subsample_size = 40  # subsample subset of data for faster demo, try setting this to much larger values
    train_data = train_data.sample(n=subsample_size, random_state=0)
    train_data.head()

    try:
        predictor = TabularPredictor(label='class', path=save_path).fit(train_data,num_bag_folds=3)
    except:
        predictor = TabularPredictor(label='class_y', path=save_path).fit(train_data,num_bag_folds=3)


    predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file


  
    for i in range(iter):  #重复实验1000次
        bootstrapSamples = resample(test_comman_total_file, n_samples=int(23*0.8), replace=True)   #每次有放回地抽取100个人
        print("bootstrapSamples: ",bootstrapSamples)
        test_data=bootstrapSamples
        #print("test_data: ",test_data)
        #print("drop_box: ",drop_box)
        if i==0:
            header, databox=process_test(fake_test_path,limit=-1)
        build_test(header,databox,final_path=test_path,inbox=bootstrapSamples)
        
        test_data = TabularDataset(test_path)
        try:
            y_test = test_data['class']  # values to predict
            test_data_nolab = test_data.drop(columns=['class'])  # delete label column to prove we're not cheating
        except:
            y_test = test_data['class_y']  # values to predict
            test_data_nolab = test_data.drop(columns=['class_y'])  # delete label column to prove we're not cheating            

        test_data_nolab.head()

        y_pred = predictor.predict(test_path)
        y_pred_proba=predictor.predict_proba(test_path).iloc[:,1]
        #print("test_Predictions:  \n", y_pred)
        perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

        auc_test = roc_auc_score(y_test, y_pred_proba)
        print("auc_test_real: ",auc_test)
        print("perf_real: ",perf)


        sigma_acc.append(perf['accuracy'])
        sigma_auc.append(auc_test)
        sigma_f1.append(perf['f1'])

    #80%置信空间估计，则计算sigma_iter的(100 - 80) / 2 和 80 + (100 - 80) / 2分位数
    confidence_range = 0.8
    sigma_acc_interval = stats.t.interval(alpha=0.95, df=len(sigma_acc)-1, loc=np.mean(sigma_acc), scale=stats.sem(sigma_acc))
    sigma_auc_interval = stats.t.interval(alpha=0.95, df=len(sigma_auc)-1, loc=np.mean(sigma_auc), scale=stats.sem(sigma_auc))
    sigma_f1_interval = stats.t.interval(alpha=0.95, df=len(sigma_f1)-1, loc=np.mean(sigma_f1), scale=stats.sem(sigma_f1))

    print("sigma_acc: ",sigma_acc,sigma_auc, sigma_f1)
    print("sigma_acc_interval: ",sigma_acc_interval)
    print("sigma_auc_interval: ",sigma_auc_interval)
    print("sigma_f1_interval: ",sigma_f1_interval)
    
    
    process(fake_valid_path,valid_path,inbox=[])
    valid_data = TabularDataset(valid_path)
    try:
        y_valid = valid_data['class']  # values to predict
        valid_data_nolab = valid_data.drop(columns=['class'])  # delete label column to prove we're not cheating
    except:
        y_valid = valid_data['class_y']  # values to predict
        valid_data_nolab = valid_data.drop(columns=['class_y'])  # delete label column to prove we're not cheating            

    valid_data_nolab.head()

    y_pred = predictor.predict(valid_path)
    y_pred_proba=predictor.predict_proba(valid_path).iloc[:,1]
    print("valid_Predictions:  \n", y_pred)
    perf = predictor.evaluate_predictions(y_true=y_valid, y_pred=y_pred, auxiliary_metrics=True)

    auc_test = roc_auc_score(y_valid, y_pred_proba)
    print("auc_valid_real: ",auc_test)
    print("perf_valid: ",perf)


fold = 0
fake_train_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/final/sorted_fold_%d_train.csv"%(fold)
train_path ="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/final/instrance_sorted_fold_%d_train.csv"%(fold)
fake_valid_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/final/sorted_fold_%d_valid.csv"%(fold)
valid_path ="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/final/instrance_sorted_fold_%d_valid.csv"%(fold)

fake_test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/final/sorted_all_test.csv"
test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/all_Radiomics/final/instance_sorted_all_test.csv"

check_boostrap_test(fake_train_path,train_path,fake_test_path,test_path,fake_valid_path,valid_path)

"""
process(fake_train_path,train_path,limit=limit)

train_data = TabularDataset(train_path)
subsample_size = 40  # subsample subset of data for faster demo, try setting this to much larger values
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head()

predictor = TabularPredictor(label='class', path=save_path).fit(train_data,num_bag_folds=3)

predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file



#fake_test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics/t1c_t1_t2_radiomics_fold_2_test.csv"
#test_path="/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/t1c_t1_t2_radiomics/instance_t1c_t1_t2_radiomics_fold_2_test.csv"


process(fake_test_path,test_path,limit=limit)



test_data = TabularDataset(test_path)
y_test = test_data['class']  # values to predict
test_data_nolab = test_data.drop(columns=['class'])  # delete label column to prove we're not cheating
test_data_nolab.head()

y_pred = predictor.predict(test_path)
y_pred_proba=predictor.predict_proba(test_path).iloc[:,1]
print("test_Predictions:  \n", y_pred)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

auc_test = roc_auc_score(y_test, y_pred_proba)
print("auc_test_real: ",auc_test)
print("perf_real: ",perf)



process(fake_valid_path,valid_path,limit=limit)

test_data = TabularDataset(valid_path)
y_test = test_data['class']  # values to predict
test_data_nolab = test_data.drop(columns=['class'])  # delete label column to prove we're not cheating
test_data_nolab.head()

y_pred = predictor.predict(valid_path)
y_pred_proba=predictor.predict_proba(valid_path).iloc[:,1]
print("valid_Predictions:  \n", y_pred)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

auc_test = roc_auc_score(y_test, y_pred_proba)
print("auc_valid: ",auc_test)
print("perf: ",perf)
#model_performance(p_test_AutoGluon, p_train_AutoGluon, p_test_proba_AutoGluon, p_train_proba_AutoGluon, y_test, y_train, 'AutoGluon')
"""

"""
test_data = TabularDataset("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/none_composed_0_valid.csv")
y_test = test_data['class']  # values to predict
test_data_nolab = test_data.drop(columns=['class'])  # delete label column to prove we're not cheating
test_data_nolab.head()

y_pred_proba=predictor.predict_proba("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/none_composed_0_valid.csv").iloc[:,1]
y_pred = predictor.predict("/home/chenxr/Pineal_region/after_12_08/Results/old_feats/united/none_composed_0_valid.csv")
print("valid_Predictions:  \n", y_pred)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

auc_test = roc_auc_score(y_test, y_pred_proba)
print("auc_test: ",auc_test)
print("perf: ",perf)
"""
