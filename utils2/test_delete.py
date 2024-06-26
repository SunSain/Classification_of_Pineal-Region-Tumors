import numpy as np

def cal_metrics(CM):
    tn=CM[0][0]
    tp=CM[1][1]
    fp=CM[0][1]
    fn=CM[1][0]
    
    
    num = CM.shape[0]
    print("num: ",num)
    print("cm.shape: ",CM.shape)
    print(CM.size)
    print(CM)
    tps = [0 for i in range(num)]
    fps = [0 for i in range(num)]
    fns = [0 for i in range(num)]
    tns = [0 for i in range(num)]
    acc=np.sum(np.diag(CM)/np.sum(CM))
    mets = []
    
    for i in range(num):
        tp_0 = CM[i][i]
        j=0
        fp_0 = sum(CM[j][i] for j in range(num) if  j != i)
        j=0
        fn_0 = sum(CM[i][j] for j in range(num) if j!=i)
        j,k = 0,0
        tn_0 = sum(sum(CM[j][k] for j in range(num) if j!=i)for k in range(num) if k!=i)
        print(i,tn_0)
        sen_0=tp_0/(tp_0+fn_0)
       
        pre_0=tp_0/(tp_0+fp_0)
      
        
        F1_0= (2*sen_0*pre_0)/(sen_0+pre_0)
        
        spe_0 = tn_0/(tn_0+fp_0)
       
        met_0= []
        mets.append(met_0)
    
    pre = sum(mets[i].pre for i in range(num))/num
    sen = sum(mets[i].sen for i in range(num))/num
    spe = sum(mets[i].spe for i in range(num))/num
    F1 = sum(mets[i].F1 for i in range(num))/num
    return acc,pre,sen,spe,F1,mets

CM = np.array([[1,2],[4,5]])
print(cal_metrics(CM))