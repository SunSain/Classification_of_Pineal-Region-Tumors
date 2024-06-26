import numpy as np
"""
aa = np.array([2.,3.,9.,6,8])
bb = np.array([5.,6.,3.,7,9])
cc = np.array([aa, bb])
print(cc)

cc_mean = np.mean(cc, axis=0)  #axis=0,表示按列求均值 ——— 即第一维，每一列可看做一个维度或者特征
cc_std = np.std(cc, axis=0)
cc_zscore = (cc-cc_mean)/cc_std   #直接计算，对数组进行标准化，一定要注意维度
print(cc_mean)
print(cc_std)
print(cc_zscore)
"""

a =np.array(262.1)
print("a: ",a)
b = float(a)
print("b:",b)