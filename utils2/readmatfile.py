import scipy.io
import numpy as np
filepath="/opt/chenxingru/opt/Data/0528_ax_T1/001_3_C08A/001_3_C08ADF3CAA0842E888967BDA078D1BF_ax_t1_flair_Turn.mat"
mat = scipy.io.loadmat(filepath)

data = np.array(mat)
print("data: ",data)