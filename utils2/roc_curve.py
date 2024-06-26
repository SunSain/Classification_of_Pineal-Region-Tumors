
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
all_fpr= [np.array([0., 0.2, 0.5, 1.]), np.array([0., 0., 1., 1.]), np.array([0., 0.3, 0.4, 1.])]                                                                                 
all_tpr=  [np.array([0. , 0.5, 1. , 1. ]), np.array([0.8, 0.8, 0.8 , 1. ]), np.array([0.0 , 0.6, 0.5 , 1. ])]                                                                     
all_testauc=  [1.0, 0.8, ]                                                                          
color_box=['b-','r-','y-']
label=["a","b","c"]
plt.figure(figsize=(6,6)) 
for i, path in enumerate(all_fpr):
        
    fold_test_fpr=all_fpr[i]
    fold_test_tpr=all_tpr[i]
    roc_auc = auc(fold_test_fpr, fold_test_tpr)
    print("i roc_auc: ",i,roc_auc)
    plt.plot(fold_test_fpr, fold_test_tpr, color_box[i], label=label[i]+'Test ROC (area = {0:.4f})'.format(roc_auc), lw=2)

plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('Test ROC Curve')
plt.plot([0, 1], [0, 1],'g--')
plt.legend(loc="lower right")
#plt.show()
test_roc_save_path = "/home/chenxr/all_test_auc.png"
plt.savefig(test_roc_save_path)
plt.close()
