import os
import numpy as np

feature_lists = []  # 这里是100个1*614维特征的list
labels = []  # 对应的类别标签，id<50为0，否则为1

test_feature_lists = []  # 这里是100个1*614维特征的list
test_labels = []  # 对应的类别标签，id<50为0，否则为1
txtroot="/home/chenxr/radiomics/train_T1C/"

train_sids=[]
test_sids=[]

for i, f in enumerate(sorted(os.listdir(txtroot))):
    sid=int(f.split(".")[0])
    with open(txtroot+'/%d.txt'%sid, 'r') as file:
        content = file.read()
        my_list_from_file = eval(content)
        feature_lists.append(my_list_from_file)
    if sid<61:
        labels.append(0)
    else:
        labels.append(1)
    train_sids.append(sid)
        
test_txtroot="/home/chenxr/radiomics/test_T1C/"
for i, f in enumerate(sorted(os.listdir(test_txtroot))):
    sid=int(f.split(".")[0])
    with open(test_txtroot+'/%d.txt'%sid, 'r') as file:
        content = file.read()
        my_list_from_file = eval(content)
        test_feature_lists.append(my_list_from_file)
    if sid<61:
        test_labels.append(0)
    else:
        test_labels.append(1)
    test_sids.append(sid)
    
             
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# 将数据转换为NumPy数组格式
X = np.array(feature_lists)
print("X.shape: ",X.shape)
y = np.where([i < 61 for i in train_sids], [0], [1])

test_X=np.array(test_feature_lists)
test_y= np.where([i < 61 for i in test_sids], [0], [1])
# 初始化Random Forest模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
mean_importance = np.mean(importances)
selected_features = np.where(importances > mean_importance)[0]
X_selected = X[:, selected_features]
print("X_selected.shape: ",X_selected.shape)
print(f"Selected {len(selected_features)} features out of 614 based on mean importance threshold.")

test_selected=test_X[:, selected_features]
print("y.shape: ",y.shape)

from sklearn.model_selection import cross_val_score

# 进行五折交叉验证
from sklearn.ensemble import ExtraTreesClassifier

automl = ExtraTreesClassifier(n_estimators=100, random_state=42)
automl.fit(X_selected, y)

random_state_value = 42
from sklearn.model_selection import cross_val_score, KFold
# 创建分割对象，例如 KFold，并设置随机种子
kf = KFold(n_splits=3, shuffle=True, random_state=random_state_value)
cv_scores = cross_val_score(automl, X_selected, y, cv=kf)
mean_accuracy = cv_scores.mean()
print(f"Validation mean ACC: {mean_accuracy:.4f}")
pred_test=automl.predict(test_selected)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_y, pred_test)
print(f"Test Accuracy: {accuracy}")


from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score, precision_score,f1_score
from sklearn.metrics import make_scorer

auc_scorer = make_scorer(roc_auc_score)
recall_scorer = make_scorer(recall_score)
specificity_scorer = make_scorer(lambda y_true, y_pred: (1 - precision_score(y_true, y_pred, pos_label=0)))


X_test=X_selected
y_test=y
y_pred_prob = automl.predict_proba(X_test)[:, 1]  # 获取正类的概率
y_pred = automl.predict(X_test)  # 获取预测标签

auc = roc_auc_score(y_test, y_pred_prob)

acc = accuracy_score(y_test, y_pred)

sen = recall_score(y_test, y_pred)

spe = precision_score(y_test, y_pred, pos_label=0)
# 计算 F1 分数
f1 = f1_score(y_test, y_pred)
print(f"train AUC1: {auc:.4f}")
print(f"train Accuracy1: {acc:.4f}")
print(f"train Specificity1: {spe:.4f}")
print(f"train Sensitivity1: {sen:.4f}")
print(f"train F1 Score1: {f1:.4f}")

# 执行交叉验证并计算 AUC、敏感度和特异性的均值
cv_auc = cross_val_score(automl, X_selected, y, cv=5, scoring=auc_scorer)
cv_recall = cross_val_score(automl, X_selected, y, cv=5, scoring=recall_scorer)
cv_specificity = cross_val_score(automl, X_selected, y, cv=5, scoring=specificity_scorer)
f1_scores = cross_val_score(automl, X_selected, y, cv=5, scoring='f1')
mean_f1 = f1_scores.mean()

mean_auc = cv_auc.mean()
mean_recall = cv_recall.mean()
mean_specificity = cv_specificity.mean()
print(f"Validate Mean AUC: {mean_auc:.4f}")
print(f"Validate Mean Sensitivity (Recall): {mean_recall:.4f}")
print(f"Validate Mean Specificity: {mean_specificity:.4f}")
print(f"Validate Mean F1 score on cross-validation: {mean_f1:.4f}")

# test metrics
X_test=test_selected
y_test=test_y
y_pred_prob = automl.predict_proba(X_test)[:, 1]  # 获取正类的概率
y_pred = automl.predict(X_test)  # 获取预测标签

auc = roc_auc_score(y_test, y_pred_prob)

acc = accuracy_score(y_test, y_pred)

sen = recall_score(y_test, y_pred)

spe = precision_score(y_test, y_pred, pos_label=0)
# 计算 F1 分数
f1 = f1_score(y_test, y_pred)
print(f"test AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Specificity: {spe:.4f}")
print(f"Sensitivity: {sen:.4f}")
print(f"F1 Score: {f1:.4f}")

