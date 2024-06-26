

import numpy as np #机器学习基础包
from sklearn.linear_model import LogisticRegression #逻辑回归算法库
import matplotlib.pyplot as plt #绘图工具包
import matplotlib as mpl #绘图地图包
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


lr = Pipeline([('sc', StandardScaler()),
                        ('clf', LogisticRegression(multi_class="multinomial",solver="newton-cg")) ])

lr.fit(X_train,y_train)
